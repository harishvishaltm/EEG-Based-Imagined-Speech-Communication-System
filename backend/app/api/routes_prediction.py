"""Prediction Endpoints for EEG File Upload."""
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.config import settings
from backend.app.schemas.prediction import PredictionResponse, VectorPredictionRequest
from backend.app.services.inference_service import inference_service

logger = logging.getLogger("routes_prediction")
router = APIRouter(prefix="/api", tags=["Prediction"])

ALLOWED_EXTENSIONS = {".npz", ".mat", ".cnt"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Upload EEG recording and run imagined speech inference",
)
async def predict_eeg(
    file: UploadFile = File(..., description="Pre-recorded EEG file (.npz, .mat, or .cnt)"),
    dataset: Optional[str] = Form("kara_one", description="Dataset identifier (kara_one, feis, nguyen)"),
    subject: Optional[str] = Form("MM05", description="Subject identifier (e.g. MM05)"),
):
    """Executes the full imagined speech communication pipeline:
    EEG Upload -> Preprocess & Harmonize -> Model Inference -> LLM Sentence -> TTS Audio.

    STRICT INFERENCE: Zero retraining occurs during this request.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided in upload."
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: {sorted(list(ALLOWED_EXTENSIONS))}",
        )

    # Validate file size limit
    file.file.seek(0, 2)
    file_size_bytes = file.file.tell()
    file.file.seek(0)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    if file_size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    # Save to safe temporary file
    temp_dir = tempfile.mkdtemp(prefix="eeg_upload_")
    temp_path = Path(temp_dir) / f"input{file_ext}"

    try:
        with open(temp_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        # Run inference pipeline
        result = inference_service.run_pipeline(
            file_path=temp_path,
            dataset_name=dataset.lower() if dataset else "kara_one",
            subject_name=subject,
            original_filename=file.filename,
        )
        return result

    except ValueError as ve:
        logger.warning(f"Validation error processing {file.filename}: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error processing {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process EEG file during inference.",
        )
    finally:
        # Cleanup temporary files safely
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@router.post(
    "/predict-vector",
    response_model=PredictionResponse,
    summary="Direct inference from numerical EEG vector",
)
def predict_eeg_vector(req: VectorPredictionRequest):
    """Executes the imagined speech communication pipeline directly from a 2D numerical array of numbers."""
    import numpy as np
    from backend.app.services.eeg_preprocessing import preprocessor

    raw_arr = np.array(req.eeg, dtype=np.float32)
    if raw_arr.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"EEG data must be 2D (channels x time samples), got shape {raw_arr.shape}",
        )

    # If already harmonized (122, 1280)
    if raw_arr.shape == (preprocessor.canonical_size, preprocessor.target_samples):
        harmonized = raw_arr
        mask = np.ones(preprocessor.canonical_size, dtype=np.float32)
    else:
        ch_names = req.channel_names or preprocessor.named_channels[: raw_arr.shape[0]]
        harmonized, mask = preprocessor.preprocess_trial(
            raw_trial=raw_arr,
            channel_names=ch_names,
            sfreq=req.sfreq or 256.0,
            dataset_name=req.dataset or "kara_one",
        )

    tensor = np.expand_dims(harmonized, 0)
    mask_tensor = np.expand_dims(mask, 0)

    result = inference_service.run_pipeline(
        preprocessed_data={
            "tensor": tensor,
            "channel_mask": mask_tensor,
            "dataset_name": req.dataset or "kara_one",
            "subject_name": req.subject or "MM05",
            "ground_truth": req.ground_truth,
        },
        dataset_name=req.dataset or "kara_one",
        subject_name=req.subject or "MM05",
        original_filename="raw_vector_input",
    )
    return result

