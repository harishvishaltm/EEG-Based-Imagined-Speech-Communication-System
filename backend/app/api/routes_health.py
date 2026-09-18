"""Health and Model Information Endpoints."""
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.models.model_loader import model_manager
from backend.app.schemas.prediction import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["Health & Model Info"])


@router.get("/", summary="Root API index")
def root_index():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": f"{settings.API_PREFIX}/health",
        "model_info": f"{settings.API_PREFIX}/model-info",
        "demo_samples": f"{settings.API_PREFIX}/demo-samples",
        "predict": f"{settings.API_PREFIX}/predict",
    }


@router.get("/api/health", response_model=HealthResponse, summary="Check API and Model Health")
def health_check():
    if not model_manager.model_loaded:
        model_manager.initialize()
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.model_loaded,
        device=str(model_manager.device),
        dataset="Kara One",
        mode="inference",
        is_calibrated=model_manager.is_calibrated,
        version=settings.VERSION,
    )


@router.get("/api/model-info", response_model=ModelInfoResponse, summary="Retrieve Model Metadata")
def model_info():
    if not model_manager.model_loaded:
        model_manager.initialize()
    cfg = model_manager.config
    classes = model_manager.label_mapping.get("kara_one", {}).get("classes", [])

    return ModelInfoResponse(
        model_name=cfg.get("model_name", "ImaginedSpeechModel"),
        dataset="Kara One",
        number_of_classes=len(classes),
        classes=classes,
        input_shape=[1, cfg.get("canonical_channels", 122), cfg.get("target_epoch_samples", 1280)],
        sampling_frequency=cfg.get("target_sfreq", 256),
        model_loaded=model_manager.model_loaded,
        device=str(model_manager.device),
        architecture=cfg.get(
            "architecture",
            "LearnedChannelGate + TemporalCNN + MultiHeadAttention + BiLSTM + FeatureFusion + Heads",
        ),
    )
