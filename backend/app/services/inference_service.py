"""Inference Service orchestrating the complete prediction pipeline:
Preprocessing -> Forward Inference -> LLM Sentence Generation -> TTS Audio Synthesis.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from backend.app.models.model_loader import model_manager
from backend.app.services.eeg_preprocessing import preprocessor
from backend.app.services.llm_service import llm_service
from backend.app.services.tts_service import tts_service

logger = logging.getLogger("inference_service")


class InferenceService:
    def __init__(self):
        pass

    def run_pipeline(
        self,
        file_path: Optional[Path] = None,
        preprocessed_data: Optional[Dict[str, Any]] = None,
        dataset_name: str = "kara_one",
        subject_name: Optional[str] = None,
        original_filename: str = "eeg_input.npz",
    ) -> Dict[str, Any]:
        """Runs the complete imagined speech communication pipeline.

        STRICT RULE: Only forward inference is executed. ZERO retraining.
        """
        t0 = time.perf_counter()

        # Step 1: Preprocessing
        t_prep_start = time.perf_counter()
        if preprocessed_data is not None:
            data = preprocessed_data
        elif file_path is not None:
            data = preprocessor.load_and_preprocess_file(file_path, dataset_name=dataset_name)
        else:
            raise ValueError("Either file_path or preprocessed_data must be provided.")
        t_prep_ms = round((time.perf_counter() - t_prep_start) * 1000, 2)

        tensor = data["tensor"]  # numpy (1, 122, 1280)
        mask = data["channel_mask"]  # numpy (1, 122)
        ds = data.get("dataset_name", dataset_name)
        subj = subject_name or data.get("subject_name", "MM05")
        ground_truth = data.get("ground_truth")

        # Convert to torch tensor
        torch_tensor = torch.from_numpy(tensor).float()
        torch_mask = torch.from_numpy(mask).float()

        # Step 2: Model Inference (strictly model.eval(), torch.inference_mode())
        t_infer_start = time.perf_counter()
        # Map subject string to subject id if possible
        subj_id = 0
        if subj in ["MM05", "MM08", "MM09"]:
            subj_id = ["MM05", "MM08", "MM09"].index(subj)

        pred_result = model_manager.predict_single(
            eeg_tensor=torch_tensor,
            channel_mask=torch_mask,
            subject_id=subj_id,
            dataset_name=ds,
        )
        t_infer_ms = round((time.perf_counter() - t_infer_start) * 1000, 2)

        predicted_word = pred_result["predicted_word"]
        confidence = pred_result["confidence"]
        top_predictions = pred_result["top_predictions"]

        # Step 3: Visualization extraction
        vis_data = preprocessor.extract_visualization_data(
            harmonized_tensor=tensor,
            channel_mask=mask,
            max_channels=8,
            decimate_factor=4,
        )

        # Step 4: Assistive LLM Sentence Generation
        sentence_result = llm_service.generate_sentence(predicted_word)
        generated_sentence = sentence_result["sentence"]
        generation_mode = sentence_result["generation_mode"]

        # Step 5: Text-to-Speech Synthesis
        tts_result = tts_service.synthesize(text=generated_sentence, word_hint=predicted_word)
        audio_url = tts_result["audio_url"]

        # Step 6: Assemble Response
        response = {
            "success": True,
            "input": {
                "filename": original_filename,
                "dataset": ds,
                "subject": subj,
            },
            "prediction": {
                "word": predicted_word,
                "confidence": confidence,
            },
            "top_predictions": top_predictions,
            "ground_truth": ground_truth,
            "evaluation": {
                "correct": (ground_truth.strip().lower() == predicted_word.strip().lower())
            }
            if ground_truth
            else None,
            "communication": {
                "sentence": generated_sentence,
                "generation_mode": generation_mode,
            },
            "audio": {
                "url": audio_url,
            },
            "processing": {
                "inference_time_ms": t_infer_ms,
                "preprocessing_time_ms": t_prep_ms,
                "device": str(model_manager.device),
            },
            "visualization": vis_data,
        }

        logger.info(
            f"Pipeline executed: word={predicted_word}, conf={confidence:.3f}, "
            f"mode={generation_mode}, audio={audio_url}, time={t_infer_ms}ms"
        )
        return response


inference_service = InferenceService()
