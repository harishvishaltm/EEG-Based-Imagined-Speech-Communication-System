"""Model Loader and In-Memory Singleton Manager.

Guarantees:
1. Model is loaded ONCE during application startup.
2. Model is placed in model.eval() mode with gradients disabled.
3. Kept resident in RAM/VRAM for all subsequent API requests.
4. ZERO retraining, backward(), optimizer steps, or epoch loops during inference.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from backend.app.config import settings
from backend.app.models.imagined_speech_model import ImaginedSpeechModel

logger = logging.getLogger("model_loader")


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.model: Optional[ImaginedSpeechModel] = None
        self.device: torch.device = torch.device("cpu")
        self.model_loaded: bool = False
        self.is_calibrated: bool = False
        self.config: Dict[str, Any] = {}
        self.label_mapping: Dict[str, Any] = {}
        self.channel_mapping: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Called once during FastAPI lifespan startup."""
        if self.model_loaded:
            return

        # Determine compute device
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Initializing ModelManager on device: {self.device}")

        # Load metadata JSONs
        with open(settings.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(settings.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)
        with open(settings.CHANNEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.channel_mapping = json.load(f)

        # Check if checkpoint exists; if not, export initialized model
        if not settings.MODEL_PATH.exists():
            logger.warning(
                f"Model checkpoint not found at {settings.MODEL_PATH}. "
                "Generating initial model checkpoint using scripts/export_model.py..."
            )
            from scripts.export_model import export_model
            export_model(output_model_path=settings.MODEL_PATH, device=str(self.device))

        # Load weights
        logger.info(f"Loading checkpoint from: {settings.MODEL_PATH}")
        ckpt = torch.load(settings.MODEL_PATH, map_location=self.device)

        state_dict = ckpt.get("model_state_dict", ckpt)
        self.is_calibrated = ckpt.get("is_calibrated", False)

        # Construct model
        self.model = ImaginedSpeechModel(
            dataset_name_to_id=self.config["dataset_name_to_id"],
            num_classes_per_dataset=self.config["classes_per_dataset"],
            n_subjects=self.config.get("n_subjects", 21),
            n_datasets=self.config.get("n_datasets", 3),
            canonical_channels=self.config.get("canonical_channels", 122),
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

        # Strictly freeze model for inference
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model_loaded = True
        logger.info(
            f"ImaginedSpeechModel successfully loaded into memory. "
            f"(calibrated={self.is_calibrated}, eval_mode=True, grad_disabled=True)"
        )

    @staticmethod
    def normalize_dataset_name(name: str) -> str:
        clean = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        if clean in ["kara_one", "karaone"]:
            return "kara_one"
        return clean

    def predict_single(
        self,
        eeg_tensor: torch.Tensor,
        channel_mask: torch.Tensor,
        subject_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        dataset_name: str = "kara_one",
    ) -> Dict[str, Any]:
        """Runs strictly forward inference on a single sample without retraining.

        Parameters
        ----------
        eeg_tensor   : (1, 122, 1280) float32
        channel_mask : (1, 122) float32
        subject_id   : optional subject integer (defaults to 0)
        dataset_id   : optional dataset integer (defaults to mapped ID for dataset_name)
        dataset_name : string identifier ('kara_one', 'feis', 'nguyen')

        Returns
        -------
        Dictionary with predicted class, confidence, top_predictions list, probabilities.
        """
        if not self.model_loaded or self.model is None:
            self.initialize()

        dataset_name = self.normalize_dataset_name(dataset_name)
        if dataset_id is None:
            dataset_id = self.config["dataset_name_to_id"].get(dataset_name, 1)
        if subject_id is None:
            subject_id = 0

        # Prepare batch tensors
        x = eeg_tensor.to(self.device)
        mask = channel_mask.to(self.device)
        s_id = torch.tensor([subject_id], dtype=torch.long, device=self.device)
        d_id = torch.tensor([dataset_id], dtype=torch.long, device=self.device)

        # STRICT INFERENCE MODE: No gradients, no retraining
        with torch.inference_mode():
            logits = self.model.forward_single_dataset(
                eeg=x,
                channel_mask=mask,
                subject_id=s_id,
                dataset_id=d_id,
                dataset_name=dataset_name,
                return_gate=False,
            )  # (1, n_classes)

            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        classes = self.label_mapping[dataset_name]["classes"]
        pred_idx = int(probabilities.argmax())
        pred_word = classes[pred_idx]
        confidence = float(probabilities[pred_idx])

        # Calculate top-k predictions
        top_indices = probabilities.argsort()[::-1]
        top_predictions = [
            {"word": classes[i], "confidence": round(float(probabilities[i]), 4)}
            for i in top_indices[:5]
        ]

        return {
            "predicted_word": pred_word,
            "confidence": round(confidence, 4),
            "top_predictions": top_predictions,
            "all_probabilities": {classes[i]: round(float(probabilities[i]), 4) for i in range(len(classes))},
        }


model_manager = ModelManager.get_instance()
