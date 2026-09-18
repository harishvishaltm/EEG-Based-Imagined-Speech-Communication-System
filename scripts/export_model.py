"""Model export and preparation script.

This script constructs the exact ImaginedSpeechModel architecture from the research notebook,
loads an existing checkpoint if provided, or creates a valid initialized checkpoint with full
metadata so the inference backend can start, execute evaluation-mode predictions, and run health
checks without retraining.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.imagined_speech_model import ImaginedSpeechModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("export_model")


def export_model(
    checkpoint_input: Path = None,
    output_model_path: Path = None,
    models_dir: Path = None,
    device: str = "cpu",
):
    models_dir = models_dir or (PROJECT_ROOT / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    output_model_path = output_model_path or (models_dir / "kara_one_model.pth")
    config_path = models_dir / "model_config.json"
    label_path = models_dir / "label_mapping.json"

    if not config_path.exists() or not label_path.exists():
        logger.error(f"Missing config files in {models_dir}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    logger.info("Initializing ImaginedSpeechModel architecture...")
    model = ImaginedSpeechModel(
        dataset_name_to_id=config["dataset_name_to_id"],
        num_classes_per_dataset=config["classes_per_dataset"],
        n_subjects=config.get("n_subjects", 21),
        n_datasets=config.get("n_datasets", 3),
        canonical_channels=config.get("canonical_channels", 122),
    )

    loaded_from_ckpt = False
    val_loss = 0.0
    epoch = 0

    if checkpoint_input and checkpoint_input.exists():
        logger.info(f"Loading weights from checkpoint: {checkpoint_input}")
        ckpt = torch.load(checkpoint_input, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]} (total {len(missing)})")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]} (total {len(unexpected)})")
        loaded_from_ckpt = True
        val_loss = ckpt.get("val_loss", 0.0)
        epoch = ckpt.get("epoch", 0)
        logger.info("Successfully loaded weights from existing checkpoint.")
    else:
        logger.info(
            "No pre-existing checkpoint file found. Initializing clean model weights with seed."
        )
        torch.manual_seed(42)

    model.eval()

    save_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "is_calibrated": loaded_from_ckpt,
        "model_config": config,
        "dataset_name_to_id": config["dataset_name_to_id"],
        "classes_per_dataset": config["classes_per_dataset"],
    }

    torch.save(save_payload, output_model_path)
    logger.info(f"Exported inference model to: {output_model_path}")
    logger.info(f"Model file size: {output_model_path.stat().st_size / (1024**2):.2f} MB")
    logger.info(f"Status: is_calibrated={loaded_from_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ImaginedSpeechModel for backend inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to an existing .pth training checkpoint if available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for exported model (defaults to models/kara_one_model.pth).",
    )
    args = parser.parse_args()

    export_model(checkpoint_input=args.checkpoint, output_model_path=args.output)
