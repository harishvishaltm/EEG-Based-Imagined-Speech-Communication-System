"""Validation script comparing Backend Model and Preprocessing against Notebook specification.

Verifies:
1. Model component parameter counts (Gate: 3,281, Backbone: 668,348, Heads: 26,718 -> Total: 698,347)
2. Preprocessed tensor shape: (1, 122, 1280)
3. Binary channel mask shape: (1, 122)
4. Model forward pass on single dataset (Kara One -> logits shape (1, 11))
5. Top-k probability summation (sums to ~1.0)
6. Label mapping correspondence (11 classes for Kara One)
"""
import json
import logging
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.models.imagined_speech_model import ImaginedSpeechModel
from backend.app.services.eeg_preprocessing import preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validate_backend")


def run_validation():
    print("=" * 80)
    print("VALIDATING BACKEND AGAINST RESEARCH NOTEBOOK SPECIFICATION")
    print("=" * 80)

    # 1. Check Metadata
    with open(settings.MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(settings.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    kara_one_classes = labels["kara_one"]["classes"]
    print(f"[CHECK 1] Kara One classes count: {len(kara_one_classes)} (Expected: 11)")
    assert len(kara_one_classes) == 11, "Kara One classes must be exactly 11."
    expected_classes = ["/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot"]
    assert kara_one_classes == expected_classes, f"Classes mismatch: {kara_one_classes} vs {expected_classes}"
    print("  -> PASSED: Label mapping matches notebook.")

    # 2. Check Architecture Parameter Count
    model = ImaginedSpeechModel(
        dataset_name_to_id=cfg["dataset_name_to_id"],
        num_classes_per_dataset=cfg["classes_per_dataset"],
        n_subjects=cfg.get("n_subjects", 21),
        n_datasets=cfg.get("n_datasets", 3),
        canonical_channels=cfg.get("canonical_channels", 122),
    )

    gate_params = sum(p.numel() for p in model.gate.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    heads_params = sum(p.numel() for p in model.heads.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print("\n[CHECK 2] Parameter Counts:")
    print(f"  Gate params     : {gate_params:,} (Notebook: 3,281)")
    print(f"  Backbone params : {backbone_params:,} (Notebook: 668,348)")
    print(f"  Heads params    : {heads_params:,} (Notebook: 26,718)")
    print(f"  Total params    : {total_params:,} (Notebook: 698,347)")

    assert gate_params == 3281, f"Gate params {gate_params} != 3281"
    assert backbone_params == 668348, f"Backbone params {backbone_params} != 668,348"
    assert heads_params == 26718, f"Heads params {heads_params} != 26,718"
    assert total_params == 698347, f"Total params {total_params} != 698,347"
    print("  -> PASSED: Exact parameter count match.")

    # 3. Check Preprocessing Pipeline & Tensor Shapes
    print("\n[CHECK 3] Preprocessing on simulated EEG trial:")
    raw_trial = np.random.randn(62, 1000).astype(np.float32)  # 62 channels, 1.0s at 1000Hz
    harmonized, mask = preprocessor.preprocess_trial(
        raw_trial=raw_trial,
        channel_names=preprocessor.named_channels,
        sfreq=1000.0,
        dataset_name="kara_one",
    )

    print(f"  Harmonized trial shape : {harmonized.shape} (Expected: (122, 1280))")
    print(f"  Channel mask shape     : {mask.shape} (Expected: (122,))")
    print(f"  Active channels in mask: {int(mask.sum())} (Expected: 62)")
    assert harmonized.shape == (122, 1280), f"Shape mismatch: {harmonized.shape}"
    assert mask.shape == (122,), f"Mask shape mismatch: {mask.shape}"
    assert int(mask.sum()) == 62, f"Active channel mismatch: {int(mask.sum())}"
    print("  -> PASSED: Preprocessing shapes match canonical 122x1280 layout.")

    # 4. Check Inference Forward Pass
    print("\n[CHECK 4] Forward pass through ImaginedSpeechModel:")
    model.eval()
    x = torch.from_numpy(harmonized).unsqueeze(0).float()
    m = torch.from_numpy(mask).unsqueeze(0).float()
    s = torch.tensor([0], dtype=torch.long)
    d = torch.tensor([1], dtype=torch.long)  # Kara One dataset_id = 1

    with torch.inference_mode():
        logits = model.forward_single_dataset(
            eeg=x,
            channel_mask=m,
            subject_id=s,
            dataset_id=d,
            dataset_name="kara_one",
        )
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    print(f"  Logits shape : {logits.shape} (Expected: (1, 11))")
    print(f"  Probabilities sum : {probs.sum():.6f} (Expected: ~1.000000)")
    assert logits.shape == (1, 11), f"Logits shape mismatch: {logits.shape}"
    assert abs(probs.sum() - 1.0) < 1e-5, f"Probabilities don't sum to 1: {probs.sum()}"
    print("  -> PASSED: Model forward pass produces valid 11-class probability distribution.")

    print("\n" + "=" * 80)
    print("ALL VALIDATION CHECKS PASSED: BACKEND STRICTLY CONFORMS TO NOTEBOOK SPECIFICATION")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
