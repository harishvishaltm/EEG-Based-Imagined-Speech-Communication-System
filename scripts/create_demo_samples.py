"""Generate reference demonstration samples for college presentation.

These samples follow the exact Kara One tensor format (122 canonical channels x 1280 samples)
with ground-truth labels and 62-channel active masks.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "data" / "kara_one" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 122 channels: first 62 are named 10-20 scalp EEG channels, remaining 60 generic
NAMED_CHANNELS_COUNT = 62
CANONICAL_TOTAL = 122
TIME_SAMPLES = 1280  # 5.0 seconds at 256 Hz

SAMPLES_SPEC = [
    {"id": "sample_001", "subject": "MM05", "label": "pot", "seed": 101},
    {"id": "sample_002", "subject": "MM05", "label": "pat", "seed": 102},
    {"id": "sample_003", "subject": "MM08", "label": "knew", "seed": 103},
    {"id": "sample_004", "subject": "MM08", "label": "gnaw", "seed": 104},
    {"id": "sample_005", "subject": "MM09", "label": "/uw/", "seed": 105},
]


def generate_demo_samples():
    print(f"Generating demonstration samples in {DEMO_DIR}...")
    for spec in SAMPLES_SPEC:
        rng = np.random.RandomState(spec["seed"])
        # Generate realistic EEG frequency band signals (alpha 8-12Hz, beta 13-30Hz) for the 62 active channels
        time = np.linspace(0, 5.0, TIME_SAMPLES)
        eeg_62 = np.zeros((NAMED_CHANNELS_COUNT, TIME_SAMPLES), dtype=np.float32)

        for ch in range(NAMED_CHANNELS_COUNT):
            alpha = 0.6 * np.sin(2 * np.pi * rng.uniform(8.0, 12.0) * time + rng.uniform(0, 2 * np.pi))
            beta = 0.3 * np.sin(2 * np.pi * rng.uniform(15.0, 25.0) * time + rng.uniform(0, 2 * np.pi))
            noise = 0.2 * rng.randn(TIME_SAMPLES)
            raw_sig = alpha + beta + noise
            # Z-score normalize
            eeg_62[ch] = (raw_sig - raw_sig.mean()) / (raw_sig.std() + 1e-8)

        # Harmonize to 122 canonical channels
        harmonized = np.zeros((CANONICAL_TOTAL, TIME_SAMPLES), dtype=np.float32)
        harmonized[:NAMED_CHANNELS_COUNT, :] = eeg_62

        # Create binary channel mask (1 for named channels, 0 for generic padding)
        mask = np.zeros(CANONICAL_TOTAL, dtype=np.float32)
        mask[:NAMED_CHANNELS_COUNT] = 1.0

        sample_file = DEMO_DIR / f"{spec['id']}.npz"
        np.savez_compressed(
            sample_file,
            eeg=harmonized,
            channel_mask=mask,
            subject=spec["subject"],
            dataset="Kara One",
            label=spec["label"],
            sample_id=spec["id"],
            sampling_rate=256,
        )
        print(f"  Created {sample_file.name}: subject={spec['subject']}, ground_truth='{spec['label']}'")

    print(f"Successfully created {len(SAMPLES_SPEC)} demo samples.")


if __name__ == "__main__":
    generate_demo_samples()
