"""Tests for EEG Preprocessing and Harmonization Pipeline."""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.eeg_preprocessing import preprocessor


def test_channel_normalization():
    assert preprocessor.normalize_channel_name(" Fp1 ") == "FP1"
    assert preprocessor.normalize_channel_name("cz") == "CZ"
    assert preprocessor.is_eeg_channel("FP1") is True
    assert preprocessor.is_eeg_channel("EKG") is False
    assert preprocessor.is_eeg_channel("Trigger") is False


def test_car_referencing():
    trial = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    car_trial = preprocessor.apply_car(trial)
    # Mean across channels at each time step should be zero
    np.testing.assert_allclose(car_trial.mean(axis=0), np.zeros(2), atol=1e-6)


def test_epoch_standardization():
    # Longer epoch -> center crop to 1280
    long_trial = np.ones((62, 2000), dtype=np.float32)
    std_long = preprocessor.standardize_epoch_length(long_trial)
    assert std_long.shape == (62, 1280)

    # Shorter epoch -> center pad to 1280
    short_trial = np.ones((62, 500), dtype=np.float32)
    std_short = preprocessor.standardize_epoch_length(short_trial)
    assert std_short.shape == (62, 1280)


def test_zscore_normalization():
    trial = np.random.randn(62, 1280).astype(np.float32) * 5.0 + 10.0
    norm_trial = preprocessor.zscore_normalize(trial)
    # Mean ~0, std ~1 per channel
    np.testing.assert_allclose(norm_trial.mean(axis=1), np.zeros(62), atol=1e-5)
    np.testing.assert_allclose(norm_trial.std(axis=1), np.ones(62), atol=1e-4)


def test_full_preprocessing_shapes():
    raw_trial = np.random.randn(62, 1000).astype(np.float32)
    harmonized, mask = preprocessor.preprocess_trial(
        raw_trial=raw_trial,
        channel_names=preprocessor.named_channels,
        sfreq=1000.0,
        dataset_name="kara_one",
    )
    assert harmonized.shape == (122, 1280)
    assert mask.shape == (122,)
    assert int(mask.sum()) == 62
