"""EEG Preprocessing Service.

Direct implementation of preprocessing and channel harmonization from project (1).ipynb:
- Filtering (IIR Butterworth Bandpass 0.5-100 Hz, Notch 50/60 Hz)
- Common Average Referencing (CAR)
- Resampling (Target 256 Hz)
- Standardization of epoch length (1280 samples = 5.0 s)
- Z-score normalization per channel
- Canonical channel harmonization (122 channels, 62 named channels)
- Binary channel mask generation
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.io
import scipy.signal

from backend.app.config import settings

logger = logging.getLogger("eeg_preprocessing")


class EEGPreprocessor:
    def __init__(self):
        self._load_configs()

    def _load_configs(self):
        # Load channel mapping
        with open(settings.CHANNEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.channel_mapping_data = json.load(f)

        self.named_channels = self.channel_mapping_data["named_channels"]
        self.non_eeg_channels = set(self.channel_mapping_data["excluded_non_eeg_channels"])
        self.canonical_size = self.channel_mapping_data["total_canonical_channels"]  # 122
        self.generic_slots = self.channel_mapping_data["generic_channels_count"]  # 60

        # Load preprocessing config
        with open(settings.PREPROCESSING_CONFIG_PATH, "r", encoding="utf-8") as f:
            self.prep_config = json.load(f)

        self.target_sfreq = self.prep_config["target_sfreq"]  # 256
        self.target_samples = self.prep_config["target_epoch_samples"]  # 1280
        self.bandpass_low = self.prep_config["bandpass_low"]  # 0.5
        self.bandpass_high = self.prep_config["bandpass_high"]  # 100.0
        self.notch_freqs = self.prep_config["notch_freqs"]  # [50.0, 60.0]

    @staticmethod
    def normalize_channel_name(name: str) -> str:
        """Uppercase and alphanumeric filter: e.g. ' Fp1 ' -> 'FP1'."""
        return "".join(ch for ch in str(name).strip().upper() if ch.isalnum())

    def is_eeg_channel(self, name: str) -> bool:
        """Check if channel is a genuine scalp EEG channel."""
        return self.normalize_channel_name(name) not in self.non_eeg_channels

    def filter_trial(
        self,
        trial: np.ndarray,
        sfreq: float,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        notch_freqs: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Bandpass (0.5-100Hz) and Notch filter (50, 60Hz) using Butterworth IIR."""
        l_freq = l_freq or self.bandpass_low
        h_freq = h_freq or self.bandpass_high
        notch_freqs = notch_freqs or self.notch_freqs

        data = trial.astype(np.float64)
        nyquist = sfreq / 2.0
        h_freq_safe = min(h_freq, nyquist - 1.0)

        # Try MNE first if installed, otherwise fallback to scipy
        try:
            import mne
            filtered = mne.filter.filter_data(
                data,
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq_safe,
                method="iir",
                iir_params=dict(order=4, ftype="butter"),
                verbose=False,
            )
            for nf in notch_freqs:
                if nf < nyquist:
                    filtered = mne.filter.notch_filter(
                        filtered,
                        Fs=sfreq,
                        freqs=[nf],
                        method="iir",
                        iir_params=dict(order=4, ftype="butter"),
                        verbose=False,
                    )
            return filtered.astype(np.float32)
        except Exception:
            # Scipy SOS IIR Butterworth fallback
            # Bandpass
            sos = scipy.signal.butter(
                4, [l_freq, h_freq_safe], btype="bandpass", fs=sfreq, output="sos"
            )
            filtered = scipy.signal.sosfiltfilt(sos, data, axis=-1)

            # Notch filters
            for nf in notch_freqs:
                if nf < nyquist:
                    w0 = nf / nyquist
                    bw = 2.0 / nyquist
                    b, a = scipy.signal.iirnotch(w0, w0 / bw)
                    filtered = scipy.signal.filtfilt(b, a, filtered, axis=-1)

            return filtered.astype(np.float32)

    @staticmethod
    def apply_car(trial: np.ndarray) -> np.ndarray:
        """Common Average Reference: subtract across-channel mean at every timepoint."""
        return trial - trial.mean(axis=0, keepdims=True)

    def resample_trial(self, trial: np.ndarray, orig_sfreq: float) -> np.ndarray:
        """Resample to target_sfreq (256 Hz)."""
        if abs(orig_sfreq - self.target_sfreq) < 1e-6:
            return trial.astype(np.float32)

        try:
            import mne
            return mne.filter.resample(
                trial.astype(np.float64),
                up=self.target_sfreq,
                down=orig_sfreq,
                npad="auto",
                verbose=False,
            ).astype(np.float32)
        except Exception:
            num_samples = int(trial.shape[1] * self.target_sfreq / orig_sfreq)
            return scipy.signal.resample(trial, num_samples, axis=-1).astype(np.float32)

    def standardize_epoch_length(self, trial: np.ndarray) -> np.ndarray:
        """Standardize time length to exactly 1280 samples (center-crop or center-pad)."""
        n_channels, n_time = trial.shape
        if n_time == self.target_samples:
            return trial
        if n_time > self.target_samples:
            start = (n_time - self.target_samples) // 2
            return trial[:, start : start + self.target_samples]

        pad_total = self.target_samples - n_time
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(trial, ((0, 0), (pad_left, pad_right)), mode="constant").astype(np.float32)

    @staticmethod
    def zscore_normalize(trial: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Per-channel z-score normalization across time."""
        mean = trial.mean(axis=1, keepdims=True)
        std = trial.std(axis=1, keepdims=True)
        std_safe = np.where(std < eps, 1.0, std)
        return ((trial - mean) / std_safe).astype(np.float32)

    def map_channels(
        self, dataset_name: str, subject_channel_names: List[str]
    ) -> Dict[int, Optional[int]]:
        """Map subject channel index -> canonical index in 122-channel layout."""
        mapping: Dict[int, Optional[int]] = {}
        is_generic = dataset_name == "nguyen" or all(
            str(c).lower().startswith("ch_") for c in subject_channel_names
        )

        if is_generic:
            n_named = len(self.named_channels)
            for i in range(len(subject_channel_names)):
                mapping[i] = n_named + i if i < self.generic_slots else None
        else:
            name_to_canon = {name: idx for idx, name in enumerate(self.named_channels)}
            for i, ch_name in enumerate(subject_channel_names):
                norm = self.normalize_channel_name(ch_name)
                mapping[i] = name_to_canon.get(norm)

        return mapping

    def zero_pad_missing_channels(
        self, trial: np.ndarray, mapping: Dict[int, Optional[int]]
    ) -> np.ndarray:
        """Place channels at canonical indices and zero-fill missing channels to (122, T)."""
        n_time = trial.shape[1]
        out = np.zeros((self.canonical_size, n_time), dtype=np.float32)
        for orig_idx, canon_idx in mapping.items():
            if canon_idx is not None and orig_idx < trial.shape[0]:
                out[canon_idx, :] = trial[orig_idx, :]
        return out

    def create_channel_mask(self, mapping: Dict[int, Optional[int]]) -> np.ndarray:
        """Create binary 1D mask (122,) indicating active channels."""
        mask = np.zeros(self.canonical_size, dtype=np.float32)
        for _, canon_idx in mapping.items():
            if canon_idx is not None:
                mask[canon_idx] = 1.0
        return mask

    def preprocess_trial(
        self,
        raw_trial: np.ndarray,
        channel_names: List[str],
        sfreq: float,
        dataset_name: str = "kara_one",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the complete preprocessing on an input trial.

        Returns:
            harmonized_trial: (122, 1280)
            channel_mask: (122,)
        """
        # 1. Filter
        filtered = self.filter_trial(raw_trial, sfreq)
        # 2. CAR
        car_applied = self.apply_car(filtered)
        # 3. Resample to 256 Hz
        resampled = self.resample_trial(car_applied, sfreq)
        # 4. Standardize length to 1280 samples
        standardized = self.standardize_epoch_length(resampled)
        # 5. Z-score normalize
        normalized = self.zscore_normalize(standardized)
        # 6. Harmonization to canonical 122 channels
        mapping = self.map_channels(dataset_name, channel_names)
        harmonized = self.zero_pad_missing_channels(normalized, mapping)
        mask = self.create_channel_mask(mapping)

        return harmonized, mask

    def load_and_preprocess_file(
        self, file_path: Path, dataset_name: str = "kara_one"
    ) -> Dict[str, Any]:
        """Loads and preprocesses an uploaded file (.npz, .mat, or .cnt).

        Returns:
            {
                "tensor": np.ndarray of shape (1, 122, 1280),
                "channel_mask": np.ndarray of shape (1, 122),
                "dataset_name": str,
                "subject_name": str,
                "ground_truth": Optional[str],
                "visualization": Dict
            }
        """
        suffix = file_path.suffix.lower()

        if suffix == ".npz":
            data = np.load(file_path, allow_pickle=True)
            # Check if it's already a prepared demo sample
            if "eeg" in data:
                eeg_arr = data["eeg"]
                channel_mask = data["channel_mask"] if "channel_mask" in data else None
                subj = str(data.get("subject", "MM05"))
                ds = str(data.get("dataset", dataset_name))
                gt = str(data["label"]) if "label" in data else None

                # If already harmonized (122, 1280) or (1, 122, 1280)
                if eeg_arr.ndim == 2 and eeg_arr.shape == (self.canonical_size, self.target_samples):
                    tensor = np.expand_dims(eeg_arr.astype(np.float32), 0)
                    if channel_mask is None:
                        channel_mask = np.ones(self.canonical_size, dtype=np.float32)
                    mask_tensor = (
                        np.expand_dims(channel_mask, 0) if channel_mask.ndim == 1 else channel_mask
                    )
                elif eeg_arr.ndim == 3 and eeg_arr.shape[1:] == (
                    self.canonical_size,
                    self.target_samples,
                ):
                    tensor = eeg_arr.astype(np.float32)
                    if channel_mask is None:
                        channel_mask = np.ones((eeg_arr.shape[0], self.canonical_size), dtype=np.float32)
                    mask_tensor = channel_mask
                else:
                    # Raw trial saved in npz
                    ch_names = list(data["channel_names"]) if "channel_names" in data else self.named_channels
                    sfreq = float(data.get("sfreq", 1000.0 if dataset_name == "kara_one" else 256.0))
                    harmonized, mask = self.preprocess_trial(eeg_arr, ch_names, sfreq, ds)
                    tensor = np.expand_dims(harmonized, 0)
                    mask_tensor = np.expand_dims(mask, 0)

                return {
                    "tensor": tensor,
                    "channel_mask": mask_tensor,
                    "dataset_name": ds,
                    "subject_name": subj,
                    "ground_truth": gt,
                }

        elif suffix == ".mat":
            mat = scipy.io.loadmat(file_path, simplify_cells=True)
            eeg_arr = None
            for candidate in ["eeg", "data", "trial", "val"]:
                if candidate in mat:
                    eeg_arr = np.array(mat[candidate])
                    break
            if eeg_arr is None:
                raise ValueError("Could not find EEG array ('eeg', 'data', 'trial') in .mat file.")

            ch_names = mat.get("channel_names", self.named_channels)
            sfreq = float(mat.get("sfreq", 1000.0 if dataset_name == "kara_one" else 256.0))
            harmonized, mask = self.preprocess_trial(eeg_arr, ch_names, sfreq, dataset_name)
            return {
                "tensor": np.expand_dims(harmonized, 0),
                "channel_mask": np.expand_dims(mask, 0),
                "dataset_name": dataset_name,
                "subject_name": "MM05",
                "ground_truth": None,
            }

        elif suffix == ".cnt":
            try:
                import mne
                raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
                data = raw.get_data()
                # Slices first epoch
                target_len = int(raw.info["sfreq"] * 5.0)
                trial_data = data[:, :target_len] if data.shape[1] >= target_len else data
                harmonized, mask = self.preprocess_trial(
                    trial_data, raw.ch_names, raw.info["sfreq"], dataset_name
                )
                return {
                    "tensor": np.expand_dims(harmonized, 0),
                    "channel_mask": np.expand_dims(mask, 0),
                    "dataset_name": dataset_name,
                    "subject_name": "MM05",
                    "ground_truth": None,
                }
            except Exception as e:
                raise ValueError(f"Failed to parse .cnt file: {e}")

        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported formats are .npz, .mat, and .cnt."
            )

    def extract_visualization_data(
        self,
        harmonized_tensor: np.ndarray,
        channel_mask: np.ndarray,
        max_channels: int = 8,
        decimate_factor: int = 4,
    ) -> Dict[str, Any]:
        """Extracts downsampled waveform series for frontend visualization."""
        # harmonized_tensor is (1, 122, 1280) or (122, 1280)
        eeg = harmonized_tensor[0] if harmonized_tensor.ndim == 3 else harmonized_tensor
        mask = channel_mask[0] if channel_mask.ndim == 2 else channel_mask

        # Select prominent EEG channels that are active
        preferred_display = ["CZ", "FZ", "PZ", "C3", "C4", "FP1", "FP2", "O1", "O2", "T7", "T8"]
        selected_channels = []

        for ch in preferred_display:
            if ch in self.named_channels:
                idx = self.named_channels.index(ch)
                if idx < len(mask) and mask[idx] > 0.5:
                    selected_channels.append((ch, idx))
            if len(selected_channels) >= max_channels:
                break

        # If not enough preferred, pick any active canonical channels
        if len(selected_channels) < max_channels:
            for idx, active in enumerate(mask):
                if active > 0.5 and idx < len(self.named_channels):
                    ch = self.named_channels[idx]
                    if (ch, idx) not in selected_channels:
                        selected_channels.append((ch, idx))
                if len(selected_channels) >= max_channels:
                    break

        n_samples = eeg.shape[1]
        step = max(1, decimate_factor)
        time_points = [round(float(t / self.target_sfreq), 4) for t in range(0, n_samples, step)]

        waveforms = []
        for ch_name, idx in selected_channels:
            sampled_vals = [round(float(v), 4) for v in eeg[idx, ::step]]
            waveforms.append({"channel": ch_name, "values": sampled_vals})

        return {
            "sampling_rate": self.target_sfreq,
            "duration_seconds": round(float(n_samples / self.target_sfreq), 2),
            "time_points": time_points,
            "waveforms": waveforms,
        }


# Global singleton preprocessor instance
preprocessor = EEGPreprocessor()
