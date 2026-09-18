"""Kara One dataset extraction, preprocessing, and demo sample preparation.

Responsibilities:
1. Locates Kara One subject archives (e.g. MM05.tar.bz2, MM08.tar.bz2, MM09.tar.bz2) in data/kara_one/raw/.
2. Extracts them to data/kara_one/processed/.
3. Reads EEG recordings (.cnt), epoch boundaries (epoch_inds.mat), and task prompts (labels.txt).
4. Applies the exact 7-step notebook preprocessing pipeline.
5. Harmonizes channels into the canonical 122-channel layout.
6. Generates individual demo samples in data/kara_one/demo_samples/.
"""
import argparse
import logging
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.eeg_preprocessing import preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prepare_kara_one")

RAW_DIR = PROJECT_ROOT / "data" / "kara_one" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "kara_one" / "processed"
DEMO_DIR = PROJECT_ROOT / "data" / "kara_one" / "demo_samples"


def extract_archives(raw_dir: Path, processed_dir: Path) -> List[Path]:
    """Extracts .tar.bz2, .tar.gz, or .zip archives found in raw_dir."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    archives = list(raw_dir.glob("*.tar.bz2")) + list(raw_dir.glob("*.tar.gz")) + list(raw_dir.glob("*.zip"))
    extracted_dirs = []

    for archive in archives:
        subj_name = archive.name.split(".")[0]
        target_dir = processed_dir / subj_name
        if not target_dir.exists() or not any(target_dir.iterdir()):
            logger.info(f"Extracting {archive.name} -> {target_dir}...")
            if archive.name.endswith((".tar.bz2", ".tar.gz", ".tar")):
                with tarfile.open(archive, "r:*") as tar:
                    tar.extractall(path=target_dir)
            logger.info(f"Extracted {archive.name}")
        extracted_dirs.append(target_dir)

    return extracted_dirs


def load_subject_trials(subj_dir: Path) -> Optional[Dict]:
    """Reads .cnt, epoch_inds.mat, and labels.txt from an extracted subject folder."""
    import mne

    cnt_files = list(subj_dir.rglob("*.cnt"))
    epoch_files = list(subj_dir.rglob("epoch_inds.mat"))
    label_files = list(subj_dir.rglob("labels.txt"))

    if not cnt_files or not epoch_files or not label_files:
        logger.warning(
            f"Incomplete files in {subj_dir}: cnt={len(cnt_files)}, "
            f"epochs={len(epoch_files)}, labels={len(label_files)}"
        )
        return None

    raw = mne.io.read_raw_cnt(cnt_files[0], preload=True, verbose=False)
    epoch_mat = scipy.io.loadmat(epoch_files[0], simplify_cells=True)
    thinking_inds = epoch_mat["thinking_inds"]

    labels = label_files[0].read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    labels = [l.strip() for l in labels if l.strip()]

    n_trials = min(len(labels), len(thinking_inds))
    eeg_data = raw.get_data()  # (channels, total_samples)

    trials = []
    valid_labels = []

    for i in range(n_trials):
        start, end = int(thinking_inds[i][0]), int(thinking_inds[i][1])
        if 0 <= start < end <= eeg_data.shape[1]:
            trials.append(eeg_data[:, start:end])
            valid_labels.append(labels[i])

    return {
        "trials": trials,
        "labels": valid_labels,
        "channel_names": raw.ch_names,
        "sfreq": raw.info["sfreq"],
        "n_trials": len(trials),
    }


def prepare_kara_one(max_demo_samples_per_subject: int = 15):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    extracted = extract_archives(RAW_DIR, PROCESSED_DIR)

    if not extracted and not any(PROCESSED_DIR.iterdir()):
        logger.warning(
            "No Kara One data found in data/kara_one/raw/ or data/kara_one/processed/.\n"
            "Please download MM05.tar.bz2, MM08.tar.bz2, etc. from University of Toronto "
            "and place into data/kara_one/raw/.\n"
            "Run 'python scripts/setup_dataset.py' for detailed instructions."
        )
        return

    sample_counter = 1
    total_saved = 0

    subj_folders = [p for p in PROCESSED_DIR.iterdir() if p.is_dir()]
    logger.info(f"Processing {len(subj_folders)} subject directories...")

    for subj_dir in subj_folders:
        subj_name = subj_dir.name
        data = load_subject_trials(subj_dir)
        if not data:
            continue

        logger.info(f"Preprocessing {data['n_trials']} trials for subject {subj_name}...")
        count = 0

        for i in range(data["n_trials"]):
            raw_trial = data["trials"][i]
            label = data["labels"][i]

            # Run full preprocessing and harmonization
            harmonized, mask = preprocessor.preprocess_trial(
                raw_trial=raw_trial,
                channel_names=data["channel_names"],
                sfreq=data["sfreq"],
                dataset_name="kara_one",
            )

            # Save demo sample as .npz
            sample_filename = f"sample_{sample_counter:03d}.npz"
            sample_path = DEMO_DIR / sample_filename

            np.savez_compressed(
                sample_path,
                eeg=harmonized,
                channel_mask=mask,
                subject=subj_name,
                dataset="Kara One",
                label=label,
                sample_id=f"sample_{sample_counter:03d}",
                sampling_rate=256,
            )

            sample_counter += 1
            count += 1
            total_saved += 1

            if count >= max_demo_samples_per_subject:
                break

        logger.info(f"Saved {count} demo samples for {subj_name}")

    logger.info(f"Total demo samples created in {DEMO_DIR}: {total_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and prepare Kara One EEG dataset.")
    parser.add_argument(
        "--max_samples_per_subject",
        type=int,
        default=15,
        help="Maximum demo samples to extract per subject.",
    )
    args = parser.parse_args()

    prepare_kara_one(max_demo_samples_per_subject=args.max_samples_per_subject)
