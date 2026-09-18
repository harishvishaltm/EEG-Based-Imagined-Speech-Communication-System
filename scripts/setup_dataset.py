"""Dataset setup and verification utility for Kara One database.

This script guides the user on obtaining and placing Kara One subject archives
from the University of Toronto, and verifies the repository folder structure.
"""
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "kara_one" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "kara_one" / "processed"
DEMO_DIR = PROJECT_ROOT / "data" / "kara_one" / "demo_samples"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("setup_dataset")


def setup_directories():
    for d in [RAW_DIR, PROCESSED_DIR, DEMO_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory verified: {d}")


def inspect_raw_archives():
    archives = list(RAW_DIR.glob("*.tar.bz2")) + list(RAW_DIR.glob("*.tar.gz")) + list(RAW_DIR.glob("*.zip"))
    print("=" * 80)
    print("KARA ONE DATASET STATUS CHECK")
    print("=" * 80)
    print(f"Target raw directory: {RAW_DIR.resolve()}")
    if archives:
        print(f"Found {len(archives)} archive(s):")
        for a in archives:
            print(f"  - {a.name} ({a.stat().st_size / (1024**2):.2f} MB)")
        print("\nReady to run: python scripts/prepare_kara_one.py")
    else:
        print("No subject archives found in data/kara_one/raw/.")
        print("\nINSTRUCTIONS TO OBTAIN KARA ONE DATASET:")
        print("1. Source: University of Toronto Kara One database")
        print("   URL: http://www.cs.toronto.edu/~complingweb/data/karaone/karaone.html")
        print("2. Recommended primary subjects: MM05, MM08, MM09")
        print("3. Download the subject archives (e.g. MM05.tar.bz2, MM08.tar.bz2, MM09.tar.bz2)")
        print("4. Place the downloaded .tar.bz2 files into:")
        print(f"   {RAW_DIR.resolve()}")
        print("5. Run: python scripts/prepare_kara_one.py")
    print("=" * 80)


if __name__ == "__main__":
    setup_directories()
    inspect_raw_archives()
