"""
constants.py

Shared paths and label definitions for UTKFace preparation: where the raw
source images and processed output live, the age-group bin boundaries,
and the gender/race code-to-label mappings used by UTKFace filenames.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIRS = [
    REPO_ROOT / "data" / "utkface" / "raw" / "UTKFace",
    REPO_ROOT / "data" / "utkface" / "raw" / "crop_part1",
]
PROCESSED_DIR = REPO_ROOT / "data" / "utkface" / "processed"
REPORT_PATH = PROCESSED_DIR / "distribution_report.json"

AGE_BINS = [
    (0, 17, "0-17"),
    (18, 30, "18-30"),
    (31, 50, "31-50"),
    (51, 200, "51+"),
]
GENDER_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
