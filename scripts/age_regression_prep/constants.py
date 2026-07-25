"""
constants.py

Paths for the age-regression manifest. Reuses utkface_prep's raw dirs,
split ratios, and random seed so the regression split is generated from
the same source data and methodology as the classification split.
"""

from pathlib import Path

from utkface_prep.constants import RANDOM_SEED, RAW_DIRS, SPLIT_RATIOS  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "data" / "utkface" / "processed" / "age_regression"
REPORT_PATH = MANIFEST_DIR / "distribution_report.json"