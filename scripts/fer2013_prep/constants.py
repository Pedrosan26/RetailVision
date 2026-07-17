"""
constants.py

Shared paths and label definitions for FER-2013 preparation: where the
raw source images and processed output live, the seven emotion class
names, and the fraction of the official train split carved out for
validation.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIR = REPO_ROOT / "data" / "fer2013" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "fer2013" / "processed"
REPORT_PATH = PROCESSED_DIR / "distribution_report.json"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EXPECTED_IMAGE_SIZE = (48, 48)
EXPECTED_IMAGE_MODE = "L"  # grayscale

VAL_FRACTION = 0.10
RANDOM_SEED = 42