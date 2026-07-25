"""
constants.py

Shared paths and hyperparameters for the RV-007 emotion classifier
baseline run: a single yolov8n-cls classifier trained on FER-2013's 7
emotion classes. All hyperparameters are Ultralytics defaults except
imgsz, set to the standard 224 classification resolution (640 is a
detection-mode default, not a tuning choice) — same convention as the
age/gender baseline (RV-004).
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data" / "fer2013" / "processed"
BASE_CHECKPOINT = REPO_ROOT / "models" / "yolov8n-cls.pt"
RUNS_DIR = REPO_ROOT / "runs" / "emotion_baseline"
MODEL_OUT_DIR = REPO_ROOT / "models" / "emotion"
WEIGHTS_PATH = MODEL_OUT_DIR / "baseline.pt"
REPORT_PATH = MODEL_OUT_DIR / "baseline_report.json"

EPOCHS = 100
IMGSZ = 224
RANDOM_SEED = 42