"""
constants.py

Shared paths and hyperparameters for the age-regression model. Uses a
plain torchvision ResNet18 backbone with a single-output regression head
rather than YOLOv8/Ultralytics, since Ultralytics has no native regression
task (only detect, classify, pose, segment, obb, semantic).
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)
from age_regression_prep.constants import MANIFEST_DIR  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_OUT_DIR = REPO_ROOT / "models" / "age_gender"
WEIGHTS_PATH = MODEL_OUT_DIR / "regression_age.pt"
REPORT_PATH = MODEL_OUT_DIR / "regression_report.json"
RUNS_DIR = REPO_ROOT / "runs" / "age_regression"

IMGSZ = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PATIENCE = 5  # epochs without val-MAE improvement before early stopping
RANDOM_SEED = 42

# Boundaries used only to bucket the continuous predictions for per-decade
# MAE reporting — not used for training.
REPORT_AGE_BUCKETS = [
    (0, 17, "0-17"),
    (18, 30, "18-30"),
    (31, 50, "31-50"),
    (51, 200, "51+"),
]