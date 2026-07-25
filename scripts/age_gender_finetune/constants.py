"""
constants.py

Shared paths and hyperparameters for RV-005 fine-tuning. Retrains both
classifiers from the original yolov8n-cls checkpoint (rather than continuing
from the RV-004 baseline weights) so augmentation and a lower learning rate
get a clean run instead of compounding the mild overfitting already present
in baseline_age.pt past ~epoch 70. Augmentation values match the strategy
decided in docs/datasets/utkface.md (RV-003) but not applied to the RV-004
baseline, which needed to be a faithful "defaults" floor.
"""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TASKS = {
    "age": REPO_ROOT / "data" / "utkface" / "processed" / "age",
    "gender": REPO_ROOT / "data" / "utkface" / "processed" / "gender",
}

BASE_CHECKPOINT = REPO_ROOT / "models" / "yolov8n-cls.pt"
RUNS_DIR = REPO_ROOT / "runs" / "age_gender_finetune"
MODEL_OUT_DIR = REPO_ROOT / "models" / "age_gender"
REPORT_PATH = MODEL_OUT_DIR / "final_report.json"

IMGSZ = 224
RANDOM_SEED = 42

# Adjusted from baseline defaults (epochs=100, batch=16, lr0=0.01, patience=100).
# Lower LR + smaller batch + early stopping target the overfitting seen in the
# age baseline after ~epoch 70; augmentation adds the variation UTKFace's
# pre-cropped, studio-lit chips otherwise lack.
EPOCHS = 60
BATCH = 32
LR0 = 0.005
PATIENCE = 15

AUGMENTATION = {
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.2,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.2,
    "erasing": 0.2,
}

# RV-005 acceptance thresholds, evaluated on the held-out test split.
MIN_ACCURACY = {
    "age": 0.75,
    "gender": 0.85,
}


def resolve_device() -> str:
    """Pick the fastest available torch device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"