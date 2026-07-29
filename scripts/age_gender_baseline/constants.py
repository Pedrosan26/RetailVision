"""
constants.py

Shared paths and hyperparameters for the baseline training run.
Two independent yolov8n-cls classifiers are trained (age-group, gender)
since YOLOv8 classification mode is single-label per run and the UTKFace
prep, already laid the data out as two separate classification
trees. All hyperparameters are Ultralytics defaults except imgsz, which is
set to the standard 224 classification resolution (the framework-wide
default of 640 is a detection-mode default, not a tuning choice).
"""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TASKS = {
    "age": REPO_ROOT / "data" / "utkface" / "processed" / "age",
    "gender": REPO_ROOT / "data" / "utkface" / "processed" / "gender",
}

BASE_CHECKPOINT = REPO_ROOT / "models" / "yolov8n-cls.pt"
RUNS_DIR = REPO_ROOT / "runs" / "age_gender_baseline"
MODEL_OUT_DIR = REPO_ROOT / "models" / "age_gender"
REPORT_PATH = MODEL_OUT_DIR / "baseline_report.json"

EPOCHS = 100
IMGSZ = 224
RANDOM_SEED = 42


def resolve_device() -> str:
    """Pick the fastest available torch device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"