"""
constants.py

Shared paths and hyperparameters for the WIDER FACE face-detector
baseline run. Unlike the classification baselines (age/gender, emotion),
which override imgsz down to 224 (the classification-mode default),
this uses Ultralytics' actual detection-mode default resolution
(imgsz=640) unchanged -- 640 is the correct default here, not a tuning
choice deviated from.
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_YAML = REPO_ROOT / "data" / "widerface" / "processed" / "widerface.yaml"
BASE_CHECKPOINT = REPO_ROOT / "models" / "yolov8n.pt"
RUNS_DIR = REPO_ROOT / "runs" / "widerface_baseline"
MODEL_OUT_DIR = REPO_ROOT / "models" / "face_detection"
WEIGHTS_PATH = MODEL_OUT_DIR / "baseline.pt"
REPORT_PATH = MODEL_OUT_DIR / "baseline_report.json"

EPOCHS = 100
IMGSZ = 640
RANDOM_SEED = 42

# Reduced from Ultralytics' detection-mode default of 16. WIDER FACE's
# crowd-scene images occasionally carry hundreds to ~2,000 ground-truth
# boxes in a single frame; when a batch's mosaic composites happen to
# include one of these, the label-assignment step's memory cost (which
# scales with total boxes in the batch) can spike far past what typical
# detection datasets (a handful of objects per image) ever produce.
# A first attempt at batch=16 hit an out-of-memory condition that
# corrupted training state and crashed outright. Halving batch size
# lowers that worst-case spike; Ultralytics normalizes its loss/LR
# scaling against a fixed nominal batch size (separate from this value),
# so this doesn't require re-tuning the learning rate to compensate.
BATCH = 8