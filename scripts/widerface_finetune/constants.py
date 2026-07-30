"""
constants.py

Shared paths and hyperparameters for WIDER FACE fine-tuning. Retrains
from the original yolov8n checkpoint (rather than continuing from
baseline.pt), same rationale as every other fine-tune in this project:
a clean run with a lower learning rate and tuned augmentation, instead
of compounding whatever the baseline run already converged to.

Augmentation is tuned specifically for the domain gap documented in
docs/datasets/widerface.md: WIDER FACE is dense crowd-scene photography
(median annotated face is only 15px), while the retail camera use case
this detector is ultimately built for expects one or a few large,
prominent faces at close-to-medium range. Two changes target that gap
directly:

- scale=0.9 (up from the detection default 0.5): a wider random-zoom
  range exposes the model to more "zoomed in, large face" training
  crops than WIDER FACE's raw size distribution would otherwise provide.
- mosaic=0.5 (down from the detection default 1.0): mosaic stitches 4
  images into one frame, so each sub-image only occupies ~1/4 of the
  training frame -- this makes every face look smaller/more crowd-like
  than it actually is, which compounds WIDER FACE's existing bias in
  exactly the wrong direction for this use case. Lowering (not
  eliminating -- mosaic still has real generalization value) the
  probability lets more batches train on full-frame, single-scene
  images that better resemble retail camera framing.

degrees=10.0 is also added (baseline used the detection default, 0.0)
for mild rotation robustness to camera angle / head tilt, matching the
rationale already used for UTKFace's augmentation (docs/datasets/utkface.md).
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_YAML = REPO_ROOT / "data" / "widerface" / "processed" / "widerface.yaml"
BASE_CHECKPOINT = REPO_ROOT / "models" / "yolov8n.pt"
RUNS_DIR = REPO_ROOT / "runs" / "widerface_finetune"
MODEL_OUT_DIR = REPO_ROOT / "models" / "face_detection"
WEIGHTS_PATH = MODEL_OUT_DIR / "final.pt"
REPORT_PATH = MODEL_OUT_DIR / "final_report.json"

IMGSZ = 640
RANDOM_SEED = 42

# Adjusted from the baseline (epochs=100, batch=8 [see widerface_baseline's
# own adjustment from the detection default 16], lr0=0.01, patience=100).
# Same batch=8 ceiling applies here -- WIDER FACE's dense-crowd memory-spike
# risk that motivated it doesn't go away just because augmentation changed.
# patience=15 lets training stop early if it plateaus; the baseline showed
# no overfitting even at epoch 100, so this isn't fighting instability the
# way the classification fine-tunes' shorter epoch budgets were.
EPOCHS = 100
BATCH = 8
LR0 = 0.005
PATIENCE = 15

AUGMENTATION = {
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.9,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "mosaic": 0.5,
}

# Minimum acceptable Hard-difficulty recall (WIDER FACE's official
# partition, evaluated in official_eval.py), evaluated on the held-out
# test split. Hard is the most retail-relevant tier available at this
# stage: not because it encodes the same difficulty axes as retail
# footage (angle, lighting, distance -- that comparison is RV-028's job,
# against live-camera conditions, not this dataset), but because
# fine-tuning's whole purpose is to not regress on the baseline's own
# result (71.10%).
MIN_HARD_RECALL = 0.7110
