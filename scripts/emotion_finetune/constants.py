"""
constants.py

Shared paths and hyperparameters for emotion classifier fine-tuning.
Retrains from the original yolov8n-cls checkpoint (rather than continuing
from the baseline weights) so augmentation and a lower learning rate get a
clean run instead of compounding the mild overfitting already present in
baseline.pt past ~epoch 40 — same rationale as the age/gender fine-tune.

Hyperparameters reuse the age/gender fine-tune's recipe (epochs=60,
batch=32, SGD, lr0=0.005, patience=15) as a reasonable first fine-tuning
iteration for a similarly-sized single yolov8n-cls classification task.

Augmentation follows the same philosophy as docs/datasets/utkface.md, with
one dataset-specific adjustment: FER-2013 images are grayscale, so hue
(`hsv_h`) and saturation (`hsv_s`) jitter are omitted entirely — a
zero-saturation pixel has no hue to shift, so those two would be pure
no-ops here. Value/brightness jitter (`hsv_v`) is kept since it still
affects grayscale pixel intensity and gives lighting robustness.

Iteration 2 (current): reduced from iteration 1's values (erasing=0.2,
degrees=10.0, translate=0.1) after iteration 1 regressed on nearly every
class, not just the target Neutral/Happy pair. FER-2013 is natively 48x48
(vs. UTKFace's full-resolution source photos), upscaled ~4.7x to the
model's 224 input — augmentation that occludes/distorts a fixed fraction
of the image removes proportionally far more signal here. `erasing`
specifically is suspected of blanking out the small mouth/eyebrow regions
that distinguish "neutral" from "sad" (see docs/models/emotion_finetune.md
for the confusion-matrix evidence behind this). See that doc for the full
iteration-1 writeup and rationale.
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data" / "fer2013" / "processed"
BASE_CHECKPOINT = REPO_ROOT / "models" / "yolov8n-cls.pt"
RUNS_DIR = REPO_ROOT / "runs" / "emotion_finetune"
MODEL_OUT_DIR = REPO_ROOT / "models" / "emotion"
WEIGHTS_PATH = MODEL_OUT_DIR / "final.pt"
REPORT_PATH = MODEL_OUT_DIR / "final_report.json"

IMGSZ = 224
RANDOM_SEED = 42

# Adjusted from baseline defaults (epochs=100, batch=16, lr0=0.01, patience=100).
EPOCHS = 60
BATCH = 32
LR0 = 0.005
PATIENCE = 15

AUGMENTATION = {
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.2,
    "hsv_v": 0.2,
    "erasing": 0.0,
}

# Minimum acceptable per-class recall for the two classes considered
# commercially relevant for retail — not an aggregate top1 bar like the
# age/gender models use.
MIN_CLASS_RECALL = {
    "happy": 0.80,
    "neutral": 0.80,
}