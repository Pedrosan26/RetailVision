"""
constants.py

Shared paths for the emotion classifier's real-world evaluation: capturing
live camera frames at different camera distances and non-frontal face
angles, running the fine-tuned emotion classifier on detected faces, and
logging per-frame predictions against a user-supplied ground truth emotion
for later accuracy-degradation analysis.
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)
from emotion_finetune.constants import DATA_DIR, WEIGHTS_PATH  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LOG_DIR = REPO_ROOT / "runs" / "emotion_real_world_eval"

# Ground-truth choices, read from the same class folders training used, so
# the CLI's --true-emotion flag can't drift from the model's actual class set.
EMOTION_CLASSES = sorted(p.name for p in (DATA_DIR / "test").iterdir() if p.is_dir())