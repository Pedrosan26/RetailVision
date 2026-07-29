"""
constants.py

Shared paths for the age/gender classifiers' real-world evaluation:
capturing live camera frames under different conditions (lighting,
occlusion, angle), running the fine-tuned age/gender classifiers on
detected faces, and logging per-frame predictions against a user-supplied
ground truth for later accuracy-degradation analysis.
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)
from age_gender_finetune.constants import TASKS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

WEIGHTS = {
    "age": REPO_ROOT / "models" / "age_gender" / "final_age.pt",
    "gender": REPO_ROOT / "models" / "age_gender" / "final_gender.pt",
}

LOG_DIR = REPO_ROOT / "runs" / "real_world_eval"

# Ground-truth choices, read from the same class folders training used, so
# the CLI's --true-age/--true-gender flags can't drift from the model's
# actual class set.
AGE_CLASSES = sorted(p.name for p in (TASKS["age"] / "test").iterdir() if p.is_dir())
GENDER_CLASSES = sorted(p.name for p in (TASKS["gender"] / "test").iterdir() if p.is_dir())