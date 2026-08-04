"""
constants.py

Shared paths and reference numbers for the YOLOv8 face detectors'
real-world evaluation.
"""

from pathlib import Path

from age_gender_baseline.constants import resolve_device  # noqa: F401 (re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

WEIGHTS = {
    "baseline": REPO_ROOT / "models" / "face_detection" / "baseline.pt",
    "final": REPO_ROOT / "models" / "face_detection" / "final.pt",
}

VIDEO_DIR = REPO_ROOT / "runs" / "widerface_real_world_eval" / "recordings"
LOG_DIR = REPO_ROOT / "runs" / "widerface_real_world_eval" / "logs"
REPORT_PATH = REPO_ROOT / "models" / "face_detection" / "real_world_eval_report.json"

# Same distance/angle conditions as the emotion model's real-world eval
# (docs/model_evaluation.md), for a direct comparison, plus a dedicated
# false-positive stress test: point the camera at a background with no
# person in frame. Any detected box there is an unambiguous false
# positive -- a cleaner test than relying on an incidental background
# object showing up during a normal session, which is how the Haar
# cascade's own false-positive failure mode (a mannequin, see
# docs/model_evaluation.md) was originally discovered.
CONDITIONS = ["close_1m", "medium_2m", "far_4m", "side_view", "looking_down", "no_person_background"]

# Haar cascade's own documented real-world face-detection rates
# (docs/model_evaluation.md), for direct comparison. side_view's Haar rate
# varied widely across single uncontrolled takes (10.6%-50.5%); the mean
# of the three recorded sessions is used here as a single reference point,
# not because the underlying variance goes away.
HAAR_REFERENCE_DETECTION_RATE = {
    "close_1m": 0.9848,
    "medium_2m": 0.9848,
    "far_4m": 0.9246,
    "side_view": 0.2703,
    "looking_down": 0.4629,
}
