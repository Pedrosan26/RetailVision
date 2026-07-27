"""
evaluate_real_world.py

CLI entry point for the age/gender classifiers' real-world evaluation:
runs one live-camera evaluation session for a single condition (e.g.
normal lighting, low light, face turned away, angled face), classifying
detected faces with the fine-tuned age/gender models and logging
per-frame predictions against a supplied ground truth. Run this once per
condition you want to test, then run scripts/summarize_real_world_eval.py
to aggregate accuracy per condition.

Usage:
  PYTHONPATH=.:scripts ./venv/bin/python3 scripts/evaluate_real_world.py \
      --condition normal_light --true-age 18-30 --true-gender Male

Press 'q' to end the session. Re-running with the same --condition appends
more frames to that condition's log (e.g. to add more footage later).
"""

import argparse

from age_gender_baseline.constants import resolve_device
from real_world_eval.capture import run_session
from real_world_eval.classify import load_classifiers
from real_world_eval.constants import AGE_CLASSES, GENDER_CLASSES, LOG_DIR


def parse_args() -> argparse.Namespace:
    """Parse the condition name and ground-truth age/gender for this session."""
    parser = argparse.ArgumentParser(description="Age/gender real-world evaluation session")
    parser.add_argument("--condition", required=True, help="Condition label, e.g. normal_light, low_light, angled_45, turned_away")
    parser.add_argument("--true-age", required=True, choices=AGE_CLASSES)
    parser.add_argument("--true-gender", required=True, choices=GENDER_CLASSES)
    return parser.parse_args()


def main() -> None:
    """Load the classifiers and run one condition's live capture session."""
    args = parse_args()
    device = resolve_device()
    print(f"Loading classifiers on device: {device}")
    models = load_classifiers(device)

    log_path = LOG_DIR / f"{args.condition}.csv"
    row_count = run_session(args.condition, args.true_age, args.true_gender, models, device, log_path)
    print(f"Logged {row_count} frames to {log_path}")


if __name__ == "__main__":
    main()