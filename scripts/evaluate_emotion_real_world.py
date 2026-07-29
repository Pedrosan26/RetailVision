"""
evaluate_emotion_real_world.py

CLI entry point for the emotion classifier's real-world evaluation: runs
one live-camera session for a single (condition, emotion) combination,
classifying detected faces with the fine-tuned emotion classifier and
logging per-frame predictions against a supplied ground-truth emotion.
Run this once per combination you want to test, then run
scripts/summarize_emotion_real_world_eval.py to aggregate accuracy across
the full condition x emotion matrix.

Usage:
  PYTHONPATH=.:scripts ./venv/bin/python3 scripts/evaluate_emotion_real_world.py \
      --condition close_1m --true-emotion happy

Hold one consistent facial expression for the whole session, matching
whatever you pass as --true-emotion. Press 'q' to end the session.
Re-running with the same --condition/--true-emotion pair appends more
frames to that combination's log (e.g. to add more footage later).
"""

import argparse

from age_gender_baseline.constants import resolve_device
from emotion_real_world_eval.capture import run_session
from emotion_real_world_eval.classify import load_classifier
from emotion_real_world_eval.constants import EMOTION_CLASSES, LOG_DIR


def parse_args() -> argparse.Namespace:
    """Parse the condition name and ground-truth emotion for this session."""
    parser = argparse.ArgumentParser(description="Emotion classifier real-world evaluation session")
    parser.add_argument(
        "--condition",
        required=True,
        help="Condition label, e.g. close_1m, medium_2m, far_4m, side_view, looking_down",
    )
    parser.add_argument("--true-emotion", required=True, choices=EMOTION_CLASSES)
    return parser.parse_args()


def main() -> None:
    """Load the emotion classifier and run one (condition, emotion) live capture session."""
    args = parse_args()
    device = resolve_device()
    print(f"Loading emotion classifier on device: {device}")
    model = load_classifier(device)

    log_path = LOG_DIR / f"{args.condition}__{args.true_emotion}.csv"
    row_count = run_session(args.condition, args.true_emotion, model, device, log_path)
    print(f"Logged {row_count} frames to {log_path}")


if __name__ == "__main__":
    main()