"""
record_widerface_eval_session.py

CLI entry point that records one condition's live-camera session to a
video file, with no model inference at capture time. Run this once per
condition (close_1m, medium_2m, far_4m, side_view, looking_down,
no_person_background), then run evaluate_widerface_real_world.py twice
per condition -- once with --model baseline, once with --model final --
to replay the same recording through both checkpoints.

Usage:
  PYTHONPATH=.:scripts ./venv/bin/python3 scripts/record_widerface_eval_session.py --condition close_1m

Press 'q' to stop recording. Re-running with the same --condition
overwrites that condition's previous recording.
"""

import argparse

from widerface_real_world_eval.constants import CONDITIONS
from widerface_real_world_eval.record import record_session


def parse_args() -> argparse.Namespace:
    """Parse the condition name for this recording session."""
    parser = argparse.ArgumentParser(description="Record one condition's session for face-detector real-world evaluation")
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    return parser.parse_args()


def main() -> None:
    """Record one condition's live-camera session to a video file."""
    args = parse_args()
    record_session(args.condition)


if __name__ == "__main__":
    main()
