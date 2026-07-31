"""
evaluate_widerface_real_world.py

CLI entry point that replays one condition's recorded video (from
record_widerface_eval_session.py) through one YOLOv8 face detector
checkpoint, logging per-frame detection results. Run twice per
condition -- once per --model -- so both checkpoints are compared on
identical frames.

Usage:
  PYTHONPATH=.:scripts ./venv/bin/python3 scripts/evaluate_widerface_real_world.py \
      --condition close_1m --model baseline
  PYTHONPATH=.:scripts ./venv/bin/python3 scripts/evaluate_widerface_real_world.py \
      --condition close_1m --model final
"""

import argparse

from ultralytics import YOLO
from widerface_real_world_eval.constants import LOG_DIR, VIDEO_DIR, WEIGHTS, resolve_device
from widerface_real_world_eval.evaluate import evaluate_session


def parse_args() -> argparse.Namespace:
    """Parse the condition and which checkpoint to evaluate."""
    parser = argparse.ArgumentParser(description="Evaluate a recorded session against one face-detector checkpoint")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model", required=True, choices=list(WEIGHTS.keys()))
    return parser.parse_args()


def main() -> None:
    """Replay one condition's recording through one checkpoint and log the results."""
    args = parse_args()
    video_path = VIDEO_DIR / f"{args.condition}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(
            f"No recording found at {video_path} -- run record_widerface_eval_session.py --condition {args.condition} first"
        )

    device = resolve_device()
    model = YOLO(str(WEIGHTS[args.model]))

    log_path = LOG_DIR / f"{args.condition}__{args.model}.csv"
    row_count = evaluate_session(video_path, args.condition, args.model, model, device, log_path)
    print(f"Logged {row_count} frames to {log_path}")


if __name__ == "__main__":
    main()
