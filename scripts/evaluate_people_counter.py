"""
evaluate_people_counter.py

Replays a pre-recorded video clip through the full detector + tracker +
virtual-line counter pipeline and checks the counted entries against a
manually-counted ground truth. Run this against a clip you've watched and
counted yourself first -- the whole point is comparing the automated
count to a known human-verified number.

Usage:
  PYTHONPATH=. ./venv/bin/python3 scripts/evaluate_people_counter.py \
      --source path/to/clip.mp4 --expected-entries 12
"""

import argparse
import json
from pathlib import Path

import cv2

from src.retailvision.counter import LineCounter
from src.retailvision.inference import InferencePipeline
from src.retailvision.tracking import CentroidTracker, bbox_centroid

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "runs" / "people_counter" / "eval_report.json"


def parse_args() -> argparse.Namespace:
    """Parse the video source, ground-truth entry count, and counting line configuration."""
    parser = argparse.ArgumentParser(description="Evaluate the people counter against a manually-counted video clip")
    parser.add_argument("--source", required=True, help="Path to the pre-recorded video clip")
    parser.add_argument("--expected-entries", type=int, required=True, help="Ground-truth entry count, counted manually")
    parser.add_argument("--line-axis", choices=["x", "y"], default="x")
    parser.add_argument("--line-position", type=float, default=None, help="Default: middle of the frame")
    parser.add_argument("--line-direction", choices=["increasing", "decreasing"], default="increasing")
    parser.add_argument("--no-preview", action="store_true", help="Run headless instead of showing the annotated video")
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict:
    """Run the counter over the video and return a report dict with counted vs. expected entries."""
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_position = args.line_position if args.line_position is not None else (
        width / 2 if args.line_axis == "x" else height / 2
    )

    pipeline = InferencePipeline()
    tracker = CentroidTracker()
    counter = LineCounter(axis=args.line_axis, position=line_position, entry_direction=args.line_direction)

    entries = 0
    exits = 0
    frame_count = 0
    timestamp = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            timestamp = frame_count / fps

            detections = pipeline.process_frame(frame)
            bboxes = [det["bbox"] for det in detections]
            track_ids = tracker.update(bboxes)
            tracks = {track_id: bbox_centroid(bbox) for track_id, bbox in zip(track_ids, bboxes)}

            for track_id, event in counter.update(tracks, timestamp):
                if event == "entry":
                    entries += 1
                else:
                    exits += 1

            if not args.no_preview:
                for x, y, w, h in bboxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                position = int(line_position)
                if args.line_axis == "x":
                    cv2.line(frame, (position, 0), (position, height), (255, 0, 0), 2)
                else:
                    cv2.line(frame, (0, position), (width, position), (255, 0, 0), 2)
                cv2.putText(
                    frame, f"Entries: {entries}  Exits: {exits}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                )
                cv2.imshow("People counter evaluation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    accuracy = 1 - abs(entries - args.expected_entries) / args.expected_entries if args.expected_entries else None

    return {
        "source": str(args.source),
        "frame_count": frame_count,
        "counted_entries": entries,
        "counted_exits": exits,
        "expected_entries": args.expected_entries,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "meets_80pct_threshold": accuracy is not None and accuracy >= 0.8,
        "line_axis": args.line_axis,
        "line_position": line_position,
        "line_direction": args.line_direction,
    }


def main() -> None:
    """Run the evaluation and print/save a report comparing counted vs. expected entries."""
    args = parse_args()
    report = run_eval(args)

    print(f"Counted {report['counted_entries']} entries, {report['counted_exits']} exits over {report['frame_count']} frames.")
    print(f"Expected {report['expected_entries']} entries -- accuracy: {report['accuracy']:.2%}")
    print("PASS" if report["meets_80pct_threshold"] else "FAIL", "-- 80% accuracy threshold")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
