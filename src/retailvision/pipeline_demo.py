"""
pipeline_demo.py

End-to-end demo of the RetailVision pipeline: opens a camera or a
pre-recorded video file, runs each frame through InferencePipeline (face
detection plus age/gender/emotion classification), tracks each detected
face across frames and counts virtual-line crossings (see tracking.py,
counter.py), shows a live preview with each detected person's bounding
box, predictions, the counting line, and the running occupancy drawn, and
appends an anonymized log record for each detection (see output_log.py).
Optionally also ships the same records to a central server in real time
(see remote_log.py), for deployments running more than one camera node --
this is a second, independent sink alongside the local log file, not a
replacement for it.
Supports a headless --benchmark mode (no display) to measure sustained
FPS, since imshow overhead would otherwise skew the number -- benchmark
mode has no window to capture a 'q' keypress, so it stops automatically
after --duration seconds (or when the source runs out, for a video file)
rather than waiting on unreachable input. Per the project's testing
convention, point --source at a pre-recorded video file first -- a
deterministic input makes debugging far easier than a live camera feed --
before testing against the live camera.
"""

import argparse
import time

import cv2

from .counter import LineCounter
from .inference import InferencePipeline
from .output_log import log_detection
from .remote_log import RemoteLogShipper
from .tracking import CentroidTracker, bbox_centroid


def parse_args() -> argparse.Namespace:
    """Parse --source, --benchmark, --duration, the --line-* counting line options, and the optional --server-url/--camera-node-id/--api-key trio."""
    parser = argparse.ArgumentParser(description="Run the RetailVision inference pipeline")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run headless (no preview window) and report average FPS on exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Benchmark mode only: seconds to run before stopping automatically (default: 15)",
    )
    parser.add_argument(
        "--line-axis",
        choices=["x", "y"],
        default="x",
        help="Axis the virtual counting line spans perpendicular to: 'x' for a vertical line, 'y' for a horizontal one (default: x)",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=None,
        help="Pixel coordinate of the counting line along --line-axis (default: middle of the frame)",
    )
    parser.add_argument(
        "--line-direction",
        choices=["increasing", "decreasing"],
        default="increasing",
        help="Crossing direction counted as an entry: 'increasing' (e.g. left-to-right) or 'decreasing' (default: increasing)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Central server base URL to also ship anonymized records to (e.g. http://localhost:8000). "
        "Omit to log locally only.",
    )
    parser.add_argument(
        "--camera-node-id",
        default=None,
        help="This machine's identifier, sent alongside shipped records. Required if --server-url is set.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the central server. Required if --server-url is set.",
    )
    args = parser.parse_args()
    if args.server_url and not (args.camera_node_id and args.api_key):
        parser.error("--server-url requires both --camera-node-id and --api-key")
    return args


def open_source(source: str) -> cv2.VideoCapture:
    """Open a camera by index or a video file by path."""
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def draw_detections(frame, detections: list[dict]) -> None:
    """Draw each detected person's bounding box and predictions onto the frame."""
    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        line1 = f"{det['age_group']} ({conf['age']:.2f}) / {det['gender']} ({conf['gender']:.2f})"
        line2 = f"{det['emotion']} ({conf['emotion']:.2f})"
        cv2.putText(frame, line1, (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, line2, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def draw_counter(frame, counter: LineCounter, width: int, height: int) -> None:
    """Draw the virtual counting line and the current occupancy count onto the frame."""
    position = int(counter.position)
    if counter.axis == "x":
        cv2.line(frame, (position, 0), (position, height), (255, 0, 0), 2)
    else:
        cv2.line(frame, (0, position), (width, position), (255, 0, 0), 2)
    cv2.putText(frame, f"Occupancy: {counter.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def main() -> None:
    """Run the inference pipeline over a camera or video source until 'q', EOF, or --duration elapses."""
    args = parse_args()
    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_position = args.line_position
    if line_position is None:
        line_position = width / 2 if args.line_axis == "x" else height / 2

    pipeline = InferencePipeline()
    tracker = CentroidTracker()
    counter = LineCounter(axis=args.line_axis, position=line_position, entry_direction=args.line_direction)
    shipper = None
    if args.server_url:
        shipper = RemoteLogShipper(args.server_url, args.camera_node_id, args.api_key)
        print(f"Shipping records to {args.server_url} as camera node '{args.camera_node_id}'.")

    print(f"Source opened at {width}x{height} on device: {pipeline.device}.")
    print(f"Counting line: {args.line_axis}={line_position:.0f}, entry direction: {args.line_direction}.")
    if args.benchmark:
        print(f"Benchmark mode: running for up to {args.duration:.0f}s (Ctrl+C also stops cleanly).")
    else:
        print("Press 'q' to quit.")

    frame_count = 0
    total_faces = 0
    start = time.perf_counter()
    last_report = start
    try:
        while True:
            if args.benchmark and (time.perf_counter() - start) >= args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            detections = pipeline.process_frame(frame)
            total_faces += len(detections)

            bboxes = [det["bbox"] for det in detections]
            track_ids = tracker.update(bboxes)
            timestamp = time.time()
            tracks = {track_id: bbox_centroid(bbox) for track_id, bbox in zip(track_ids, bboxes)}
            for track_id, event in counter.update(tracks, timestamp):
                print(f"  Track {track_id} {event} -- occupancy now {counter.count}")

            for det, track_id in zip(detections, track_ids):
                dwell = counter.dwell_seconds(track_id, timestamp)
                log_detection(det, count=counter.count, dwell_seconds=dwell)
                if shipper is not None:
                    shipper.ship(det, count=counter.count, dwell_seconds=dwell)

            if args.benchmark:
                now = time.perf_counter()
                if now - last_report >= 2.0:
                    running_fps = frame_count / (now - start)
                    print(f"  {frame_count} frames, {running_fps:.2f} FPS running avg, {len(detections)} faces this frame")
                    last_report = now
            else:
                draw_detections(frame, detections)
                draw_counter(frame, counter, width, height)
                cv2.imshow("RetailVision - inference pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        elapsed = time.perf_counter() - start
        cap.release()
        cv2.destroyAllWindows()
        if shipper is not None:
            shipper.close()
        if frame_count:
            avg_faces = total_faces / frame_count
            print(
                f"Processed {frame_count} frames in {elapsed:.1f}s "
                f"({frame_count / elapsed:.2f} FPS, avg {avg_faces:.2f} faces/frame)"
            )


if __name__ == "__main__":
    main()
