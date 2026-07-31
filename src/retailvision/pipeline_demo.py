"""
pipeline_demo.py

End-to-end demo of the RetailVision pipeline: opens a camera or a
pre-recorded video file, runs each frame through InferencePipeline (face
detection plus age/gender/emotion classification), and shows a live
preview with each detected person's bounding box and predictions drawn.
Supports a headless --benchmark mode (no display) to measure sustained
FPS, since imshow overhead would otherwise skew the number -- benchmark
mode has no window to capture a 'q' keypress, so it stops automatically
after --duration seconds (or when the source runs out, for a video file)
rather than waiting on unreachable input. Per the project's testing
convention, point --source at a pre-recorded video file first -- a
deterministic input makes debugging far easier than a live camera feed --
before testing against the live camera.

--detector lets a YOLOv8-based face detector (under real-world
evaluation as an eventual replacement for Haar cascade) be tried in the
live pipeline for a quick visual look. This is a demo-only override, not
a production capability: an actual detector swap, once real-world
evaluation supports it, belongs inside FaceDetector itself.
"""

import argparse
import time

import cv2

from .detection import FaceDetector
from .inference import InferencePipeline

_YOLO_FACE_WEIGHTS = {
    "yolo-baseline": "models/face_detection/baseline.pt",
    "yolo-final": "models/face_detection/final.pt",
}


class _YoloFaceDetector:
    """Demo-only: wraps a YOLOv8 face-detection checkpoint behind FaceDetector's same detect(frame) interface."""

    def __init__(self, weights_path: str) -> None:
        """Load the given YOLOv8 detection checkpoint."""
        from ultralytics import YOLO

        self._model = YOLO(weights_path)

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        """Detect faces in a BGR frame, returning (x, y, w, h) boxes."""
        result = self._model.predict(source=frame, verbose=False)[0]
        boxes = []
        for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes


def build_detector(name: str):
    """Construct the requested detector: Haar cascade, or a YOLOv8 checkpoint by name."""
    if name == "haar":
        return FaceDetector()
    return _YoloFaceDetector(_YOLO_FACE_WEIGHTS[name])


def parse_args() -> argparse.Namespace:
    """Parse --source, --benchmark, --duration, and --detector."""
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
        "--detector",
        default="haar",
        choices=["haar", *_YOLO_FACE_WEIGHTS.keys()],
        help="Face detector to use (default: haar, the production detector)",
    )
    return parser.parse_args()


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


def main() -> None:
    """Run the inference pipeline over a camera or video source until 'q', EOF, or --duration elapses."""
    args = parse_args()
    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = InferencePipeline(detector=build_detector(args.detector))
    print(f"Source opened at {width}x{height} on device: {pipeline.device}. Detector: {args.detector}.")
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

            if args.benchmark:
                now = time.perf_counter()
                if now - last_report >= 2.0:
                    running_fps = frame_count / (now - start)
                    print(f"  {frame_count} frames, {running_fps:.2f} FPS running avg, {len(detections)} faces this frame")
                    last_report = now
            else:
                draw_detections(frame, detections)
                cv2.imshow("RetailVision - inference pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        elapsed = time.perf_counter() - start
        cap.release()
        cv2.destroyAllWindows()
        if frame_count:
            avg_faces = total_faces / frame_count
            print(
                f"Processed {frame_count} frames in {elapsed:.1f}s "
                f"({frame_count / elapsed:.2f} FPS, avg {avg_faces:.2f} faces/frame)"
            )


if __name__ == "__main__":
    main()
