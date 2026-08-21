"""
evaluate.py

Replays one condition's recorded video through a given YOLOv8 face
detector checkpoint, logging one row per frame: whether any face was
detected, how many boxes were found (more than one, for a single-subject
session, signals a likely false positive), and the highest confidence
among them.
"""

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

FIELDNAMES = ["frame_index", "condition", "model", "face_detected", "num_boxes", "max_confidence"]


def evaluate_session(video_path: Path, condition: str, model_name: str, model: YOLO, device: str, log_path: Path) -> int:
    """Run one checkpoint over a recorded video, logging one row per frame. Returns row count."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    try:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(source=frame, device=device, verbose=False)[0]
                num_boxes = len(result.boxes)
                max_confidence = round(float(result.boxes.conf.max()), 4) if num_boxes else 0.0

                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "condition": condition,
                        "model": model_name,
                        "face_detected": num_boxes > 0,
                        "num_boxes": num_boxes,
                        "max_confidence": max_confidence,
                    }
                )
                frame_index += 1
    finally:
        cap.release()

    return frame_index
