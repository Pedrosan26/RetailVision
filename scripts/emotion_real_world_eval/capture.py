"""
capture.py

Live camera capture loop for one (condition, emotion) session: detects
faces with the existing FaceDetector, classifies each with the
fine-tuned emotion model, draws the prediction (green if it matches
the supplied ground truth emotion, red if not) on the preview window, and
logs one row per frame to a CSV for later aggregation.
"""

import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.retailvision.detection import FaceDetector

from .classify import classify_face

FIELDNAMES = [
    "timestamp",
    "condition",
    "true_emotion",
    "face_detected",
    "emotion_pred",
    "emotion_conf",
    "emotion_correct",
]


def run_session(condition: str, true_emotion: str, model: YOLO, device: str, log_path: Path) -> int:
    """Capture frames until 'q' is pressed, logging one row per frame to log_path. Returns row count."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    detector = FaceDetector()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not log_path.exists()

    row_count = 0
    print(f"Camera opened for condition '{condition}'. Press 'q' to stop this session.")
    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if is_new_file:
                writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame")
                    break

                faces = detector.detect(frame)
                row = {
                    "timestamp": time.time(),
                    "condition": condition,
                    "true_emotion": true_emotion,
                    "face_detected": bool(faces),
                    "emotion_pred": "",
                    "emotion_conf": "",
                    "emotion_correct": "",
                }

                if faces:
                    x, y, w, h = faces[0]
                    crop = frame[y : y + h, x : x + w]
                    emotion_pred, emotion_conf = classify_face(model, crop, device)
                    emotion_correct = emotion_pred == true_emotion

                    row.update(
                        emotion_pred=emotion_pred,
                        emotion_conf=emotion_conf,
                        emotion_correct=emotion_correct,
                    )

                    color = (0, 255, 0) if emotion_correct else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{emotion_pred} ({emotion_conf})"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                writer.writerow(row)
                row_count += 1

                cv2.imshow(f"RetailVision - {condition}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return row_count