"""
capture.py

Live camera capture loop for one RV-006 condition session: detects faces
with the existing Haar cascade FaceDetector, classifies each with the
fine-tuned age/gender models, draws predictions (green if they match the
supplied ground truth, red if not) on the preview window, and logs one row
per frame to a CSV for later aggregation.
"""

import csv
import time
from pathlib import Path

import cv2

from src.retailvision.detection import FaceDetector

from .classify import classify_face

FIELDNAMES = [
    "timestamp",
    "condition",
    "true_age",
    "true_gender",
    "face_detected",
    "age_pred",
    "age_conf",
    "age_correct",
    "gender_pred",
    "gender_conf",
    "gender_correct",
]


def run_session(condition: str, true_age: str, true_gender: str, models: dict, device: str, log_path: Path) -> int:
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
                    "true_age": true_age,
                    "true_gender": true_gender,
                    "face_detected": bool(faces),
                    "age_pred": "",
                    "age_conf": "",
                    "age_correct": "",
                    "gender_pred": "",
                    "gender_conf": "",
                    "gender_correct": "",
                }

                if faces:
                    x, y, w, h = faces[0]
                    crop = frame[y : y + h, x : x + w]
                    predictions = classify_face(models, crop, device)
                    age_pred, age_conf = predictions["age"]
                    gender_pred, gender_conf = predictions["gender"]
                    age_correct = age_pred == true_age
                    gender_correct = gender_pred == true_gender

                    row.update(
                        age_pred=age_pred,
                        age_conf=age_conf,
                        age_correct=age_correct,
                        gender_pred=gender_pred,
                        gender_conf=gender_conf,
                        gender_correct=gender_correct,
                    )

                    color = (0, 255, 0) if (age_correct and gender_correct) else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{age_pred} ({age_conf}) / {gender_pred} ({gender_conf})"
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