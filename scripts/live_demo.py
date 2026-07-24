"""
live_demo.py

Live webcam preview with the current age/gender classifiers running on
every detected face: opens the camera, runs FaceDetector, classifies each
face crop, and draws the predicted age bin + gender + confidence on the
preview window. No ground truth needed - this is for casually trying out
whatever classifiers currently sit at models/age_gender/final_age.pt and
final_gender.pt, not evaluation (see scripts/evaluate_real_world.py for
the real-world accuracy evaluation tooling that logs against a supplied
ground truth).

Usage: PYTHONPATH=.:scripts ./venv/bin/python3 scripts/live_demo.py
Press 'q' to quit.
"""

import cv2

from age_gender_baseline.constants import resolve_device
from real_world_eval.classify import classify_face, load_classifiers
from src.retailvision.detection import FaceDetector


def main() -> None:
    """Open the camera and run a live face-detection + age/gender preview loop."""
    device = resolve_device()
    print(f"Loading classifiers on device: {device}")
    models = load_classifiers(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    detector = FaceDetector()
    print("Camera opened. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            for x, y, w, h in detector.detect(frame):
                crop = frame[y : y + h, x : x + w]
                predictions = classify_face(models, crop, device)
                age_label, age_conf = predictions["age"]
                gender_label, gender_conf = predictions["gender"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{age_label} ({age_conf:.2f}) / {gender_label} ({gender_conf:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RetailVision - live age/gender demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()