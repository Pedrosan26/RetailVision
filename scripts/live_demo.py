"""
live_demo.py

Live webcam preview with the current age/gender classifiers, the
age-regression model, AND the emotion classifier running on every
detected face: opens the camera, runs FaceDetector, classifies each face
crop, and draws the predicted age bin + continuous age estimate + gender
+ emotion + confidences on the preview window. This mirrors the intended
production split (see "Age regression" in README.md): the classifier bin
is the analytics-facing output, the regression estimate is the
finer-grained display-facing output. No ground truth needed - this is for
casually trying out whatever weights currently sit at
models/age_gender/final_age.pt, final_gender.pt, regression_age.pt, and
models/emotion/final.pt, not evaluation (see scripts/evaluate_real_world.py
for the real-world accuracy evaluation tooling that logs against a
supplied ground truth).

Usage: PYTHONPATH=.:scripts ./venv/bin/python3 scripts/live_demo.py
Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO

from age_gender_baseline.constants import resolve_device
from age_regression.predict import load_regression_model, predict_age
from emotion_finetune.constants import WEIGHTS_PATH as EMOTION_WEIGHTS_PATH
from real_world_eval.classify import classify_face, load_classifiers
from src.retailvision.detection import FaceDetector


def main() -> None:
    """Open the camera and run a live face-detection + age/gender/emotion preview loop."""
    device = resolve_device()
    print(f"Loading classifiers and regression model on device: {device}")
    models = load_classifiers(device)
    regression_model = load_regression_model(device)
    emotion_model = YOLO(str(EMOTION_WEIGHTS_PATH))

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
                estimated_age = predict_age(regression_model, crop, device)

                emotion_result = emotion_model.predict(source=crop, device=device, verbose=False)[0]
                emotion_idx = int(emotion_result.probs.top1)
                emotion_label = emotion_result.names[emotion_idx]
                emotion_conf = float(emotion_result.probs.top1conf)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                line1 = (
                    f"{age_label} (~{estimated_age:.0f}y, {age_conf:.2f}) / "
                    f"{gender_label} ({gender_conf:.2f})"
                )
                line2 = f"{emotion_label} ({emotion_conf:.2f})"
                cv2.putText(frame, line1, (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, line2, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("RetailVision - live age/gender/emotion demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()