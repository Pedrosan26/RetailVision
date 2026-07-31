"""
debug_emotion_crops.py

Diagnostic tool: saves every detected face crop from a live session to
disk, with its predicted label and confidence in the filename, so the
crops actually feeding the emotion classifier can be visually inspected
rather than only seeing the aggregate accuracy number. Built to
investigate why close_1m/happy scored far worse (32.3%) than every other
close-range emotion (66-97%) in the real-world evaluation — e.g. to check
whether the face crop is being cut off, blurred, or overexposed at that
range rather than the classifier genuinely failing to recognize the
expression.

Usage:
  PYTHONPATH=.:scripts ./venv/bin/python3 scripts/debug_emotion_crops.py \
      --true-emotion happy --out-dir runs/debug_close_1m_happy

Press 'q' to stop. Not part of the regular evaluation pipeline — this is
a one-off inspection tool.
"""

import argparse
from pathlib import Path

import cv2

from age_gender_baseline.constants import resolve_device
from emotion_real_world_eval.classify import classify_face, load_classifier
from src.retailvision.detection import FaceDetector


def parse_args() -> argparse.Namespace:
    """Parse the ground-truth emotion label and output directory for saved crops."""
    parser = argparse.ArgumentParser(description="Save detected face crops + predictions for visual debugging")
    parser.add_argument("--true-emotion", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Open the camera, save every detected face crop with its prediction, until 'q' is pressed."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    print(f"Loading emotion classifier on device: {device}")
    model = load_classifier(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    detector = FaceDetector()
    print(f"Camera opened. Saving crops to {out_dir}. Press 'q' to stop.")
    frame_index = 0
    saved_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            faces = detector.detect(frame)
            if faces:
                x, y, w, h = faces[0]
                crop = frame[y : y + h, x : x + w]
                pred, conf = classify_face(model, crop, device)
                filename = out_dir / f"{frame_index:04d}_{pred}_{conf:.2f}.jpg"
                cv2.imwrite(str(filename), crop)
                saved_count += 1

                correct = pred == args.true_emotion
                color = (0, 255, 0) if correct else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{pred} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("debug capture - press q to stop", frame)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved_count} face crops (of {frame_index} frames seen) to {out_dir}")


if __name__ == "__main__":
    main()
