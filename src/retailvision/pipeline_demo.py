"""
pipeline_demo.py

First end-to-end demo of the RetailVision pipeline: opens the laptop's
default camera, runs each frame through the FaceDetector, and shows a
live preview with bounding boxes drawn around detected faces. This is
the starting point that later stages (demographics, emotion, tracking,
zone scoring) will be added onto.
"""

import cv2

from .detection import FaceDetector


def main() -> None:
    """Open the camera and run the live face-detection preview loop."""
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("RetailVision - face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
