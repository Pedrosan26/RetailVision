"""
detection.py

Handles face detection for the RetailVision pipeline. This is the first
detection stage: it takes a single camera frame and returns the bounding
boxes of any faces found in it, using OpenCV's bundled Haar cascade
classifier (no extra model downloads required).
"""

import cv2
import numpy as np


class FaceDetector:
    def __init__(self) -> None:
        """Load the Haar cascade classifier used to detect faces."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError(f"Could not load cascade classifier from {cascade_path}")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a BGR frame, returning (x, y, w, h) boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [tuple(box) for box in faces]
