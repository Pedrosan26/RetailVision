"""
detection.py

Handles face detection for the RetailVision pipeline. This is the first
detection stage: it takes a single camera frame and returns the bounding
boxes of any faces found in it, using a YOLOv8 detection-mode model
trained from scratch on WIDER FACE and fine-tuned for retail camera
conditions (see RESULTS.md for the training results and for the
real-world evaluation that motivated
replacing this stage's original Haar cascade implementation).
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = REPO_ROOT / "models" / "face_detection" / "final.pt"


class FaceDetector:
    def __init__(self, device: str | None = None) -> None:
        """Load the fine-tuned YOLOv8 face-detection checkpoint onto the given device."""
        self._model = YOLO(str(WEIGHTS_PATH))
        self._device = device

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a BGR frame, returning (x, y, w, h) boxes."""
        result = self._model.predict(source=frame, device=self._device, verbose=False)[0]
        boxes = []
        for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes
