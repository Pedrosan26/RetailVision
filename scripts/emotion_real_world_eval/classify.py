"""
classify.py

Runs the fine-tuned emotion classifier on a single detected face crop.
"""

import numpy as np
from ultralytics import YOLO

from .constants import WEIGHTS_PATH


def load_classifier(device: str) -> YOLO:
    """Load the fine-tuned emotion classifier once, ready for per-frame inference."""
    return YOLO(str(WEIGHTS_PATH))


def classify_face(model: YOLO, face_crop: np.ndarray, device: str) -> tuple[str, float]:
    """Predict (class_name, confidence) for one face crop."""
    result = model.predict(source=face_crop, device=device, verbose=False)[0]
    predicted_index = int(result.probs.top1)
    return result.names[predicted_index], round(float(result.probs.top1conf), 4)