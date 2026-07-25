"""
classify.py

Runs the RV-005 fine-tuned age/gender classifiers on a single detected
face crop.
"""

import numpy as np
from ultralytics import YOLO

from .constants import WEIGHTS


def load_classifiers(device: str) -> dict[str, YOLO]:
    """Load both fine-tuned classifiers once, ready for per-frame inference."""
    return {task: YOLO(str(path)) for task, path in WEIGHTS.items()}


def classify_face(models: dict[str, YOLO], face_crop: np.ndarray, device: str) -> dict[str, tuple[str, float]]:
    """Predict (class_name, confidence) for age and gender on one face crop."""
    predictions: dict[str, tuple[str, float]] = {}
    for task, model in models.items():
        result = model.predict(source=face_crop, device=device, verbose=False)[0]
        predicted_index = int(result.probs.top1)
        predictions[task] = (result.names[predicted_index], round(float(result.probs.top1conf), 4))
    return predictions