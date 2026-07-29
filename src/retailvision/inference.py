"""
inference.py

Unified real-time inference pipeline: for each camera frame, detects faces
with the existing Haar cascade FaceDetector, then classifies age group,
gender, and emotion for every detected face using the fine-tuned YOLOv8
classifiers (models/age_gender/final_age.pt, final_gender.pt,
models/emotion/final.pt).

Design note on bounding-box matching: age/gender and emotion predictions
are generated from the exact same detected crop, since one shared detector
feeds all three classification heads. There is nothing to reconcile across
models because they never produce independent boxes for the same face --
IoU-based matching only becomes necessary if a future detector swap moves
to per-model detection passes.
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .detection import FaceDetector

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

WEIGHTS = {
    "age": REPO_ROOT / "models" / "age_gender" / "final_age.pt",
    "gender": REPO_ROOT / "models" / "age_gender" / "final_gender.pt",
    "emotion": REPO_ROOT / "models" / "emotion" / "final.pt",
}


def resolve_device() -> str:
    """Pick the fastest available torch device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InferencePipeline:
    """Detects faces and classifies age group, gender, and emotion per frame."""

    def __init__(self, device: str | None = None) -> None:
        """Load the face detector and all three fine-tuned classifiers once."""
        self.device = device or resolve_device()
        self.detector = FaceDetector()
        self.models: dict[str, YOLO] = {task: YOLO(str(path)) for task, path in WEIGHTS.items()}

    def _classify(self, task: str, crop: np.ndarray) -> tuple[str, float]:
        """Run one fine-tuned classifier on a face crop, returning (label, confidence)."""
        result = self.models[task].predict(source=crop, device=self.device, verbose=False)[0]
        predicted_index = int(result.probs.top1)
        return result.names[predicted_index], round(float(result.probs.top1conf), 4)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a BGR frame and classify age group, gender, and emotion for each."""
        detections = []
        for x, y, w, h in self.detector.detect(frame):
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            age_group, age_conf = self._classify("age", crop)
            gender, gender_conf = self._classify("gender", crop)
            emotion, emotion_conf = self._classify("emotion", crop)

            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "age_group": age_group,
                    "gender": gender,
                    "emotion": emotion,
                    "confidence": {
                        "age": age_conf,
                        "gender": gender_conf,
                        "emotion": emotion_conf,
                    },
                }
            )
        return detections