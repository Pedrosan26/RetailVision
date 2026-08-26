"""
predict.py

Runs the trained age-regression model on a single face crop (a BGR numpy
array, as returned by FaceDetector), for live/interactive use outside of
the batch evaluate.py pipeline.
"""

import numpy as np
import torch
from PIL import Image

from .constants import WEIGHTS_PATH
from .dataset import EVAL_TRANSFORM
from .model import build_model


def load_regression_model(device: str) -> torch.nn.Module:
    """Load the trained regression weights, ready for per-frame inference."""
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_age(model: torch.nn.Module, face_crop_bgr: np.ndarray, device: str) -> float:
    """Predict a continuous age for one BGR face crop."""
    image = Image.fromarray(face_crop_bgr[:, :, ::-1])
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).squeeze().item()