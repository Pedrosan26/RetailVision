"""
train.py

Runs the yolov8n-cls baseline training job on the FER-2013 emotion
classification tree, using Ultralytics default hyperparameters throughout.
"""

from ultralytics import YOLO

from .constants import BASE_CHECKPOINT, DATA_DIR, EPOCHS, IMGSZ, RANDOM_SEED, RUNS_DIR


def train_baseline(device: str):
    """Train the yolov8n-cls emotion baseline and return the training results."""
    model = YOLO(BASE_CHECKPOINT)
    results = model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=device,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name="emotion",
        exist_ok=True,
    )
    return results