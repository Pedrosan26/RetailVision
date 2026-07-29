"""
train.py

Runs a single yolov8n-cls baseline training job on one of the two UTKFace
classification trees (age or gender), using Ultralytics default hyperparameters throughout .
"""

from pathlib import Path

from ultralytics import YOLO

from .constants import BASE_CHECKPOINT, EPOCHS, IMGSZ, RANDOM_SEED, RUNS_DIR


def train_baseline(task_name: str, data_dir: Path, device: str):
    """Train a yolov8n-cls baseline for one task and return the training results."""
    model = YOLO(BASE_CHECKPOINT)
    results = model.train(
        data=str(data_dir),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=device,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name=task_name,
        exist_ok=True,
    )
    return results