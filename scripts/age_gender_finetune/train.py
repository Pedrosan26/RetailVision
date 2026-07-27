"""
train.py

Runs a single yolov8n-cls fine-tuning job on one of the two UTKFace
classification trees (age or gender), with augmentation enabled and
learning rate, batch size, epoch count and patience adjusted from the
the baseline defaults.
"""

from pathlib import Path

from ultralytics import YOLO

from .constants import AUGMENTATION, BASE_CHECKPOINT, BATCH, EPOCHS, IMGSZ, LR0, PATIENCE, RANDOM_SEED, RUNS_DIR


def train_finetuned(task_name: str, data_dir: Path, device: str):
    """Train a fine-tuned yolov8n-cls classifier for one task and return the training results."""
    model = YOLO(BASE_CHECKPOINT)
    results = model.train(
        data=str(data_dir),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer="SGD",
        lr0=LR0,
        patience=PATIENCE,
        device=device,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name=task_name,
        exist_ok=True,
        **AUGMENTATION,
    )
    return results