"""
train.py

Runs the yolov8n fine-tuning job on the prepared WIDER FACE dataset,
with augmentation tuned for retail camera conditions, a lower learning
rate, and early stopping enabled -- see constants.py for the rationale
behind each adjustment from the baseline.
"""

from ultralytics import YOLO

from .constants import AUGMENTATION, BASE_CHECKPOINT, BATCH, DATASET_YAML, EPOCHS, IMGSZ, LR0, PATIENCE, RANDOM_SEED, RUNS_DIR


def train_finetuned(device: str):
    """Train the fine-tuned yolov8n face detector and return the training results."""
    model = YOLO(BASE_CHECKPOINT)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer="SGD",
        lr0=LR0,
        patience=PATIENCE,
        device=device,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name="widerface",
        exist_ok=True,
        cache="disk",
        **AUGMENTATION,
    )
    return results
