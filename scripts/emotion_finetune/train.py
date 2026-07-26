"""
train.py

Runs the yolov8n-cls fine-tuning job on the FER-2013 emotion
classification tree, with augmentation enabled and learning rate, batch
size, epoch count and patience adjusted from the RV-007 baseline defaults.
"""

from ultralytics import YOLO

from .constants import AUGMENTATION, BASE_CHECKPOINT, BATCH, DATA_DIR, EPOCHS, IMGSZ, LR0, PATIENCE, RANDOM_SEED, RUNS_DIR


def train_finetuned(device: str):
    """Fine-tune the yolov8n-cls emotion classifier and return the training results."""
    model = YOLO(BASE_CHECKPOINT)
    results = model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer="SGD",
        lr0=LR0,
        patience=PATIENCE,
        device=device,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name="emotion",
        exist_ok=True,
        **AUGMENTATION,
    )
    return results