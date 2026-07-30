"""
train.py

Runs the yolov8n baseline detection training job on the prepared WIDER
FACE dataset, using Ultralytics default hyperparameters throughout (no
augmentation tuning) with two exceptions,
neither of which changes what the model learns -- both are throughput/
stability adjustments, not tuning choices:

- cache="disk". WIDER FACE's source images are much larger than
  anything else this project has trained on (commonly 1024px wide,
  1024-1500px tall, vs. FER-2013's 48x48 or UTKFace's small pre-cropped
  chips), so with caching off every epoch re-reads and re-decodes every
  image from scratch. cache="disk" decodes and resizes each image once,
  caching the result to disk, and skips that repeated decode cost on
  every later epoch -- cache="ram" is faster still, but needs enough
  memory to hold the whole decoded train set at once, which isn't
  guaranteed across the range of machines this might run on. Caching
  only affects how fast images are read, not what the model sees: the
  same decoded pixels flow into the same (seeded) augmentation pipeline
  either way.
- batch=8, down from Ultralytics' detection-mode default of 16 (see
  BATCH in constants.py for why: WIDER FACE's crowd-scene images can
  carry up to ~2,000 ground-truth boxes in one frame, which spiked
  label-assignment memory far past typical detection datasets and
  crashed a first attempt at batch=16).
"""

from ultralytics import YOLO

from .constants import BASE_CHECKPOINT, BATCH, DATASET_YAML, EPOCHS, IMGSZ, RANDOM_SEED, RUNS_DIR


def train_baseline(device: str):
    """Train the yolov8n face-detection baseline and return the training results."""
    model = YOLO(BASE_CHECKPOINT)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        seed=RANDOM_SEED,
        project=str(RUNS_DIR),
        name="widerface",
        exist_ok=True,
        cache="disk",
    )
    return results
