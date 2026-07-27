"""
train_emotion_baseline.py

CLI entry point that trains a yolov8n-cls baseline classifier on
FER-2013 (7 emotion classes) using Ultralytics default hyperparameters,
and copies the best checkpoint to models/emotion/baseline.pt. This is a
long-running job (100 epochs) intended to run to completion unattended;
run scripts/evaluate_emotion_baseline.py afterwards to compute the
baseline metrics report.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/train_emotion_baseline.py
"""

import shutil

from emotion_baseline.constants import MODEL_OUT_DIR, WEIGHTS_PATH, resolve_device
from emotion_baseline.train import train_baseline


def main() -> None:
    """Train the emotion baseline and save its best weights."""
    device = resolve_device()
    print(f"Training device: {device}")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = train_baseline(device)
    best_weights = results.save_dir / "weights" / "best.pt"
    shutil.copy2(best_weights, WEIGHTS_PATH)
    print(f"Saved emotion baseline weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()