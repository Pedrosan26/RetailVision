"""
train_widerface_baseline.py

CLI entry point that trains a yolov8n detection-mode baseline face
detector on the prepared WIDER FACE dataset using Ultralytics default
hyperparameters, and copies the best checkpoint to
models/face_detection/baseline.pt. This is a long-running job (100
epochs, detection mode is heavier per-epoch than the classification
baselines) intended to run to completion unattended; run
scripts/evaluate_widerface_baseline.py afterwards to compute the
baseline metrics report.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/train_widerface_baseline.py
"""

import shutil

from widerface_baseline.constants import MODEL_OUT_DIR, WEIGHTS_PATH, resolve_device
from widerface_baseline.train import train_baseline


def main() -> None:
    """Train the WIDER FACE baseline detector and save its best weights."""
    device = resolve_device()
    print(f"Training device: {device}")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = train_baseline(device)
    best_weights = results.save_dir / "weights" / "best.pt"
    shutil.copy2(best_weights, WEIGHTS_PATH)
    print(f"Saved WIDER FACE baseline weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()