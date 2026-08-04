"""
finetune_widerface.py

CLI entry point that fine-tunes the yolov8n face detector on WIDER FACE
with retail-tuned augmentation, and copies the best checkpoint to
models/face_detection/final.pt. Long-running job intended to run to
completion unattended; run scripts/evaluate_widerface_finetune.py
afterwards to compute the fine-tune metrics report.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/finetune_widerface.py
"""

import shutil

from widerface_finetune.constants import MODEL_OUT_DIR, WEIGHTS_PATH, resolve_device
from widerface_finetune.train import train_finetuned


def main() -> None:
    """Fine-tune the WIDER FACE face detector and save its best weights."""
    device = resolve_device()
    print(f"Training device: {device}")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = train_finetuned(device)
    best_weights = results.save_dir / "weights" / "best.pt"
    shutil.copy2(best_weights, WEIGHTS_PATH)
    print(f"Saved WIDER FACE fine-tuned weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
