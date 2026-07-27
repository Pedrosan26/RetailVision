"""
finetune_emotion.py

CLI entry point that fine-tunes the yolov8n-cls emotion classifier
on FER-2013 from the original yolov8n-cls checkpoint, with augmentation
and adjusted learning rate/batch size/epoch count/patience (see
emotion_finetune/constants.py), and copies the best checkpoint into
models/emotion/final.pt. Run scripts/evaluate_emotion_finetune.py
afterwards to validate against the required per-class recall thresholds.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/finetune_emotion.py
"""

import shutil

from emotion_finetune.constants import MODEL_OUT_DIR, WEIGHTS_PATH, resolve_device
from emotion_finetune.train import train_finetuned


def main() -> None:
    """Fine-tune the emotion classifier and save its best weights."""
    device = resolve_device()
    print(f"Training device: {device}")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = train_finetuned(device)
    best_weights = results.save_dir / "weights" / "best.pt"
    shutil.copy2(best_weights, WEIGHTS_PATH)
    print(f"Saved fine-tuned emotion weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()