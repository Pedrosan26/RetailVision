"""
finetune_age_gender.py

CLI entry point that fine-tunes two yolov8n-cls classifiers on
UTKFace (age-group, gender) from the original yolov8n-cls checkpoint, with
augmentation and adjusted learning rate/batch size/epoch count/patience
(see age_gender_finetune/constants.py), and copies the best checkpoint of
each into models/age_gender/. Run scripts/evaluate_age_gender_finetune.py
afterwards to validate against the required accuracy thresholds.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/finetune_age_gender.py
"""

import shutil

from age_gender_finetune.constants import MODEL_OUT_DIR, TASKS, resolve_device
from age_gender_finetune.train import train_finetuned


def main() -> None:
    """Fine-tune the age and gender classifiers sequentially and save their best weights."""
    device = resolve_device()
    print(f"Training device: {device}")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, data_dir in TASKS.items():
        print(f"\n=== Fine-tuning: {task_name} ({data_dir}) ===")
        results = train_finetuned(task_name, data_dir, device)
        best_weights = results.save_dir / "weights" / "best.pt"
        out_path = MODEL_OUT_DIR / f"final_{task_name}.pt"
        shutil.copy2(best_weights, out_path)
        print(f"Saved {task_name} fine-tuned weights to {out_path}")


if __name__ == "__main__":
    main()