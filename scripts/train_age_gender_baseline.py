"""
train_age_gender_baseline.py

CLI entry point for RV-004: trains two yolov8n-cls baseline classifiers on
UTKFace (age-group, gender) using Ultralytics default hyperparameters, and
copies the best checkpoint of each into models/age_gender/. This is a long
running job (100 epochs x 2 tasks) intended to run to completion unattended;
run scripts/evaluate_age_gender_baseline.py afterwards to compute the
baseline metrics report.

Usage: ./venv/bin/python3 scripts/train_age_gender_baseline.py
"""

import shutil

from age_gender_baseline.constants import MODEL_OUT_DIR, TASKS, resolve_device
from age_gender_baseline.train import train_baseline


def main() -> None:
    """Train the age and gender baselines sequentially and save their best weights."""
    device = resolve_device()
    print(f"Training device: {device}")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, data_dir in TASKS.items():
        print(f"\n=== Training baseline: {task_name} ({data_dir}) ===")
        results = train_baseline(task_name, data_dir, device)
        best_weights = results.save_dir / "weights" / "best.pt"
        out_path = MODEL_OUT_DIR / f"baseline_{task_name}.pt"
        shutil.copy2(best_weights, out_path)
        print(f"Saved {task_name} baseline weights to {out_path}")


if __name__ == "__main__":
    main()
