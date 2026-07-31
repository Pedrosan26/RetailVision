"""
evaluate.py

Baseline evaluation for the trained yolov8n face-detection checkpoint:
overall mAP@0.5 / mAP@0.5:0.95 / precision / recall on the held-out test
split via Ultralytics' own detection validator. See official_eval.py for
the official Easy/Medium/Hard recall breakdown.
"""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from widerface_prep.constants import PROCESSED_DIR


def overall_metrics(model: YOLO, device: str) -> dict:
    """Run Ultralytics' detection validator on the full test split."""
    metrics = model.val(data=str(PROCESSED_DIR / "widerface.yaml"), split="test", device=device, verbose=False)
    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def load_loss_curves(results_csv: Path) -> dict:
    """Read per-epoch train/val loss, precision, recall, and mAP from Ultralytics' results.csv."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    curve_columns = [c for c in df.columns if "loss" in c or "mAP" in c or "precision" in c or "recall" in c]
    return {column: df[column].round(5).tolist() for column in ["epoch", *curve_columns] if column in df.columns}