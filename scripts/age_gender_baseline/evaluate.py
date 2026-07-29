"""
evaluate.py

Baseline evaluation for a trained yolov8n-cls checkpoint. YOLOv8
classification mode reports top1/top5 accuracy rather than mAP@0.5,
precision, recall or F1 (those are detection-mode metrics), so per-class
precision/recall/F1 are computed here directly from predictions on the
held-out test split via scikit-learn, and top1/top5 accuracy is reported
as the classification-appropriate substitute for mAP@0.5.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO


def run_test_predictions(model: YOLO, data_dir: Path) -> tuple[list[str], list[str]]:
    """Predict the top1 class for every image in the test split and return (y_true, y_pred)."""
    class_names = sorted(p.name for p in (data_dir / "test").iterdir() if p.is_dir())
    y_true: list[str] = []
    y_pred: list[str] = []
    for class_name in class_names:
        image_paths = sorted((data_dir / "test" / class_name).iterdir())
        if not image_paths:
            continue
        results = model.predict(source=[str(p) for p in image_paths], verbose=False)
        for result in results:
            predicted_index = int(result.probs.top1)
            y_true.append(class_name)
            y_pred.append(result.names[predicted_index])
    return y_true, y_pred


def per_class_metrics(y_true: list[str], y_pred: list[str], class_names: list[str]) -> dict:
    """Compute precision/recall/F1 per class from predictions on the test split."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    return {
        class_name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, class_name in enumerate(class_names)
    }


def top_k_accuracy(model: YOLO, data_dir: Path, device: str) -> dict:
    """Run Ultralytics' own classification validator on the test split for top1/top5 accuracy."""
    metrics = model.val(data=str(data_dir), split="test", device=device, verbose=False)
    return {
        "top1_accuracy": round(float(metrics.top1), 4),
        "top5_accuracy": round(float(metrics.top5), 4),
    }


def load_loss_curves(results_csv: Path) -> dict:
    """Read per-epoch train/val loss and accuracy from Ultralytics' results.csv."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    curve_columns = [c for c in df.columns if "loss" in c or "accuracy" in c]
    return {column: df[column].round(5).tolist() for column in ["epoch", *curve_columns] if column in df.columns}