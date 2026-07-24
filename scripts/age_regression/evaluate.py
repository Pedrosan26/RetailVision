"""
evaluate.py

Evaluates the trained age-regression model on the held-out test manifest:
overall MAE, plus MAE bucketed into the RV-005 4-bin age ranges for a
per-age-group breakdown (RET-31's acceptance criteria), even though the
model itself never saw those bins during training.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .constants import BATCH_SIZE, MANIFEST_DIR, REPORT_AGE_BUCKETS, WEIGHTS_PATH
from .dataset import EVAL_TRANSFORM, AgeRegressionDataset
from .model import build_model


def _bucket_for_age(age: float) -> str:
    """Map a raw age to its RV-005-style report bucket, for per-group MAE breakdown."""
    for low, high, label in REPORT_AGE_BUCKETS:
        if low <= age <= high:
            return label
    return REPORT_AGE_BUCKETS[-1][2]


def evaluate_regression(device: str) -> dict:
    """Load the best checkpoint and compute overall + per-age-bucket MAE on the test split."""
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    test_ds = AgeRegressionDataset(MANIFEST_DIR / "test.csv", EVAL_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    true_ages: list[float] = []
    predicted_ages: list[float] = []
    with torch.no_grad():
        for images, ages in test_loader:
            images = images.to(device)
            predictions = model(images).squeeze(1).cpu()
            true_ages.extend(ages.tolist())
            predicted_ages.extend(predictions.tolist())

    overall_mae = sum(abs(t - p) for t, p in zip(true_ages, predicted_ages)) / len(true_ages)

    bucket_errors: dict[str, list[float]] = {label: [] for _, _, label in REPORT_AGE_BUCKETS}
    for true_age, predicted_age in zip(true_ages, predicted_ages):
        bucket_errors[_bucket_for_age(true_age)].append(abs(true_age - predicted_age))

    per_bucket_mae = {
        label: {
            "mae": round(sum(errors) / len(errors), 4) if errors else None,
            "support": len(errors),
        }
        for label, errors in bucket_errors.items()
    }

    return {
        "weights": str(WEIGHTS_PATH),
        "overall_mae": round(overall_mae, 4),
        "per_age_group_mae": per_bucket_mae,
    }