"""
train_age_regression.py

CLI entry point for RET-31: trains the ResNet18-based age-regression
model on the UTKFace manifests produced by scripts/prepare_age_regression.py,
with early stopping on validation MAE. Saves the best checkpoint to
models/age_gender/regression_age.pt and a loss/MAE curve plot. Run
scripts/evaluate_age_regression.py afterwards for the held-out test MAE.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/train_age_regression.py
"""

from age_gender_baseline.constants import resolve_device
from age_regression.constants import RUNS_DIR
from age_regression.plotting import save_training_curves
from age_regression.train import train_regression


def main() -> None:
    """Train the age-regression model and save weights + a training-curve plot."""
    device = resolve_device()
    print(f"Training device: {device}")

    history = train_regression(device)

    plot_path = RUNS_DIR / "training_curves.png"
    save_training_curves(history, plot_path)
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()