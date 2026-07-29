"""
plotting.py

Saves train/val loss and val MAE curves to a PNG so training can be
visually inspected for overfitting, matching the convention used for the
YOLOv8 classifiers' loss curve plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_curves(history: dict, out_path: Path) -> None:
    """Plot train/val loss and val MAE against epoch and save to out_path."""
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in ("train_loss", "val_loss", "val_mae"):
        ax.plot(epochs, history[column], label=column)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Age regression: loss & MAE curves")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)