"""
plotting.py

Saves train/val loss and mAP curves to a PNG so they can be visually
inspected for overfitting.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_curves(curves: dict, out_path: Path, title: str = "widerface baseline: loss & mAP curves") -> None:
    """Plot every loss/mAP/precision/recall column against epoch and save to out_path."""
    epochs = curves["epoch"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for column, values in curves.items():
        if column == "epoch":
            continue
        ax.plot(epochs, values, label=column)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)