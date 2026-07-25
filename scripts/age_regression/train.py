"""
train.py

Standard PyTorch training loop for the age-regression model: L1 loss
(directly optimizes MAE, the metric this model is evaluated on, and is
more robust to any mislabeled/outlier ages in the scraped UTKFace data
than MSE), Adam, early stopping on validation MAE, and a per-epoch
results.csv matching the loss-curve logging convention used by the
YOLOv8 classifiers (train/val loss, val MAE) for consistency across the
project's model docs. Prints live per-batch progress (batches/sec,
running loss) during each phase, since this plain PyTorch loop has no
built-in progress bar the way Ultralytics' training does.
"""

import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import BATCH_SIZE, EPOCHS, LR, MANIFEST_DIR, PATIENCE, RANDOM_SEED, RUNS_DIR, WEIGHTS_PATH
from .dataset import EVAL_TRANSFORM, TRAIN_TRANSFORM, AgeRegressionDataset
from .model import build_model


def _run_epoch(model, loader, device, phase: str, optimizer=None) -> tuple[float, float]:
    """Run one epoch (training if optimizer given, else evaluation-only); return (loss, MAE)."""
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.L1Loss()

    total_loss = 0.0
    total_abs_error = 0.0
    total_examples = 0
    num_batches = len(loader)
    start = time.time()
    with torch.set_grad_enabled(is_train):
        for batch_index, (images, ages) in enumerate(loader, start=1):
            images, ages = images.to(device), ages.to(device)
            predictions = model(images).squeeze(1)
            loss = criterion(predictions, ages)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = ages.size(0)
            total_loss += loss.item() * batch_size
            total_abs_error += torch.sum(torch.abs(predictions - ages)).item()
            total_examples += batch_size

            elapsed = time.time() - start
            rate = batch_index / elapsed if elapsed > 0 else 0.0
            print(
                f"\r  {phase} {batch_index}/{num_batches}  "
                f"loss={total_loss / total_examples:.4f}  {rate:.1f} batch/s",
                end="",
                flush=True,
            )
    print()

    return total_loss / total_examples, total_abs_error / total_examples


def train_regression(device: str) -> dict:
    """Train the age-regression model with early stopping; return the run history."""
    torch.manual_seed(RANDOM_SEED)

    train_ds = AgeRegressionDataset(MANIFEST_DIR / "train.csv", TRAIN_TRANSFORM)
    val_ds = AgeRegressionDataset(MANIFEST_DIR / "val.csv", EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RUNS_DIR / "results.csv"
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": []}

    best_val_mae = float("inf")
    epochs_without_improvement = 0

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mae"])

        for epoch in range(1, EPOCHS + 1):
            print(f"epoch {epoch}/{EPOCHS}")
            train_loss, _ = _run_epoch(model, train_loader, device, "train", optimizer)
            val_loss, val_mae = _run_epoch(model, val_loader, device, "val")

            writer.writerow([epoch, round(train_loss, 5), round(val_loss, 5), round(val_mae, 5)])
            f.flush()
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            print(f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mae={val_mae:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                epochs_without_improvement = 0
                WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), WEIGHTS_PATH)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"No val MAE improvement for {PATIENCE} epochs, stopping early.")
                    break

    print(f"Best val MAE: {best_val_mae:.4f}. Weights saved to {WEIGHTS_PATH}")
    return history