"""
splitting.py

Derives train/val/test splits from FER-2013's records. The official test
set is kept fully intact as the held-out evaluation set; a validation
split is carved out of the official train set only, stratified per
emotion class, so every class is proportionally represented in val
regardless of how rare it is in the full dataset (relevant for the
severely underrepresented 'disgust' class).
"""

import random
from collections import defaultdict

from .constants import RANDOM_SEED, VAL_FRACTION


def carve_validation_split(records: list[dict]) -> dict[str, list[dict]]:
    """Split records into train/val/test: test passes through untouched, val is carved from train."""
    rng = random.Random(RANDOM_SEED)

    train_by_class = defaultdict(list)
    test_records = []
    for record in records:
        if record["raw_split"] == "test":
            test_records.append(record)
        else:
            train_by_class[record["emotion"]].append(record)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": test_records}
    for class_records in train_by_class.values():
        rng.shuffle(class_records)
        n_val = round(len(class_records) * VAL_FRACTION)
        splits["val"].extend(class_records[:n_val])
        splits["train"].extend(class_records[n_val:])
    return splits