"""
splitting.py

Splits parsed UTKFace records into train/val/test sets, stratifying by
the (age_group, gender) pair so that each split preserves the overall
class distribution rather than skewing toward whichever stratum happens
to shuffle first.
"""

import random
from collections import defaultdict

from .constants import RANDOM_SEED, SPLIT_RATIOS


def stratified_split(records: list[dict]) -> dict[str, list[dict]]:
    """Split records into train/val/test, preserving each (age_group, gender) stratum's proportions."""
    rng = random.Random(RANDOM_SEED)
    strata = defaultdict(list)
    for record in records:
        strata[(record["age_group"], record["gender"])].append(record)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for stratum_records in strata.values():
        rng.shuffle(stratum_records)
        n = len(stratum_records)
        n_train = round(n * SPLIT_RATIOS["train"])
        n_val = round(n * SPLIT_RATIOS["val"])
        splits["train"].extend(stratum_records[:n_train])
        splits["val"].extend(stratum_records[n_train : n_train + n_val])
        splits["test"].extend(stratum_records[n_train + n_val :])
    return splits
