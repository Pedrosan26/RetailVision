"""
splitting.py

The official WIDER FACE train split is kept as-is for our train split.
The official val split (the only other portion with public ground truth)
is divided into our own val and test sets, stratified by a face-count
bucket so neither ends up skewed toward sparse or crowd-dense images
relative to the other.
"""

import random
from collections import defaultdict

from .constants import FACE_COUNT_BUCKETS, RANDOM_SEED, VAL_TEST_SPLIT_RATIO


def bucket_face_count(n: int) -> str:
    """Map a record's (filtered) face count to its stratification bucket label."""
    for low, high, label in FACE_COUNT_BUCKETS:
        if low <= n <= high:
            return label
    raise ValueError(f"Face count {n} did not match any bucket")


def split_val_records(val_records: list[dict]) -> dict[str, list[dict]]:
    """Split the official val records into our val/test sets, stratified by face-count bucket."""
    rng = random.Random(RANDOM_SEED)
    strata = defaultdict(list)
    for record in val_records:
        strata[bucket_face_count(len(record["boxes"]))].append(record)

    splits: dict[str, list[dict]] = {"val": [], "test": []}
    for stratum_records in strata.values():
        rng.shuffle(stratum_records)
        n_val = round(len(stratum_records) * VAL_TEST_SPLIT_RATIO)
        splits["val"].extend(stratum_records[:n_val])
        splits["test"].extend(stratum_records[n_val:])
    return splits