"""
manifest.py

Writes each split as a CSV manifest (image path + continuous age label)
rather than a folder-per-class layout, since regression has no discrete
classes to symlink images into.
"""

import csv

from .constants import MANIFEST_DIR


def write_manifests(splits: dict[str, list[dict]]) -> None:
    """Write one CSV manifest per split, with columns: path, age, gender."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        manifest_path = MANIFEST_DIR / f"{split_name}.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "age", "gender"])
            writer.writeheader()
            for record in split_records:
                writer.writerow({"path": str(record["path"]), "age": record["age"], "gender": record["gender"]})