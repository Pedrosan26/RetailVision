"""
layout.py

Materializes a split as a YOLOv8 classification dataset:
data/fer2013/processed/<split>/<emotion>/<image>. Images are symlinked
rather than copied to avoid duplicating the raw FER-2013 JPEGs.
"""

from .constants import PROCESSED_DIR


def write_classification_layout(splits: dict[str, list[dict]]) -> None:
    """Symlink each image into processed/<split>/<emotion>/ folders."""
    for split_name, split_records in splits.items():
        for record in split_records:
            class_dir = PROCESSED_DIR / split_name / record["emotion"]
            class_dir.mkdir(parents=True, exist_ok=True)
            link_path = class_dir / f"{record['raw_split']}_{record['path'].name}"
            if not link_path.exists():
                link_path.symlink_to(record["path"])
