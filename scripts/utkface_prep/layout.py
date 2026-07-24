"""
layout.py

Materializes a stratified split as a YOLOv8 classification dataset:
data/utkface/processed/<task>/<split>/<class>/<image>. Images are
symlinked rather than copied so the age and gender folder trees don't
duplicate the ~1-2GB of underlying UTKFace image data.
"""

from .constants import PROCESSED_DIR


def write_classification_layout(splits: dict[str, list[dict]]) -> None:
    """Symlink each image into age/<split>/<class>/ and gender/<split>/<class>/ folders."""
    for task, label_key in (("age", "age_group"), ("gender", "gender")):
        for split_name, split_records in splits.items():
            for record in split_records:
                class_dir = PROCESSED_DIR / task / split_name / record[label_key]
                class_dir.mkdir(parents=True, exist_ok=True)
                link_path = class_dir / record["unique_name"]
                if not link_path.exists():
                    link_path.symlink_to(record["path"])
