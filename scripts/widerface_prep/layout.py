"""
layout.py

Materializes a split as an Ultralytics YOLOv8 detection dataset:
data/widerface/processed/images/<split>/<image> (symlinked, same
approach as utkface_prep/fer2013_prep, to avoid duplicating the several
GB of underlying WIDER FACE images) plus a matching
data/widerface/processed/labels/<split>/<image>.txt with one
normalized "class x_center y_center width height" line per retained
face box (empty for images with zero faces after filtering -- valid
"background" images for detection training, not an error case). Also
writes widerface.yaml, the Ultralytics dataset config pointing at this
layout.
"""

import shutil
from pathlib import Path

from .constants import CLASS_NAMES, DATASET_YAML_PATH, PROCESSED_DIR


def _yolo_label_text(record: dict) -> str:
    """Format one record's filtered boxes as YOLO detection label lines."""
    width, height = record["width"], record["height"]
    lines = []
    for x, y, w, h in record["boxes"]:
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        norm_w = w / width
        norm_h = h / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_detection_layout(splits: dict[str, list[dict]]) -> None:
    """Symlink each split's images and write matching YOLO label files."""
    images_root = PROCESSED_DIR / "images"
    labels_root = PROCESSED_DIR / "labels"
    # Wipe first: a re-run with a changed filtering threshold or split ratio
    # shouldn't leave stale images/labels from the previous run mixed in.
    for root in (images_root, labels_root):
        if root.exists():
            shutil.rmtree(root)

    for split_name, records in splits.items():
        image_dir = images_root / split_name
        label_dir = labels_root / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            image_link = image_dir / record["unique_name"]
            if not image_link.exists():
                image_link.symlink_to(record["path"])

            label_path = label_dir / (Path(record["unique_name"]).stem + ".txt")
            label_path.write_text(_yolo_label_text(record))


def write_dataset_yaml() -> None:
    """Write the Ultralytics dataset config (widerface.yaml) for this layout."""
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {PROCESSED_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_block}\n"
    )
    DATASET_YAML_PATH.write_text(content)