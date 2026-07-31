"""
prepare_widerface.py

CLI entry point for WIDER FACE preparation: parses the official train and
val ground-truth annotations, keeps the official train split as-is,
splits the official val into our own val/test sets, lays the result out
as an Ultralytics YOLOv8 detection dataset, and writes a distribution
report. See scripts/widerface_prep/ for the individual processing steps.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_widerface.py
"""

import json

from widerface_prep.constants import RAW_TRAIN_IMAGES_DIR, RAW_VAL_IMAGES_DIR, REPORT_PATH, TRAIN_GT_PATH, VAL_GT_PATH
from widerface_prep.layout import write_dataset_yaml, write_detection_layout
from widerface_prep.parsing import parse_annotation_file
from widerface_prep.report import build_report
from widerface_prep.splitting import split_val_records


def main() -> None:
    """Run the full WIDER FACE preparation pipeline end to end."""
    print("Parsing official train annotations...")
    train_records, train_stats = parse_annotation_file(TRAIN_GT_PATH, RAW_TRAIN_IMAGES_DIR, "train")
    print(f"  {len(train_records)} images, {sum(len(r['boxes']) for r in train_records)} valid faces")

    print("Parsing official val annotations...")
    val_records, val_stats = parse_annotation_file(VAL_GT_PATH, RAW_VAL_IMAGES_DIR, "val")
    print(f"  {len(val_records)} images, {sum(len(r['boxes']) for r in val_records)} valid faces")

    print("Splitting official val into val/test (stratified by face-count bucket)...")
    splits = {"train": train_records, **split_val_records(val_records)}
    for split_name, split_records in splits.items():
        print(f"  {split_name}: {len(split_records)} images")

    print("Writing YOLOv8 detection layout...")
    write_detection_layout(splits)
    write_dataset_yaml()

    combined_stats = {
        key: train_stats[key] + val_stats[key] for key in train_stats
    }
    report = build_report(splits, combined_stats)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Distribution report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()