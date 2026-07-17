"""
prepare_utkface.py

CLI entry point for UTKFace preparation: parses raw filenames, builds a
stratified 70/15/15 train/val/test split, lays out the images as a
YOLOv8 classification dataset, and writes a class-distribution report.
See scripts/utkface_prep/ for the individual processing steps.

Usage: ./venv/bin/python3 scripts/prepare_utkface.py
"""

import json

from utkface_prep.constants import PROCESSED_DIR, REPORT_PATH
from utkface_prep.layout import write_classification_layout
from utkface_prep.parsing import collect_records
from utkface_prep.report import build_report
from utkface_prep.splitting import stratified_split


def main() -> None:
    """Run the full UTKFace preparation pipeline end to end."""
    print("Scanning raw UTKFace directories...")
    records, skipped = collect_records()
    print(f"Found {len(records)} valid images, skipped {len(skipped)} malformed/duplicate filenames.")

    print("Building stratified 70/15/15 train/val/test split...")
    splits = stratified_split(records)
    for split_name, split_records in splits.items():
        print(f"  {split_name}: {len(split_records)} images")

    print(f"Writing YOLOv8 classification layout to {PROCESSED_DIR}...")
    write_classification_layout(splits)

    report = build_report(records, skipped, splits)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Distribution report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
