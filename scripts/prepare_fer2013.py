"""
prepare_fer2013.py

CLI entry point for FER-2013 preparation: verifies the raw train/test
folders match the expected seven emotion classes and image format,
carves a stratified validation split out of the official train set
(test is kept untouched), lays out a YOLOv8 classification dataset, and
writes a class-distribution report.

See scripts/fer2013_prep/ for the individual processing steps.

Usage: ./venv/bin/python3 scripts/prepare_fer2013.py
"""

import json

from fer2013_prep.constants import PROCESSED_DIR, REPORT_PATH
from fer2013_prep.layout import write_classification_layout
from fer2013_prep.report import build_report
from fer2013_prep.scanning import collect_records
from fer2013_prep.splitting import carve_validation_split


def main() -> None:
    """Run the full FER-2013 preparation pipeline end to end."""
    print("Scanning raw FER-2013 directories...")
    records, skipped = collect_records()
    print(f"Found {len(records)} valid images, skipped {len(skipped)} malformed images.")

    print("Carving a stratified validation split out of train/ (test/ kept untouched)...")
    splits = carve_validation_split(records)
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