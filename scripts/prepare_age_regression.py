"""
prepare_age_regression.py

CLI entry point for RET-31 data preparation: parses raw UTKFace filenames,
builds the same stratified 70/15/15 split as the classification pipeline
(RV-002), and writes CSV manifests (image path + continuous age) instead
of a folder-per-class layout, since regression has no discrete classes.
Reuses utkface_prep's scanning/splitting/report logic directly.

Usage: PYTHONPATH=scripts ./venv/bin/python3 scripts/prepare_age_regression.py
"""

import json

from age_regression_prep.constants import MANIFEST_DIR, REPORT_PATH
from age_regression_prep.manifest import write_manifests
from utkface_prep.parsing import collect_records
from utkface_prep.report import build_report
from utkface_prep.splitting import stratified_split


def main() -> None:
    """Run the age-regression manifest preparation pipeline end to end."""
    print("Scanning raw UTKFace directories...")
    records, skipped = collect_records()
    print(f"Found {len(records)} valid images, skipped {len(skipped)} malformed/duplicate filenames.")

    print("Building stratified 70/15/15 train/val/test split (same methodology as RV-002)...")
    splits = stratified_split(records)
    for split_name, split_records in splits.items():
        print(f"  {split_name}: {len(split_records)} images")

    print(f"Writing CSV manifests to {MANIFEST_DIR}...")
    write_manifests(splits)

    report = build_report(records, skipped, splits)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Distribution report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()