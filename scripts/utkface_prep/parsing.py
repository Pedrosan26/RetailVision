"""
parsing.py

Parses UTKFace's filename-encoded labels (age_gender_race_date.jpg.chip.jpg)
into the project's age-group and gender classes, and scans the raw source
directories into a list of valid, deduplicated records plus a list of
filenames that were skipped for being malformed or duplicated.
"""

import re
from pathlib import Path

from .constants import AGE_BINS, GENDER_LABELS, RACE_LABELS, RAW_DIRS

FILENAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_\d+\.jpg\.chip\.jpg$")


def bin_age(age: int) -> str:
    """Map a raw integer age to one of the four project age-group labels."""
    for low, high, label in AGE_BINS:
        if low <= age <= high:
            return label
    raise ValueError(f"Age {age} did not match any bin")


def parse_filename(path: Path) -> dict | None:
    """Parse a UTKFace filename into age/gender/race labels, or None if malformed."""
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    age, gender, race = (int(g) for g in match.groups())
    if gender not in GENDER_LABELS:
        return None
    return {
        "path": path,
        "age": age,
        "age_group": bin_age(age),
        "gender": GENDER_LABELS[gender],
        "race": RACE_LABELS.get(race, "Unknown"),
    }


def collect_records() -> tuple[list[dict], list[str]]:
    """Scan all raw source directories, returning valid records and a list of skipped filenames."""
    records = []
    skipped = []
    seen_names = set()
    for raw_dir in RAW_DIRS:
        for path in sorted(raw_dir.glob("*.jpg")):
            record = parse_filename(path)
            if record is None:
                skipped.append(f"{raw_dir.name}/{path.name}")
                continue
            # Prefix with source dir to avoid collisions between UTKFace/ and crop_part1/.
            unique_name = f"{raw_dir.name}_{path.name}"
            if unique_name in seen_names:
                skipped.append(f"{raw_dir.name}/{path.name} (duplicate)")
                continue
            seen_names.add(unique_name)
            record["unique_name"] = unique_name
            records.append(record)
    return records, skipped
