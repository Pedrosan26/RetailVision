"""
report.py

Builds the distribution report documenting class balance across age
group, gender, and race, both overall and per split. This is what
Week 6's imbalance documentation (RV-002 acceptance criteria) is
generated from, so accuracy gaps on underrepresented groups are
explainable rather than a surprise during evaluation.
"""

from collections import defaultdict


def _counts_by(key: str, subset: list[dict]) -> dict[str, int]:
    """Count how many records in subset fall into each value of the given label key."""
    counter: dict[str, int] = defaultdict(int)
    for record in subset:
        counter[record[key]] += 1
    return dict(sorted(counter.items()))


def build_report(records: list[dict], skipped: list[str], splits: dict[str, list[dict]]) -> dict:
    """Compute class-distribution counts (overall, per split, per race) for documentation."""
    return {
        "total_valid_images": len(records),
        "total_skipped_malformed": len(skipped),
        "skipped_filenames": skipped,
        "overall": {
            "age_group": _counts_by("age_group", records),
            "gender": _counts_by("gender", records),
            "race": _counts_by("race", records),
        },
        "by_split": {
            split_name: {
                "count": len(split_records),
                "age_group": _counts_by("age_group", split_records),
                "gender": _counts_by("gender", split_records),
                "race": _counts_by("race", split_records),
            }
            for split_name, split_records in splits.items()
        },
    }
