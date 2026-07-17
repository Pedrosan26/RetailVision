"""
report.py

Builds the distribution report documenting class balance across the
seven emotion classes, both overall and per split. FER-2013's 'disgust'
class is known to be severely underrepresented, and 'fear' is known to
be noisy/confusable with other classes even though its raw count is not
small — this report flags both explicitly rather than leaving imbalance
to be discovered during training.
"""

from collections import defaultdict

KNOWN_PROBLEM_CLASSES = {
    "disgust": "severely underrepresented relative to other classes",
    "fear": "not underrepresented by count, but documented in the literature as noisy "
    "and frequently confused with 'sad' and 'surprise'",
}


def _counts_by_emotion(subset: list[dict]) -> dict[str, int]:
    """Count how many records in subset belong to each emotion class."""
    counter: dict[str, int] = defaultdict(int)
    for record in subset:
        counter[record["emotion"]] += 1
    return dict(sorted(counter.items()))


def build_report(records: list[dict], skipped: list[str], splits: dict[str, list[dict]]) -> dict:
    """Compute class-distribution counts (overall, per split) for documentation."""
    return {
        "total_valid_images": len(records),
        "total_skipped_invalid": len(skipped),
        "skipped_filenames": skipped,
        "known_problem_classes": KNOWN_PROBLEM_CLASSES,
        "overall": _counts_by_emotion(records),
        "by_split": {
            split_name: {
                "count": len(split_records),
                "emotion": _counts_by_emotion(split_records),
            }
            for split_name, split_records in splits.items()
        },
    }