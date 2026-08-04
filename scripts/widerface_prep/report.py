"""
report.py

Builds the distribution report documenting box/image counts per split,
how much annotation data was dropped by filtering (and why), and the
face-count-per-image distribution -- so the crowd-scene-vs-retail domain
gap this dataset is known for is documented up front rather than
discovered as a surprise during training/evaluation.
"""

from collections import defaultdict

from .splitting import bucket_face_count


def _face_count_distribution(records: list[dict]) -> dict[str, int]:
    """Count how many records fall into each face-count bucket."""
    counter: dict[str, int] = defaultdict(int)
    for record in records:
        counter[bucket_face_count(len(record["boxes"]))] += 1
    return dict(sorted(counter.items()))


def _split_summary(records: list[dict]) -> dict:
    """Summarize one split's image/box counts and face-count distribution."""
    box_count = sum(len(r["boxes"]) for r in records)
    background_images = sum(1 for r in records if len(r["boxes"]) == 0)
    return {
        "images": len(records),
        "boxes": box_count,
        "background_images": background_images,
        "face_count_distribution": _face_count_distribution(records),
    }


def build_report(splits: dict[str, list[dict]], parse_stats: dict) -> dict:
    """Compute the full distribution report across all splits."""
    return {
        "raw_annotation_stats": parse_stats,
        "by_split": {name: _split_summary(records) for name, records in splits.items()},
    }