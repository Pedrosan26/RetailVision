"""
output_log.py

Privacy layer for the inference pipeline: converts each detection produced
by InferencePipeline.process_frame() into an anonymized log record --
demographic and emotion labels only, never the pixel data (frame or face
crop) they were derived from -- and appends it to the on-disk inference
log. Records are newline-delimited JSON (one JSON object per line) rather
than a single JSON array, since the array form would require rewriting the
entire file on every append.

zone_id and engagement_score are not yet computable -- they depend on the
zone configuration and zone-emotion correlation modules -- so they're
still logged as null placeholders. count and dwell_seconds are populated
once a LineCounter is supplied: count is net occupancy at the moment of
the detection event, dwell_seconds is how long the detected track has
currently been present. The schema is frozen at these 8 fields so later
modules can populate the remaining ones without changing the log's shape.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "inference_log.json"

SCHEMA_FIELDS = (
    "timestamp",
    "zone_id",
    "count",
    "age_group",
    "gender",
    "emotion",
    "dwell_seconds",
    "engagement_score",
)


def build_log_record(
    detection: dict,
    timestamp: str | None = None,
    count: int | None = None,
    dwell_seconds: float | None = None,
) -> dict:
    """Build one schema-conforming, anonymized log record from a process_frame() detection."""
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "zone_id": None,
        "count": count,
        "age_group": detection["age_group"],
        "gender": detection["gender"],
        "emotion": detection["emotion"],
        "dwell_seconds": dwell_seconds,
        "engagement_score": None,
    }


def log_detection(
    detection: dict,
    log_path: Path = DEFAULT_LOG_PATH,
    timestamp: str | None = None,
    count: int | None = None,
    dwell_seconds: float | None = None,
) -> None:
    """Append one anonymized log record for a single detection event to log_path."""
    record = build_log_record(detection, timestamp=timestamp, count=count, dwell_seconds=dwell_seconds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
