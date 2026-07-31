"""
output_log.py

Privacy layer for the inference pipeline: converts each detection produced
by InferencePipeline.process_frame() into an anonymized log record --
demographic and emotion labels only, never the pixel data (frame or face
crop) they were derived from -- and appends it to the on-disk inference
log. Records are newline-delimited JSON (one JSON object per line) rather
than a single JSON array, since the array form would require rewriting the
entire file on every append.

zone_id, count, dwell_seconds, and engagement_score are part of the
schema but not yet computable: they depend on the zone configuration and
people-counting/zone-emotion modules (a later epic), so they're logged as
null placeholders for now. The schema is frozen at these 8 fields so that
epic can populate real values without changing the log's shape.
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


def build_log_record(detection: dict, timestamp: str | None = None) -> dict:
    """Build one schema-conforming, anonymized log record from a process_frame() detection."""
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "zone_id": None,
        "count": None,
        "age_group": detection["age_group"],
        "gender": detection["gender"],
        "emotion": detection["emotion"],
        "dwell_seconds": None,
        "engagement_score": None,
    }


def log_detection(detection: dict, log_path: Path = DEFAULT_LOG_PATH, timestamp: str | None = None) -> None:
    """Append one anonymized log record for a single detection event to log_path."""
    record = build_log_record(detection, timestamp=timestamp)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
