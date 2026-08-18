"""
output_log.py

Privacy layer for the inference pipeline: converts each detection produced
by InferencePipeline.process_frame() into an anonymized log record --
demographic and emotion labels only, never the pixel data (frame or face
crop) they were derived from -- and appends it to the on-disk inference
log. Records are newline-delimited JSON (one JSON object per line) rather
than a single JSON array, since the array form would require rewriting the
entire file on every append.

zone_id is populated when the pipeline is running with marker-based zones
(see zones.ZoneResolver); it stays null when zones are not configured, and
also when they are but the camera cannot currently see a mapped marker, so
"unknown" and "outside every zone" are not conflated. world_x and world_y are that person's position on the zone's floor, in
meters, in the shared world frame every camera watching a zone agrees on.
They are what lets the server recognise that one person seen by two
cameras is one person rather than two, so they are anonymized position
data about a location, not about an identity, and are null whenever
zone_id is. engagement_score is still a null placeholder pending the
zone-emotion correlation work. count
and dwell_seconds are populated once a counter is supplied: count is
occupancy at the moment of the detection event, dwell_seconds is how long
the detected track has currently been present. The schema is frozen at
these 8 fields so later modules can populate the remaining ones without
changing the log's shape.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "inference_log.json"

SCHEMA_FIELDS = (
    "timestamp",
    "zone_id",
    "world_x",
    "world_y",
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
    zone_id: str | None = None,
    world_position: tuple[float, float] | None = None,
) -> dict:
    """Build one schema-conforming, anonymized log record from a process_frame() detection."""
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "zone_id": zone_id,
        "world_x": None if world_position is None else float(world_position[0]),
        "world_y": None if world_position is None else float(world_position[1]),
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
    zone_id: str | None = None,
    world_position: tuple[float, float] | None = None,
) -> None:
    """Append one anonymized log record for a single detection event to log_path."""
    record = build_log_record(
        detection,
        timestamp=timestamp,
        count=count,
        dwell_seconds=dwell_seconds,
        zone_id=zone_id,
        world_position=world_position,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
