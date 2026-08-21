"""
detection.py

Pydantic request/response models for the ingestion and read endpoints.
DetectionRecord mirrors the log schema exactly -- see output_log.py in
the repo root.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DetectionRecord(BaseModel):
    """Validates one anonymized detection record against the log schema."""

    timestamp: datetime
    # Groups one person's records together. Optional so a node that has not
    # been upgraded keeps ingesting unchanged.
    track_id: str | None = None
    zone_id: str | None = None
    # Optional so that camera nodes running without zones, and nodes not yet
    # upgraded, keep validating against this model unchanged.
    world_x: float | None = None
    world_y: float | None = None
    count: int | None = None
    age_group: str
    gender: str
    emotion: str
    dwell_seconds: float | None = None
    engagement_score: float | None = None


class IngestRequest(BaseModel):
    """A batch of records shipped from one camera node."""

    camera_node_id: str
    records: list[DetectionRecord] = Field(min_length=1, max_length=500)


class IngestResponse(BaseModel):
    """Acknowledges how many records were persisted."""

    accepted: int


class DetectionOut(BaseModel):
    """One persisted detection event, as returned by the read endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_node_id: str
    timestamp: datetime
    track_id: str | None
    zone_id: str | None
    world_x: float | None
    world_y: float | None
    count: int | None
    age_group: str
    gender: str
    emotion: str
    dwell_seconds: float | None
    engagement_score: float | None


class OccupancyOut(BaseModel):
    """Latest known occupancy reported by one camera node, for one zone."""

    key: str
    camera_node_id: str
    zone_id: str | None
    count: int | None
    timestamp: datetime


class ZoneOccupancyOut(BaseModel):
    """A zone's headcount with people seen by several cameras counted once."""

    zone_id: str
    total: int
    per_camera: dict[str, int]
    cameras_reporting: int
    timestamp: datetime


class SummaryOut(BaseModel):
    """Headline figures over one time range, for the dashboard's KPI row.

    unique_people counts distinct (camera node, track) pairs, so it is
    per-camera: someone visible to two cameras counts twice here, the same
    honesty caveat the aggregates carry. peak_occupancy is the largest
    headcount any record carried -- the node-reported count at its moment
    -- so it is deduplicated the way the node deduplicated it, not summed
    across cameras.
    """

    since: datetime
    until: datetime
    total_detections: int
    unique_people: int
    avg_dwell_seconds: float | None
    emotion_distribution: dict[str, int]
    busiest_hour_start: datetime | None
    busiest_hour_people: int
    peak_occupancy: int


class VisitOut(BaseModel):
    """One person's visit as seen by one camera: their track's records folded into a single row.

    The unit the dashboard reasons about -- a person who stayed four
    minutes is one visit, not two hundred events. duration_seconds spans
    first to last record; emission is change-driven with a 10s heartbeat,
    so it understates a stay by at most that heartbeat.
    """

    camera_node_id: str
    track_id: str
    first_seen: datetime
    last_seen: datetime
    duration_seconds: float
    zone_id: str | None
    age_group: str
    gender: str
    dominant_emotion: str
    emotion_distribution: dict[str, int]
    events: int


class ZoneGeometryIn(BaseModel):
    """One zone's floor polygon in world meters, as a ring of [x, y] pairs."""

    zone_id: str
    polygon: list[list[float]] = Field(min_length=3)


class ZoneGeometryRequest(BaseModel):
    """A camera node's zone geometry upload, sent once at startup."""

    camera_node_id: str
    zones: list[ZoneGeometryIn] = Field(min_length=1)


class ZoneGeometryOut(BaseModel):
    """A zone's stored floor polygon, for the dashboard's floor map."""

    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    camera_node_id: str
    polygon: list[list[float]]
    updated_at: datetime


class AggregateBucket(BaseModel):
    """One time-windowed rollup: counts, demographic/emotion distribution, and averages."""

    bucket_start: datetime
    detection_count: int
    # Distinct people behind those events. A person present across a bucket
    # contributes several events but one person, so the two diverge and the
    # difference is meaningful rather than noise.
    unique_people: int
    age_group_distribution: dict[str, int]
    gender_distribution: dict[str, int]
    emotion_distribution: dict[str, int]
    avg_dwell_seconds: float | None
    avg_engagement_score: float | None
