"""
detection.py

Pydantic request/response models for the ingestion and read endpoints.
DetectionRecord mirrors the log schema exactly -- see docs/schema.md in
the repo root.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DetectionRecord(BaseModel):
    """Validates one anonymized detection record against the log schema."""

    timestamp: datetime
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


class AggregateBucket(BaseModel):
    """One time-windowed rollup: counts, demographic/emotion distribution, and averages."""

    bucket_start: datetime
    detection_count: int
    age_group_distribution: dict[str, int]
    gender_distribution: dict[str, int]
    emotion_distribution: dict[str, int]
    avg_dwell_seconds: float | None
    avg_engagement_score: float | None
