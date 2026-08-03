"""
detection.py

Pydantic request/response models for the ingestion endpoint.
DetectionRecord mirrors the frozen 8-field log schema exactly -- see
docs/schema.md in the repo root.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class DetectionRecord(BaseModel):
    """Validates one anonymized detection record against the frozen 8-field schema."""

    timestamp: datetime
    zone_id: str | None = None
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
