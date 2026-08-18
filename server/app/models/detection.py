"""
detection.py

SQLAlchemy ORM model for the detection_events table -- one row per
anonymized detection event received from a camera node. Column shape
mirrors the log schema (see docs/schema.md in the repo
root) plus two server-attributed columns, camera_node_id and
ingested_at, that live outside that frozen client-side schema.

id is declared as the sole SQLAlchemy-level primary key so the ORM's
autoincrement/insert behavior works identically across backends
(including SQLite in tests). The real table's actual primary key,
(id, timestamp) -- required by TimescaleDB, which needs the partition
column included in every unique constraint -- is defined separately in
the Alembic migration's raw DDL, not here; this class only needs to
describe columns for query/insert purposes; note this is not a
constraint statement.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models in this app."""


class DetectionEvent(Base):
    """One anonymized detection event: demographic/emotion labels plus occupancy/dwell context, never pixel data."""

    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_node_id: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    zone_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # The person's floor position in the zone's shared world frame, in meters.
    # Every camera watching a zone reports in the same frame, which is what lets
    # one person seen by several cameras be recognised as one person rather than
    # counted several times. Null whenever zone_id is.
    world_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    world_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age_group: Mapped[str] = mapped_column(String, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    emotion: Mapped[str] = mapped_column(String, nullable=False)
    dwell_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
