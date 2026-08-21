"""
zone.py

SQLAlchemy ORM model for zone geometry: each zone's floor polygon in the
shared world frame, as surveyed by the camera nodes. The polygon lives
in the nodes' marker-map config, which the server otherwise never sees;
nodes upload it at startup so the dashboard can draw a floor map and
place world positions on it. One row per zone, last writer wins --
every node loads the same surveyed map, so they agree.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .detection import Base


class ZoneGeometry(Base):
    """One zone's floor polygon in world meters, as most recently uploaded by a camera node."""

    __tablename__ = "zone_geometry"

    zone_id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_node_id: Mapped[str] = mapped_column(String, nullable=False)
    # A ring of [x, y] world coordinates in meters. JSON rather than a child
    # table: the polygon is only ever read whole, never queried by vertex.
    polygon: Mapped[list] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
