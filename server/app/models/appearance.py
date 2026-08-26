"""
appearance.py

What each track looked like, so one person can be recognised across
cameras that never saw them at the same moment.

One row per (camera node, track), overwritten as the node refines its
running average, rather than one per detection event: the description
belongs to the track, and storing it per row would repeat the same vector
thousands of times for one visitor.

This is the only table in the system holding data derived from how
someone looks, and it is the reason retention is a column concern rather
than an afterthought. An appearance vector is biometric data under GDPR
Article 9 and equivalent laws, so rows carry the time they were last
updated and are pruned past a fixed window -- see APPEARANCE_RETENTION in
the ingest router. Nothing here identifies a person by name, and the
vector describes clothing and build rather than a face, so it stops being
useful the moment someone changes their jacket; that bounds both its
value and its sensitivity, but it does not take it outside those rules.

The vector is stored as JSON rather than a Postgres array or a pgvector
column. Comparison happens in Python during clustering, over the handful
of tracks in a query's time range, so there is nothing to gain from
in-database distance operators yet -- and JSON is the one representation
that behaves identically on the SQLite the tests run against. Moving to
pgvector is the right change once the number of tracks makes a Python
loop the bottleneck, and it is a migration rather than a redesign.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .detection import Base


class TrackAppearance(Base):
    """One track's appearance vector, as last reported by the camera node that saw it."""

    __tablename__ = "track_appearances"

    # A track is identified by the node that assigned it plus the id itself;
    # neither is unique on its own, since nodes number their tracks separately.
    camera_node_id: Mapped[str] = mapped_column(String, primary_key=True)
    track_id: Mapped[str] = mapped_column(String, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    # Both when this description was last refined and what the retention
    # window is measured from.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
