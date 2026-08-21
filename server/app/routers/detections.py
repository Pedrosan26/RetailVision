"""
detections.py

GET /api/v1/detections: recent detection events, filterable by camera
node, zone, and time range.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models.detection import DetectionEvent
from ..schemas.detection import DetectionOut
from ..utils import as_utc

router = APIRouter(prefix="/api/v1", tags=["detections"])

MAX_LIMIT = 1000


@router.get("/detections", response_model=list[DetectionOut])
async def get_detections(
    limit: int = Query(100, gt=0, le=MAX_LIMIT),
    camera_node_id: str | None = None,
    zone_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[DetectionOut]:
    """Return the most recent detections matching the given filters, newest first."""
    stmt = select(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(limit)
    if camera_node_id:
        stmt = stmt.where(DetectionEvent.camera_node_id == camera_node_id)
    if zone_id:
        stmt = stmt.where(DetectionEvent.zone_id == zone_id)
    if since:
        stmt = stmt.where(DetectionEvent.timestamp >= since)
    if until:
        stmt = stmt.where(DetectionEvent.timestamp <= until)

    result = await db.execute(stmt)
    events = result.scalars().all()
    # Built explicitly (not returned as ORM objects) so as_utc() can
    # normalize the timestamp without mutating session-tracked state.
    return [
        DetectionOut(
            id=event.id,
            camera_node_id=event.camera_node_id,
            timestamp=as_utc(event.timestamp),
            track_id=event.track_id,
            zone_id=event.zone_id,
            world_x=event.world_x,
            world_y=event.world_y,
            count=event.count,
            age_group=event.age_group,
            gender=event.gender,
            emotion=event.emotion,
            dwell_seconds=event.dwell_seconds,
            engagement_score=event.engagement_score,
        )
        for event in events
    ]
