"""
occupancy.py

GET /api/v1/occupancy/live: the latest known occupancy count per zone,
falling back to grouping by camera node until zone configuration exists.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models.detection import DetectionEvent
from ..schemas.detection import OccupancyOut
from ..utils import as_utc

router = APIRouter(prefix="/api/v1", tags=["occupancy"])


@router.get("/occupancy/live", response_model=list[OccupancyOut])
async def get_live_occupancy(db: AsyncSession = Depends(get_db)) -> list[OccupancyOut]:
    """Return the most recent detection's count for each zone (or camera node, before zones exist)."""
    # COALESCE(zone_id, camera_node_id): once zone configuration starts
    # populating zone_id, this transparently starts grouping by real
    # zones -- no query change needed at that point.
    group_key = func.coalesce(DetectionEvent.zone_id, DetectionEvent.camera_node_id)
    ranked = (
        select(
            DetectionEvent.camera_node_id,
            DetectionEvent.zone_id,
            DetectionEvent.count,
            DetectionEvent.timestamp,
            group_key.label("group_key"),
            func.row_number().over(partition_by=group_key, order_by=DetectionEvent.timestamp.desc()).label("rn"),
        )
    ).subquery()

    stmt = select(ranked).where(ranked.c.rn == 1)
    result = await db.execute(stmt)
    rows = result.all()

    return [
        OccupancyOut(
            key=row.group_key,
            camera_node_id=row.camera_node_id,
            zone_id=row.zone_id,
            count=row.count,
            timestamp=as_utc(row.timestamp),
        )
        for row in rows
    ]
