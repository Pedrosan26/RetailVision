"""
occupancy.py

Live occupancy, at two levels.

GET /api/v1/occupancy/live reports what each camera node last said, one
row per camera and zone. It is deliberately not aggregated: a camera whose
count looks wrong is only visible if its own figure is.

GET /api/v1/occupancy/zones reports each zone's actual headcount, with a
person seen by several cameras counted once. Cameras covering one area see
each other's subjects, so summing their counts is not a headcount -- with
three cameras on one room it can be several times the truth. Merging is
possible because every camera watching a zone reports positions in the same
world frame; see app/dedup.py for how, and what it deliberately does not
merge.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..dedup import DEFAULT_MERGE_RADIUS_METERS, Observation, deduplicated_headcount
from ..models.detection import DetectionEvent
from ..schemas.detection import OccupancyOut, ZoneOccupancyOut
from ..utils import as_utc

router = APIRouter(prefix="/api/v1", tags=["occupancy"])

# How far back a zone's "now" reaches. Long enough to cover every camera's most
# recent frame including network jitter, short enough that a person who has left
# is not still counted. dedup then narrows this to one frame per camera.
DEFAULT_LIVE_WINDOW_SECONDS = 5.0


@router.get("/occupancy/live", response_model=list[OccupancyOut])
async def get_live_occupancy(db: AsyncSession = Depends(get_db)) -> list[OccupancyOut]:
    """Return each camera node's most recent count, one row per camera and zone."""
    # Grouped by camera and zone rather than by zone alone: several cameras
    # watching one zone each have their own answer, and collapsing them here
    # would silently report whichever happened to arrive last.
    group_key = func.coalesce(DetectionEvent.zone_id, DetectionEvent.camera_node_id)
    ranked = select(
        DetectionEvent.camera_node_id,
        DetectionEvent.zone_id,
        DetectionEvent.count,
        DetectionEvent.timestamp,
        group_key.label("group_key"),
        func.row_number()
        .over(
            partition_by=(DetectionEvent.camera_node_id, group_key),
            order_by=DetectionEvent.timestamp.desc(),
        )
        .label("rn"),
    ).subquery()

    result = await db.execute(select(ranked).where(ranked.c.rn == 1))
    return [
        OccupancyOut(
            key=row.group_key,
            camera_node_id=row.camera_node_id,
            zone_id=row.zone_id,
            count=row.count,
            timestamp=as_utc(row.timestamp),
        )
        for row in result.all()
    ]


@router.get("/occupancy/zones", response_model=list[ZoneOccupancyOut])
async def get_zone_occupancy(
    window_seconds: float = Query(DEFAULT_LIVE_WINDOW_SECONDS, gt=0, le=120),
    merge_radius: float = Query(DEFAULT_MERGE_RADIUS_METERS, gt=0, le=10),
    db: AsyncSession = Depends(get_db),
) -> list[ZoneOccupancyOut]:
    """Return each zone's headcount with people seen by several cameras counted once."""
    since = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    stmt = (
        select(
            DetectionEvent.zone_id,
            DetectionEvent.camera_node_id,
            DetectionEvent.timestamp,
            DetectionEvent.world_x,
            DetectionEvent.world_y,
        )
        .where(
            DetectionEvent.timestamp >= since,
            DetectionEvent.zone_id.is_not(None),
            DetectionEvent.world_x.is_not(None),
            DetectionEvent.world_y.is_not(None),
        )
        .order_by(DetectionEvent.timestamp)
    )
    result = await db.execute(stmt)

    by_zone: dict[str, list[Observation]] = {}
    latest_seen: dict[str, datetime] = {}
    for row in result.all():
        timestamp = as_utc(row.timestamp)
        by_zone.setdefault(row.zone_id, []).append(
            Observation(
                camera_node_id=row.camera_node_id,
                timestamp=timestamp,
                world_x=row.world_x,
                world_y=row.world_y,
            )
        )
        latest_seen[row.zone_id] = max(latest_seen.get(row.zone_id, timestamp), timestamp)

    zones = []
    for zone_id, observations in sorted(by_zone.items()):
        total, per_camera = deduplicated_headcount(observations, merge_radius=merge_radius)
        zones.append(
            ZoneOccupancyOut(
                zone_id=zone_id,
                total=total,
                per_camera=per_camera,
                cameras_reporting=len(per_camera),
                timestamp=latest_seen[zone_id],
            )
        )
    return zones
