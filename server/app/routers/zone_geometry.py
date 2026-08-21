"""
zone_geometry.py

Zone floor polygons, uploaded by camera nodes and read by the dashboard.

The polygon is computed on the nodes from the surveyed marker map, which
the server never sees -- so a node uploads its ready zones' geometry once
at startup (POST, authenticated like ingest), and the dashboard reads it
back (GET) to draw the floor map that world positions land on. Last
writer wins per zone: every node loads the same surveyed map file, so
disagreement between them would mean a stale node, not a conflict worth
merging.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_zone_geometry_api_key
from ..db import get_db
from ..models.zone import ZoneGeometry
from ..schemas.detection import ZoneGeometryOut, ZoneGeometryRequest

router = APIRouter(prefix="/api/v1", tags=["zones"])


@router.post("/zones/geometry", status_code=204, dependencies=[Depends(verify_zone_geometry_api_key)])
async def upload_zone_geometry(body: ZoneGeometryRequest, db: AsyncSession = Depends(get_db)) -> None:
    """Upsert each uploaded zone's polygon, attributed to the uploading node."""
    for zone in body.zones:
        existing = await db.get(ZoneGeometry, zone.zone_id)
        if existing is None:
            db.add(ZoneGeometry(zone_id=zone.zone_id, camera_node_id=body.camera_node_id, polygon=zone.polygon))
        else:
            existing.camera_node_id = body.camera_node_id
            existing.polygon = zone.polygon
    await db.commit()


@router.get("/zones/geometry", response_model=list[ZoneGeometryOut])
async def get_zone_geometry(db: AsyncSession = Depends(get_db)) -> list[ZoneGeometry]:
    """Return every zone's stored floor polygon."""
    result = await db.execute(select(ZoneGeometry).order_by(ZoneGeometry.zone_id))
    return list(result.scalars().all())
