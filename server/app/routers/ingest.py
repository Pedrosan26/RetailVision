"""
ingest.py

POST /api/v1/ingest: receives a batch of anonymized detection records
from one camera node and persists them.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_camera_node_api_key
from ..db import get_db
from ..models.appearance import TrackAppearance
from ..models.detection import DetectionEvent
from ..schemas.detection import AppearanceIn, IngestRequest, IngestResponse

# How long an appearance vector is kept. It describes clothing rather than
# a face, so it stops being useful within a day or so anyway; the window is
# what makes that a stated policy rather than an accident of how often the
# table happens to be cleared. Shorten it freely -- nothing downstream needs
# a vector older than the visits being analysed.
APPEARANCE_RETENTION = timedelta(days=30)

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post(
    "/ingest",
    status_code=202,
    response_model=IngestResponse,
    dependencies=[Depends(verify_camera_node_api_key)],
)
async def ingest(body: IngestRequest, db: AsyncSession = Depends(get_db)) -> IngestResponse:
    """Bulk-insert a batch of detection records, attributing them all to body.camera_node_id."""
    rows = [DetectionEvent(camera_node_id=body.camera_node_id, **record.model_dump()) for record in body.records]
    db.add_all(rows)

    if body.appearances:
        await _store_appearances(db, body.camera_node_id, body.appearances)

    await db.commit()
    return IngestResponse(accepted=len(rows))


async def _store_appearances(db: AsyncSession, camera_node_id: str, appearances: list[AppearanceIn]) -> None:
    """Upsert each track's current appearance, and drop any that have outlived the retention window.

    Upsert rather than insert because a node re-sends a track's description
    as it refines it; keeping every version would store thousands of nearly
    identical vectors per visitor and would quietly extend how long the
    earliest one survives.

    Pruning happens here rather than on a schedule so retention needs no
    process outside the server to be running for it to hold. The cost is one
    small delete per batch, against an indexed column.
    """
    now = datetime.now(timezone.utc)
    for appearance in appearances:
        existing = await db.get(TrackAppearance, (camera_node_id, appearance.track_id))
        if existing is None:
            db.add(
                TrackAppearance(
                    camera_node_id=camera_node_id,
                    track_id=appearance.track_id,
                    embedding=appearance.embedding,
                    updated_at=now,
                )
            )
        else:
            existing.embedding = appearance.embedding
            existing.updated_at = now

    await db.execute(delete(TrackAppearance).where(TrackAppearance.updated_at < now - APPEARANCE_RETENTION))
