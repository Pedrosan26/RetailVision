"""
ingest.py

POST /api/v1/ingest: receives a batch of anonymized detection records
from one camera node and persists them.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
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

    A real upsert rather than "look, then insert or update". Nodes ship
    batches concurrently and re-send a track's description as they refine
    it, so two batches for one track would both find nothing and both
    insert -- which is precisely what happened, failing the whole ingest
    with a duplicate key and taking the detection records down with it.
    Letting the database resolve the conflict is the only version of this
    without a race in it.

    Pruning happens here rather than on a schedule so retention needs no
    process outside the server to be running for it to hold. The cost is
    one small delete per batch, against an indexed column.
    """
    now = datetime.now(timezone.utc)
    rows = [
        {
            "camera_node_id": camera_node_id,
            "track_id": appearance.track_id,
            "embedding": appearance.embedding,
            "updated_at": now,
        }
        for appearance in appearances
    ]

    # Both backends spell ON CONFLICT the same way but expose it through
    # their own insert construct, so the dialect picks which one to build.
    insert = pg_insert if db.bind.dialect.name == "postgresql" else sqlite_insert
    statement = insert(TrackAppearance).values(rows)
    await db.execute(
        statement.on_conflict_do_update(
            index_elements=["camera_node_id", "track_id"],
            set_={"embedding": statement.excluded.embedding, "updated_at": statement.excluded.updated_at},
        )
    )

    await db.execute(delete(TrackAppearance).where(TrackAppearance.updated_at < now - APPEARANCE_RETENTION))
