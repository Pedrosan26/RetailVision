"""
ingest.py

POST /api/v1/ingest: receives a batch of anonymized detection records
from one camera node and persists them.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import verify_camera_node_api_key
from ..db import get_db
from ..models.detection import DetectionEvent
from ..schemas.detection import IngestRequest, IngestResponse

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
    await db.commit()
    return IngestResponse(accepted=len(rows))
