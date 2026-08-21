"""
summary.py

GET /api/v1/summary: the headline figures for one time range in one
response -- total events, distinct people, average dwell, the emotion
mix, and the busiest hour. The dashboard's KPI row needs exactly these
five numbers; deriving them client-side from /aggregates would mean
summing per-bucket unique-people counts, which overcounts anyone present
across a bucket boundary. Only the server can deduplicate a person
across the whole range, because only it sees every row at once.

Computed in Python from one filtered query, matching the deliberate
portability decision documented in aggregates.py.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..models.detection import DetectionEvent
from ..schemas.detection import SummaryOut
from ..utils import as_utc

router = APIRouter(prefix="/api/v1", tags=["summary"])

DEFAULT_LOOKBACK = timedelta(hours=24)
HOUR = timedelta(hours=1)


def _person_key(event: DetectionEvent) -> tuple[str, str]:
    """One person is one track within one camera node; pre-track rows count as one person each."""
    return (event.camera_node_id, event.track_id or f"row-{event.id}")


@router.get("/summary", response_model=SummaryOut)
async def get_summary(
    since: datetime | None = None,
    until: datetime | None = None,
    zone_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> SummaryOut:
    """Return headline figures over detection events in [since, until] (default: last 24h)."""
    range_until = as_utc(until) if until else datetime.now(timezone.utc)
    range_since = as_utc(since) if since else (range_until - DEFAULT_LOOKBACK)

    stmt = select(DetectionEvent).where(
        DetectionEvent.timestamp >= range_since, DetectionEvent.timestamp <= range_until
    )
    if zone_id:
        stmt = stmt.where(DetectionEvent.zone_id == zone_id)
    events = (await db.execute(stmt)).scalars().all()

    dwell_values = [e.dwell_seconds for e in events if e.dwell_seconds is not None]

    # Busiest hour: distinct people per hour-aligned bucket, peak wins. A person
    # spanning two hours counts in both, which is what "how busy was that hour"
    # means -- they were there during it.
    per_hour: dict[int, set[tuple[str, str]]] = {}
    for event in events:
        index = (as_utc(event.timestamp) - range_since) // HOUR
        per_hour.setdefault(index, set()).add(_person_key(event))
    busiest_index = max(per_hour, key=lambda i: len(per_hour[i]), default=None)

    return SummaryOut(
        since=range_since,
        until=range_until,
        total_detections=len(events),
        unique_people=len({_person_key(e) for e in events}),
        avg_dwell_seconds=sum(dwell_values) / len(dwell_values) if dwell_values else None,
        emotion_distribution=dict(Counter(e.emotion for e in events)),
        busiest_hour_start=None if busiest_index is None else range_since + busiest_index * HOUR,
        busiest_hour_people=0 if busiest_index is None else len(per_hour[busiest_index]),
        # The node's own deduplicated headcount at each record's moment, so its
        # maximum is a real "most people at once", not a sum across cameras.
        peak_occupancy=max((e.count for e in events if e.count is not None), default=0),
    )
