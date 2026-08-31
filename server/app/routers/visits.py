"""
visits.py

GET /api/v1/visits: detection events folded into one row per person --
per (camera node, track), their first and last sighting, how long they
stayed, where, and what their emotion mix was. Since nodes emit one
record per person per change, the raw event stream is machinery; a visit
is the unit anything user-facing should reason about.

Rows without a track_id (from nodes predating per-person emission) are
excluded rather than shown as hundreds of one-event visits, which would
be the old per-frame noise wearing a new name.

Computed in Python from one filtered query, matching the deliberate
portability decision documented in aggregates.py.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..dedup import person_ids_in_thread
from ..models.appearance import TrackAppearance
from ..models.detection import DetectionEvent
from ..schemas.detection import VisitOut
from ..utils import as_utc

router = APIRouter(prefix="/api/v1", tags=["visits"])


async def load_appearances(db, events) -> dict[tuple[str, str], list[float]]:
    """Fetch the stored appearance for every track appearing in these events.

    Scoped to the tracks actually in the range rather than loading the
    whole table: a query over an hour touches a handful of tracks, and the
    table holds a retention window's worth.
    """
    keys = {(e.camera_node_id, e.track_id) for e in events if e.track_id is not None}
    if not keys:
        return {}
    nodes = {camera for camera, _ in keys}
    rows = (
        await db.execute(select(TrackAppearance).where(TrackAppearance.camera_node_id.in_(nodes)))
    ).scalars().all()
    return {
        (row.camera_node_id, row.track_id): row.embedding
        for row in rows
        if (row.camera_node_id, row.track_id) in keys
    }

DEFAULT_LOOKBACK = timedelta(hours=24)
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000


@router.get("/visits", response_model=list[VisitOut])
async def get_visits(
    since: datetime | None = None,
    until: datetime | None = None,
    zone_id: str | None = None,
    camera_node_id: str | None = None,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    db: AsyncSession = Depends(get_db),
) -> list[VisitOut]:
    """Return one row per person's visit in [since, until] (default: last 24h), newest first."""
    range_until = as_utc(until) if until else datetime.now(timezone.utc)
    range_since = as_utc(since) if since else (range_until - DEFAULT_LOOKBACK)

    stmt = select(DetectionEvent).where(
        DetectionEvent.timestamp >= range_since,
        DetectionEvent.timestamp <= range_until,
        DetectionEvent.track_id.is_not(None),
    )
    if zone_id:
        stmt = stmt.where(DetectionEvent.zone_id == zone_id)
    if camera_node_id:
        stmt = stmt.where(DetectionEvent.camera_node_id == camera_node_id)
    events = (await db.execute(stmt)).scalars().all()

    # A visit is a person's stay, not a camera's view of it. Grouping by track
    # produced one row per camera that could see them, so a visitor walking
    # through a room covered by three cameras appeared as three separate
    # visitors with three separate durations, each shorter than the real one.
    person_ids = await person_ids_in_thread(events, appearances=await load_appearances(db, events))

    grouped: dict[str, list[DetectionEvent]] = {}
    for event in events:
        person = person_ids.get((event.camera_node_id, event.track_id), f"{event.camera_node_id}:{event.track_id}")
        grouped.setdefault(person, []).append(event)

    visits = []
    for person, rows in grouped.items():
        node, track = person.split(":", 1)
        rows.sort(key=lambda e: as_utc(e.timestamp))
        emotions = Counter(e.emotion for e in rows)
        # Zone is the mode of the non-null sightings: a person is "in" the
        # zone they spent the visit in, not wherever the last frame put them.
        zones = Counter(e.zone_id for e in rows if e.zone_id is not None)
        first, last = as_utc(rows[0].timestamp), as_utc(rows[-1].timestamp)
        visits.append(
            VisitOut(
                camera_node_id=node,
                track_id=track,
                first_seen=first,
                last_seen=last,
                duration_seconds=(last - first).total_seconds(),
                zone_id=zones.most_common(1)[0][0] if zones else None,
                age_group=rows[0].age_group,
                gender=rows[0].gender,
                dominant_emotion=emotions.most_common(1)[0][0],
                emotion_distribution=dict(emotions),
                events=len(rows),
            )
        )

    visits.sort(key=lambda v: v.last_seen, reverse=True)
    return visits[:limit]
