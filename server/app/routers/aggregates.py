"""
aggregates.py

GET /api/v1/aggregates: time-windowed rollups of detection events --
counts, demographic/emotion distribution, and average dwell/engagement
per bucket. Bucketing happens in Python after one portable filtered
query, not via TimescaleDB's time_bucket() SQL function -- at the data
volumes implied by a handful of camera nodes, that's simple and fast
enough that raw dialect-specific SQL isn't worth the portability cost,
and it keeps this endpoint's logic testable against the in-memory
SQLite DB used in tests as well as real TimescaleDB in production.
Promote to a TimescaleDB continuous aggregate once real zone/engagement
data validates the query shape this endpoint should actually serve.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..dedup import person_ids_for_events
from ..models.appearance import TrackAppearance
from ..models.detection import DetectionEvent
from ..schemas.detection import AggregateBucket
from ..utils import as_utc

router = APIRouter(prefix="/api/v1", tags=["aggregates"])


async def load_appearances(db, events) -> dict[tuple[str, str], list[float]]:
    """Fetch the stored appearance for every track appearing in these events."""
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
WINDOW_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def parse_window(window: str) -> timedelta:
    """Parse a window string like '5m', '1h', or '24h' into a timedelta, raising ValueError if malformed."""
    unit = window[-1] if window else ""
    if unit not in WINDOW_UNITS:
        raise ValueError(f"Unsupported window unit {unit!r}; expected one of {sorted(WINDOW_UNITS)}")
    try:
        value = int(window[:-1])
    except ValueError:
        raise ValueError(f"Invalid window value: {window!r}") from None
    return timedelta(**{WINDOW_UNITS[unit]: value})


def bucket_events(
    events: list[DetectionEvent],
    since: datetime,
    window: timedelta,
    appearances: dict[tuple[str, str], list[float]] | None = None,
) -> list[AggregateBucket]:
    """Group events into fixed-size time buckets aligned to `since`, aggregating each bucket."""
    buckets: dict[int, list[DetectionEvent]] = {}
    for event in events:
        index = (as_utc(event.timestamp) - since) // window
        buckets.setdefault(index, []).append(event)

    result = []
    # One clustering pass over every event in range, shared by all buckets.
    person_ids = person_ids_for_events(events, appearances=appearances)

    for index in sorted(buckets):
        bucket = buckets[index]
        dwell_values = [e.dwell_seconds for e in bucket if e.dwell_seconds is not None]
        engagement_values = [e.engagement_score for e in bucket if e.engagement_score is not None]
        # A person is not a track: cameras watching one area each report the
        # same visitor separately, so counting tracks counted them once per
        # camera. Clustering is done once over the whole range rather than per
        # bucket, so someone spanning two buckets stays one person in both
        # instead of being re-derived either side of the boundary.
        people = {
            person_ids.get((e.camera_node_id, e.track_id), f"{e.camera_node_id}:row-{e.id}")
            if e.track_id
            else f"{e.camera_node_id}:row-{e.id}"
            for e in bucket
        }
        result.append(
            AggregateBucket(
                bucket_start=since + index * window,
                detection_count=len(bucket),
                unique_people=len(people),
                age_group_distribution=dict(Counter(e.age_group for e in bucket)),
                gender_distribution=dict(Counter(e.gender for e in bucket)),
                emotion_distribution=dict(Counter(e.emotion for e in bucket)),
                avg_dwell_seconds=sum(dwell_values) / len(dwell_values) if dwell_values else None,
                avg_engagement_score=sum(engagement_values) / len(engagement_values) if engagement_values else None,
            )
        )
    return result


@router.get("/aggregates", response_model=list[AggregateBucket])
async def get_aggregates(
    window: str = Query("5m"),
    since: datetime | None = None,
    until: datetime | None = None,
    zone_id: str | None = None,
    # Repeatable, so several values of one dimension read as "any of these"
    # (?emotion=happy&emotion=neutral) while different dimensions read as
    # "and". Omitting a dimension entirely means it is not filtered on, which
    # is what clearing a filter in the dashboard sends.
    age_group: list[str] | None = Query(None),
    gender: list[str] | None = Query(None),
    emotion: list[str] | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> list[AggregateBucket]:
    """Return time-windowed aggregates over detection events in [since, until] (default: last 24h)."""
    try:
        window_delta = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    range_until = as_utc(until) if until else datetime.now(timezone.utc)
    range_since = as_utc(since) if since else (range_until - DEFAULT_LOOKBACK)

    stmt = (
        select(DetectionEvent)
        .where(DetectionEvent.timestamp >= range_since, DetectionEvent.timestamp <= range_until)
        .order_by(DetectionEvent.timestamp)
    )
    if zone_id:
        stmt = stmt.where(DetectionEvent.zone_id == zone_id)
    for column, values in (
        (DetectionEvent.age_group, age_group),
        (DetectionEvent.gender, gender),
        (DetectionEvent.emotion, emotion),
    ):
        if values:
            stmt = stmt.where(column.in_(values))

    result = await db.execute(stmt)
    events = result.scalars().all()
    return bucket_events(events, range_since, window_delta, await load_appearances(db, events))
