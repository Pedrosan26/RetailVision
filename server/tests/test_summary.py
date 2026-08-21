"""
test_summary.py

Endpoint tests for GET /api/v1/summary: headline totals, distinct-people
counting across the whole range, dwell averaging, the busiest hour, and
the zone filter.
"""

from datetime import datetime, timedelta, timezone

from app.models.detection import DetectionEvent

BASE_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _event(
    offset_minutes: int,
    track_id: str | None = None,
    camera_node_id: str = "node-a",
    zone_id: str | None = None,
    emotion: str = "neutral",
    dwell_seconds: float | None = None,
) -> DetectionEvent:
    """Build a DetectionEvent offset_minutes after BASE_TIME."""
    return DetectionEvent(
        camera_node_id=camera_node_id,
        timestamp=BASE_TIME + timedelta(minutes=offset_minutes),
        track_id=track_id,
        zone_id=zone_id,
        count=None,
        age_group="18-40",
        gender="Female",
        emotion=emotion,
        dwell_seconds=dwell_seconds,
        engagement_score=None,
    )


async def _seed(db_session_factory, events: list[DetectionEvent]) -> None:
    """Insert the given events directly into the test DB."""
    async with db_session_factory() as session:
        session.add_all(events)
        await session.commit()


async def _summary(client, **params) -> dict:
    """Query the endpoint over the seeded three-hour range with optional extra filters."""
    response = await client.get(
        "/api/v1/summary",
        params={
            "since": BASE_TIME.isoformat(),
            "until": (BASE_TIME + timedelta(hours=3)).isoformat(),
            **params,
        },
    )
    assert response.status_code == 200
    return response.json()


async def test_one_person_many_events_counts_once_across_the_range(client, db_session_factory):
    """A track spanning several hours is one person in the summary, unlike summed per-bucket counts."""
    await _seed(db_session_factory, [_event(0, "t1"), _event(70, "t1"), _event(140, "t1")])
    body = await _summary(client)
    assert body["total_detections"] == 3
    assert body["unique_people"] == 1


async def test_busiest_hour_is_the_one_with_most_distinct_people(client, db_session_factory):
    """The peak hour is picked by people present, not by raw event count."""
    await _seed(
        db_session_factory,
        [
            # Hour 0: one person producing many events.
            _event(1, "t1"), _event(2, "t1"), _event(3, "t1"), _event(4, "t1"),
            # Hour 1: two people, one event each.
            _event(61, "t2"), _event(62, "t3"),
        ],
    )
    body = await _summary(client)
    assert body["busiest_hour_people"] == 2
    assert body["busiest_hour_start"].startswith("2026-08-03T13:00")


async def test_dwell_average_and_emotions_cover_the_range(client, db_session_factory):
    """Dwell averages the non-null values and the emotion mix counts every event."""
    await _seed(
        db_session_factory,
        [_event(0, "t1", dwell_seconds=10.0), _event(5, "t2", dwell_seconds=30.0), _event(6, "t3", emotion="happy")],
    )
    body = await _summary(client)
    assert body["avg_dwell_seconds"] == 20.0
    assert body["emotion_distribution"] == {"neutral": 2, "happy": 1}


async def test_zone_filter_narrows_every_figure(client, db_session_factory):
    """Scoping to one zone excludes other zones' events from all totals."""
    await _seed(db_session_factory, [_event(0, "t1", zone_id="a"), _event(1, "t2", zone_id="b")])
    body = await _summary(client, zone_id="a")
    assert body["total_detections"] == 1
    assert body["unique_people"] == 1


async def test_empty_range_returns_zeros_not_errors(client, db_session_factory):
    """A range with no data reports zeros and null dwell rather than failing."""
    body = await _summary(client)
    assert body["total_detections"] == 0
    assert body["unique_people"] == 0
    assert body["avg_dwell_seconds"] is None
    assert body["busiest_hour_start"] is None
    assert body["busiest_hour_people"] == 0
