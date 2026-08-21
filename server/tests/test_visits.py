"""
test_visits.py

Endpoint tests for GET /api/v1/visits: folding events into per-person
rows, duration and dominant emotion, the trackless-row exclusion, and
filters.
"""

from datetime import datetime, timedelta, timezone

from app.models.detection import DetectionEvent

BASE_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _event(
    offset_minutes: float,
    track_id: str | None = "t1",
    camera_node_id: str = "node-a",
    zone_id: str | None = None,
    emotion: str = "neutral",
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
        dwell_seconds=None,
        engagement_score=None,
    )


async def _seed(db_session_factory, events: list[DetectionEvent]) -> None:
    """Insert the given events directly into the test DB."""
    async with db_session_factory() as session:
        session.add_all(events)
        await session.commit()


async def _visits(client, **params) -> list[dict]:
    """Query the endpoint over the seeded range with optional extra filters."""
    response = await client.get(
        "/api/v1/visits",
        params={
            "since": BASE_TIME.isoformat(),
            "until": (BASE_TIME + timedelta(hours=1)).isoformat(),
            **params,
        },
    )
    assert response.status_code == 200
    return response.json()


async def test_events_fold_into_one_visit(client, db_session_factory):
    """A track's records become one row with the span, mode zone, and emotion mix."""
    await _seed(
        db_session_factory,
        [
            _event(0, emotion="neutral", zone_id="a"),
            _event(2, emotion="happy", zone_id="a"),
            _event(4, emotion="neutral", zone_id=None),
        ],
    )
    visits = await _visits(client)
    assert len(visits) == 1
    visit = visits[0]
    assert visit["duration_seconds"] == 240.0
    assert visit["events"] == 3
    assert visit["zone_id"] == "a"
    assert visit["dominant_emotion"] == "neutral"
    assert visit["emotion_distribution"] == {"neutral": 2, "happy": 1}


async def test_same_track_id_on_two_cameras_is_two_visits(client, db_session_factory):
    """Track IDs are per-camera, so identical IDs from different nodes do not merge."""
    await _seed(db_session_factory, [_event(0), _event(1, camera_node_id="node-b")])
    assert len(await _visits(client)) == 2


async def test_trackless_rows_are_excluded(client, db_session_factory):
    """Pre-upgrade rows without a track are not shown as hundreds of one-event visits."""
    await _seed(db_session_factory, [_event(0, track_id=None), _event(1, track_id=None), _event(2)])
    visits = await _visits(client)
    assert len(visits) == 1
    assert visits[0]["events"] == 1


async def test_zone_filter_and_ordering(client, db_session_factory):
    """Zone scoping keeps matching visits only, and results come newest-last-seen first."""
    await _seed(
        db_session_factory,
        [
            _event(0, track_id="early", zone_id="a"),
            _event(30, track_id="late", zone_id="a"),
            _event(10, track_id="other", zone_id="b"),
        ],
    )
    visits = await _visits(client, zone_id="a")
    assert [v["track_id"] for v in visits] == ["late", "early"]


async def test_summary_reports_peak_occupancy(client, db_session_factory):
    """The summary's peak is the largest node-reported headcount in the range."""
    first, second = _event(0), _event(1)
    first.count, second.count = 2, 5
    await _seed(db_session_factory, [first, second])
    response = await client.get("/api/v1/summary", params={"since": BASE_TIME.isoformat()})
    assert response.status_code == 200
    assert response.json()["peak_occupancy"] == 5
