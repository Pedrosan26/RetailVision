"""
test_aggregates.py

Endpoint tests for GET /api/v1/aggregates: bucketing, per-bucket
demographic/emotion distribution, dwell/engagement averages, and window
parsing errors.
"""

from datetime import datetime, timedelta, timezone

from app.models.detection import DetectionEvent
from app.routers.aggregates import parse_window

BASE_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _event(
    offset_minutes: int,
    age_group: str = "18-40",
    gender: str = "Male",
    emotion: str = "Neutral",
    dwell_seconds: float | None = None,
) -> DetectionEvent:
    """Build a DetectionEvent offset_minutes after BASE_TIME."""
    return DetectionEvent(
        camera_node_id="node-a",
        timestamp=BASE_TIME + timedelta(minutes=offset_minutes),
        zone_id=None,
        count=None,
        age_group=age_group,
        gender=gender,
        emotion=emotion,
        dwell_seconds=dwell_seconds,
        engagement_score=None,
    )


async def _seed(db_session_factory, events: list[DetectionEvent]) -> None:
    """Insert the given events directly into the test DB."""
    async with db_session_factory() as session:
        session.add_all(events)
        await session.commit()


def test_parse_window_accepts_minutes_hours_days():
    """parse_window() converts '5m'/'1h'/'2d' style strings into the right timedelta."""
    assert parse_window("5m") == timedelta(minutes=5)
    assert parse_window("1h") == timedelta(hours=1)
    assert parse_window("2d") == timedelta(days=2)


def test_parse_window_rejects_unknown_unit():
    """An unsupported unit (e.g. seconds) raises ValueError."""
    try:
        parse_window("30s")
        assert False, "expected ValueError"
    except ValueError:
        pass


async def test_events_are_grouped_into_correct_buckets(client, db_session_factory):
    """Two events within the same 5-minute window land in one bucket; a later event starts a new one."""
    await _seed(db_session_factory, [_event(0), _event(2), _event(6)])
    response = await client.get(
        "/api/v1/aggregates",
        params={"window": "5m", "since": BASE_TIME.isoformat(), "until": (BASE_TIME + timedelta(minutes=10)).isoformat()},
    )
    assert response.status_code == 200
    buckets = response.json()
    assert len(buckets) == 2
    assert buckets[0]["detection_count"] == 2
    assert buckets[1]["detection_count"] == 1


async def test_distributions_and_averages_are_correct(client, db_session_factory):
    """Demographic/emotion distributions and dwell average are computed correctly within a bucket."""
    await _seed(
        db_session_factory,
        [
            _event(0, age_group="18-40", gender="Male", emotion="Happy", dwell_seconds=10.0),
            _event(1, age_group="18-40", gender="Female", emotion="Neutral", dwell_seconds=20.0),
        ],
    )
    response = await client.get(
        "/api/v1/aggregates",
        params={"window": "5m", "since": BASE_TIME.isoformat(), "until": (BASE_TIME + timedelta(minutes=5)).isoformat()},
    )
    assert response.status_code == 200
    bucket = response.json()[0]
    assert bucket["age_group_distribution"] == {"18-40": 2}
    assert bucket["gender_distribution"] == {"Male": 1, "Female": 1}
    assert bucket["emotion_distribution"] == {"Happy": 1, "Neutral": 1}
    assert bucket["avg_dwell_seconds"] == 15.0
    assert bucket["avg_engagement_score"] is None


async def test_invalid_window_returns_400(client, db_session_factory):
    """A malformed window string is rejected with 400, not a 500."""
    response = await client.get("/api/v1/aggregates", params={"window": "not-a-window"})
    assert response.status_code == 400


async def _filtered(client, **params) -> list[dict]:
    """Query the endpoint over the whole seeded range with the given extra filters."""
    response = await client.get(
        "/api/v1/aggregates",
        params={
            "window": "5m",
            "since": BASE_TIME.isoformat(),
            "until": (BASE_TIME + timedelta(minutes=5)).isoformat(),
            **params,
        },
    )
    assert response.status_code == 200
    return response.json()


async def test_single_demographic_filter_narrows_the_bucket(client, db_session_factory):
    """Filtering on one emotion counts only those events."""
    await _seed(
        db_session_factory,
        [
            _event(0, emotion="Happy"),
            _event(1, emotion="Neutral"),
            _event(2, emotion="Neutral"),
        ],
    )
    buckets = await _filtered(client, emotion="Neutral")
    assert len(buckets) == 1
    assert buckets[0]["detection_count"] == 2
    assert buckets[0]["emotion_distribution"] == {"Neutral": 2}


async def test_repeated_filter_values_read_as_any_of_them(client, db_session_factory):
    """Several values of one dimension match events with any of them."""
    await _seed(
        db_session_factory,
        [
            _event(0, emotion="Happy"),
            _event(1, emotion="Neutral"),
            _event(2, emotion="negative"),
        ],
    )
    buckets = await _filtered(client, emotion=["Happy", "Neutral"])
    assert buckets[0]["detection_count"] == 2
    assert buckets[0]["emotion_distribution"] == {"Happy": 1, "Neutral": 1}


async def test_filters_on_different_dimensions_combine(client, db_session_factory):
    """Age and gender filters intersect rather than union."""
    await _seed(
        db_session_factory,
        [
            _event(0, age_group="18-40", gender="Male"),
            _event(1, age_group="18-40", gender="Female"),
            _event(2, age_group="41-64", gender="Male"),
        ],
    )
    buckets = await _filtered(client, age_group="18-40", gender="Male")
    assert buckets[0]["detection_count"] == 1
    assert buckets[0]["age_group_distribution"] == {"18-40": 1}
    assert buckets[0]["gender_distribution"] == {"Male": 1}


async def test_one_person_across_many_events_counts_once(client, db_session_factory):
    """Several records sharing a track are one person, however many events they produced."""
    events = [_event(0), _event(1), _event(2)]
    for event in events:
        event.track_id = "abc123"
    await _seed(db_session_factory, events)
    buckets = await _filtered(client)
    assert buckets[0]["detection_count"] == 3
    assert buckets[0]["unique_people"] == 1


async def test_same_track_id_on_different_nodes_is_two_people(client, db_session_factory):
    """Track IDs are only unique within a camera node, so identical IDs from two nodes are not one person."""
    first, second = _event(0), _event(1)
    first.track_id = second.track_id = "abc123"
    second.camera_node_id = "node-b"
    await _seed(db_session_factory, [first, second])
    buckets = await _filtered(client)
    assert buckets[0]["unique_people"] == 2


async def test_rows_without_a_track_count_as_one_person_each(client, db_session_factory):
    """A node too old to send a track_id behaves exactly as it did before the column existed."""
    await _seed(db_session_factory, [_event(0), _event(1)])
    buckets = await _filtered(client)
    assert buckets[0]["detection_count"] == 2
    assert buckets[0]["unique_people"] == 2


async def test_omitting_filters_counts_everything(client, db_session_factory):
    """No demographic filters means no narrowing -- what clearing the filters sends."""
    await _seed(
        db_session_factory,
        [_event(0, emotion="Happy"), _event(1, emotion="Neutral"), _event(2, emotion="negative")],
    )
    buckets = await _filtered(client)
    assert buckets[0]["detection_count"] == 3
