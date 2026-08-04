"""
test_detections.py

Endpoint tests for GET /api/v1/detections: ordering, limit, and
camera-node/zone filtering, seeded directly into the test DB.
"""

from datetime import datetime, timedelta, timezone

from app.models.detection import DetectionEvent

BASE_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _event(offset_minutes: int, camera_node_id: str = "node-a", zone_id: str | None = None) -> DetectionEvent:
    """Build a DetectionEvent offset_minutes after BASE_TIME, for deterministic ordering in tests."""
    return DetectionEvent(
        camera_node_id=camera_node_id,
        timestamp=BASE_TIME + timedelta(minutes=offset_minutes),
        zone_id=zone_id,
        count=None,
        age_group="18-40",
        gender="Male",
        emotion="Neutral",
        dwell_seconds=None,
        engagement_score=None,
    )


async def _seed(db_session_factory, events: list[DetectionEvent]) -> None:
    """Insert the given events directly into the test DB."""
    async with db_session_factory() as session:
        session.add_all(events)
        await session.commit()


async def test_returns_newest_first(client, db_session_factory):
    """Detections are returned newest timestamp first."""
    await _seed(db_session_factory, [_event(0), _event(10), _event(5)])
    response = await client.get("/api/v1/detections")
    assert response.status_code == 200
    timestamps = [row["timestamp"] for row in response.json()]
    assert timestamps == sorted(timestamps, reverse=True)


async def test_limit_is_respected(client, db_session_factory):
    """The limit query param caps how many rows are returned."""
    await _seed(db_session_factory, [_event(i) for i in range(5)])
    response = await client.get("/api/v1/detections", params={"limit": 2})
    assert response.status_code == 200
    assert len(response.json()) == 2


async def test_filters_by_camera_node_id(client, db_session_factory):
    """Only detections from the requested camera node are returned."""
    await _seed(
        db_session_factory,
        [_event(0, camera_node_id="node-a"), _event(1, camera_node_id="node-b")],
    )
    response = await client.get("/api/v1/detections", params={"camera_node_id": "node-b"})
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["camera_node_id"] == "node-b"


async def test_filters_by_zone_id(client, db_session_factory):
    """Only detections tagged with the requested zone are returned."""
    await _seed(
        db_session_factory,
        [_event(0, zone_id="entrance"), _event(1, zone_id="checkout"), _event(2, zone_id=None)],
    )
    response = await client.get("/api/v1/detections", params={"zone_id": "entrance"})
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["zone_id"] == "entrance"
