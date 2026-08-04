"""
test_occupancy.py

Endpoint tests for GET /api/v1/occupancy/live: returns the latest count
per zone (or camera node, before zones exist), not every historical row.
"""

from datetime import datetime, timedelta, timezone

from app.models.detection import DetectionEvent

BASE_TIME = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _event(offset_minutes: int, camera_node_id: str, count: int, zone_id: str | None = None) -> DetectionEvent:
    """Build a DetectionEvent offset_minutes after BASE_TIME with a given occupancy count."""
    return DetectionEvent(
        camera_node_id=camera_node_id,
        timestamp=BASE_TIME + timedelta(minutes=offset_minutes),
        zone_id=zone_id,
        count=count,
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


async def test_returns_latest_count_per_camera_node_without_zones(client, db_session_factory):
    """With no zone_id set, occupancy groups by camera_node_id and returns only the newest count."""
    await _seed(
        db_session_factory,
        [_event(0, "node-a", count=1), _event(5, "node-a", count=3), _event(2, "node-a", count=2)],
    )
    response = await client.get("/api/v1/occupancy/live")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["camera_node_id"] == "node-a"
    assert rows[0]["count"] == 3
    assert rows[0]["key"] == "node-a"


async def test_groups_by_zone_id_once_populated(client, db_session_factory):
    """Once zone_id is set, occupancy groups by zone instead of camera node."""
    await _seed(
        db_session_factory,
        [
            _event(0, "node-a", count=1, zone_id="entrance"),
            _event(5, "node-a", count=4, zone_id="entrance"),
        ],
    )
    response = await client.get("/api/v1/occupancy/live")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["key"] == "entrance"
    assert rows[0]["count"] == 4


async def test_separate_nodes_get_separate_entries(client, db_session_factory):
    """Two different camera nodes each get their own occupancy entry."""
    await _seed(db_session_factory, [_event(0, "node-a", count=1), _event(0, "node-b", count=5)])
    response = await client.get("/api/v1/occupancy/live")
    assert response.status_code == 200
    keys = {row["key"] for row in response.json()}
    assert keys == {"node-a", "node-b"}
