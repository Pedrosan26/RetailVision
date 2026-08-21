"""
test_occupancy.py

Endpoint tests for GET /api/v1/occupancy/live: returns the latest count
per zone (or camera node, before zones exist), not every historical row.
"""

from datetime import datetime, timedelta, timezone

from app.models.detection import DetectionEvent
from tests.conftest import TEST_API_KEY, TEST_CAMERA_NODE_ID

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


async def _ingest(client, camera_node_id: str, api_key: str, positions: list[tuple[float, float]], zone_id="working_area_a"):
    """Ingest one detection per position, as one camera's view of a zone at this instant."""
    now = datetime.now(timezone.utc)
    return await client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": api_key},
        json={
            "camera_node_id": camera_node_id,
            "records": [
                {
                    "timestamp": now.isoformat(),
                    "zone_id": zone_id,
                    "world_x": x,
                    "world_y": y,
                    "count": len(positions),
                    "age_group": "18-40",
                    "gender": "Male",
                    "emotion": "neutral",
                }
                for x, y in positions
            ],
        },
    )


async def test_one_person_seen_by_three_cameras_counts_once(client):
    """The end-to-end claim: three cameras watching one person report a headcount of one."""
    await _ingest(client, "cam-a", "key-a", [(2.0, 1.5)])
    await _ingest(client, "cam-b", "key-b", [(2.2, 1.6)])
    await _ingest(client, "cam-c", "key-c", [(1.9, 1.4)])

    zones = (await client.get("/api/v1/occupancy/zones")).json()
    assert len(zones) == 1
    assert zones[0]["zone_id"] == "working_area_a"
    assert zones[0]["total"] == 1, "one person seen by three cameras must count once"
    assert zones[0]["cameras_reporting"] == 3
    assert zones[0]["per_camera"] == {"cam-a": 1, "cam-b": 1, "cam-c": 1}


async def test_distinct_people_are_counted_separately(client):
    """Two cameras each seeing a different person report two people."""
    await _ingest(client, "cam-a", "key-a", [(0.0, 0.0)])
    await _ingest(client, "cam-b", "key-b", [(6.0, 4.0)])

    zones = (await client.get("/api/v1/occupancy/zones")).json()
    assert zones[0]["total"] == 2


async def test_partial_overlap_counts_the_union(client):
    """Cameras covering different parts of a zone contribute everyone, merging only the shared person."""
    await _ingest(client, "cam-a", "key-a", [(0.0, 0.0), (3.0, 2.0)])
    await _ingest(client, "cam-b", "key-b", [(3.1, 2.1), (6.0, 4.0)])

    zones = (await client.get("/api/v1/occupancy/zones")).json()
    assert zones[0]["total"] == 3
    assert zones[0]["per_camera"] == {"cam-a": 2, "cam-b": 2}


async def test_records_without_a_position_do_not_produce_a_zone(client):
    """Nodes running without zones report no position and cannot contribute a headcount."""
    await client.post(
        "/api/v1/ingest",
        headers={"X-API-Key": TEST_API_KEY},
        json={
            "camera_node_id": TEST_CAMERA_NODE_ID,
            "records": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "zone_id": None,
                    "count": 3,
                    "age_group": "18-40",
                    "gender": "Male",
                    "emotion": "neutral",
                }
            ],
        },
    )
    assert (await client.get("/api/v1/occupancy/zones")).json() == []


async def test_live_occupancy_keeps_cameras_separate_within_one_zone(client):
    """Several cameras on one zone each get their own row, rather than the last one silently winning."""
    await _ingest(client, "cam-a", "key-a", [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)])
    await _ingest(client, "cam-b", "key-b", [(1.0, 1.0), (2.0, 1.0)])

    rows = (await client.get("/api/v1/occupancy/live")).json()
    assert {row["camera_node_id"]: row["count"] for row in rows} == {"cam-a": 3, "cam-b": 2}
