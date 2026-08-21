"""
test_ingest.py

Endpoint tests for POST /api/v1/ingest: a valid batch is accepted and
persisted, a malformed record is rejected, and both a missing and a
wrong API key are rejected -- all against the in-memory SQLite DB set
up in conftest.py.
"""

from app.models.detection import DetectionEvent
from sqlalchemy import select
from tests.conftest import TEST_API_KEY, TEST_CAMERA_NODE_ID

VALID_RECORD = {
    "timestamp": "2026-08-03T10:00:00Z",
    "zone_id": None,
    "count": None,
    "age_group": "18-40",
    "gender": "Male",
    "emotion": "Neutral",
    "dwell_seconds": None,
    "engagement_score": None,
}


async def test_valid_batch_is_accepted_and_persisted(client, db_session_factory):
    """A well-formed batch with the correct API key is accepted, persisted, and attributed to its camera node."""
    response = await client.post(
        "/api/v1/ingest",
        json={"camera_node_id": TEST_CAMERA_NODE_ID, "records": [VALID_RECORD, VALID_RECORD]},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 202
    assert response.json() == {"accepted": 2}

    async with db_session_factory() as session:
        rows = (await session.execute(select(DetectionEvent))).scalars().all()
        assert len(rows) == 2
        assert all(row.camera_node_id == TEST_CAMERA_NODE_ID for row in rows)
        assert all(row.age_group == "18-40" for row in rows)


async def test_malformed_record_is_rejected(client):
    """A batch missing a required field (age_group) is rejected with 422."""
    malformed_record = {k: v for k, v in VALID_RECORD.items() if k != "age_group"}
    response = await client.post(
        "/api/v1/ingest",
        json={"camera_node_id": TEST_CAMERA_NODE_ID, "records": [malformed_record]},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 422


async def test_missing_api_key_is_rejected(client):
    """A request with no X-API-Key header at all is rejected with 401."""
    response = await client.post(
        "/api/v1/ingest",
        json={"camera_node_id": TEST_CAMERA_NODE_ID, "records": [VALID_RECORD]},
    )
    assert response.status_code == 401


async def test_wrong_api_key_is_rejected(client):
    """A request with an incorrect X-API-Key header is rejected with 401."""
    response = await client.post(
        "/api/v1/ingest",
        json={"camera_node_id": TEST_CAMERA_NODE_ID, "records": [VALID_RECORD]},
        headers={"X-API-Key": "not-the-right-key"},
    )
    assert response.status_code == 401


async def test_unknown_camera_node_is_rejected(client):
    """A camera_node_id with no configured key is rejected with 401, even with a plausible-looking key."""
    response = await client.post(
        "/api/v1/ingest",
        json={"camera_node_id": "some-other-node", "records": [VALID_RECORD]},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 401
