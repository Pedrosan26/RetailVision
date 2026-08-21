"""
test_zone_geometry.py

Endpoint tests for zone geometry: authenticated upload, upsert
semantics, and readback for the dashboard's floor map.
"""

SQUARE = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]


async def _upload(client, polygon=SQUARE, node="cam-a", key="key-a", zone_id="entrance"):
    """POST one zone's polygon as the given node."""
    return await client.post(
        "/api/v1/zones/geometry",
        json={"camera_node_id": node, "zones": [{"zone_id": zone_id, "polygon": polygon}]},
        headers={"X-API-Key": key},
    )


async def test_upload_and_readback(client):
    """An authenticated upload is stored and served back with its polygon intact."""
    response = await _upload(client)
    assert response.status_code == 204
    body = (await client.get("/api/v1/zones/geometry")).json()
    assert len(body) == 1
    assert body[0]["zone_id"] == "entrance"
    assert body[0]["polygon"] == SQUARE
    assert body[0]["camera_node_id"] == "cam-a"


async def test_second_upload_replaces_the_polygon(client):
    """Re-uploading a zone overwrites its polygon -- last writer wins, no duplicates."""
    await _upload(client)
    moved = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]
    response = await _upload(client, polygon=moved)
    assert response.status_code == 204
    body = (await client.get("/api/v1/zones/geometry")).json()
    assert len(body) == 1
    assert body[0]["polygon"] == moved


async def test_wrong_key_is_rejected(client):
    """A bad API key cannot write geometry."""
    response = await _upload(client, key="wrong")
    assert response.status_code == 401


async def test_degenerate_polygon_is_rejected(client):
    """Fewer than three vertices is not an area and fails validation."""
    response = await _upload(client, polygon=[[0.0, 0.0], [1.0, 1.0]])
    assert response.status_code == 422
