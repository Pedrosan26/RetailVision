"""
test_frames.py

Tests for the live frame WebSocket hub: a publisher authenticated by its
node key reaches every viewer of that camera, a viewer joining late is
handed the frame already in hand, a wrong key is refused, and a
publisher's disconnect clears its camera from the live list.
"""

import pytest
from app.config import get_settings
from app.main import app
from app.routers.frames import hub
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.conftest import TEST_API_KEY, TEST_CAMERA_NODE_ID, _test_settings


@pytest.fixture
def client():
    """A synchronous TestClient, which the async httpx fixture cannot replace here.

    WebSocket testing needs Starlette's own client: it drives the ASGI
    websocket scope directly, which httpx has no equivalent for. The frame
    hub touches no database, so this fixture overrides only the settings.
    """
    app.dependency_overrides[get_settings] = _test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

FRAME = b"\xff\xd8\xff\xe0 not really a jpeg, but bytes are bytes"
OTHER_FRAME = b"\xff\xd8\xff\xe0 a second frame"


@pytest.fixture(autouse=True)
def _clean_hub():
    """Reset the in-memory hub between tests, since it is module-level state."""
    hub._viewers.clear()
    hub._latest.clear()
    hub._publishers.clear()
    yield
    hub._viewers.clear()
    hub._latest.clear()
    hub._publishers.clear()


def test_publisher_frame_reaches_viewer(client: TestClient) -> None:
    """A frame sent by an authenticated publisher arrives at a connected viewer."""
    with client.websocket_connect(f"/api/v1/ws/publish/{TEST_CAMERA_NODE_ID}") as publisher:
        publisher.send_text(TEST_API_KEY)
        with client.websocket_connect(f"/api/v1/ws/watch/{TEST_CAMERA_NODE_ID}") as viewer:
            publisher.send_bytes(FRAME)
            assert viewer.receive_bytes() == FRAME


def test_viewer_joining_late_gets_the_latest_frame(client: TestClient) -> None:
    """A viewer connecting after a frame was published receives it immediately, not a blank wait."""
    with client.websocket_connect(f"/api/v1/ws/publish/{TEST_CAMERA_NODE_ID}") as publisher:
        publisher.send_text(TEST_API_KEY)
        publisher.send_bytes(FRAME)
        # Round-trip an HTTP call so the publish coroutine has certainly run.
        assert client.get("/api/v1/frames/cameras").json() == {"cameras": [TEST_CAMERA_NODE_ID]}
        with client.websocket_connect(f"/api/v1/ws/watch/{TEST_CAMERA_NODE_ID}") as viewer:
            assert viewer.receive_bytes() == FRAME


def test_wrong_api_key_is_refused(client: TestClient) -> None:
    """A publisher presenting the wrong key is closed and never becomes a live camera."""
    with client.websocket_connect(f"/api/v1/ws/publish/{TEST_CAMERA_NODE_ID}") as publisher:
        publisher.send_text("not-the-key")
        # The refusal shows up on the next read, not on the send: a socket the
        # peer has closed still accepts writes locally until the close is seen.
        with pytest.raises(WebSocketDisconnect) as closed:
            publisher.receive_bytes()
        assert closed.value.code == 1008
    assert hub.live_cameras == []


def test_unknown_camera_node_is_refused(client: TestClient) -> None:
    """A node id with no configured key cannot publish, even with a key valid for some other node."""
    with client.websocket_connect("/api/v1/ws/publish/never-configured") as publisher:
        publisher.send_text(TEST_API_KEY)
        with pytest.raises(WebSocketDisconnect) as closed:
            publisher.receive_bytes()
        assert closed.value.code == 1008
    assert hub.live_cameras == []


def test_disconnect_clears_the_camera_and_its_frame(client: TestClient) -> None:
    """A publisher going away removes it from the live list and drops its last frame, which is now stale."""
    with client.websocket_connect(f"/api/v1/ws/publish/{TEST_CAMERA_NODE_ID}") as publisher:
        publisher.send_text(TEST_API_KEY)
        publisher.send_bytes(FRAME)
        assert client.get("/api/v1/frames/cameras").json()["cameras"] == [TEST_CAMERA_NODE_ID]
    assert client.get("/api/v1/frames/cameras").json() == {"cameras": []}
    assert hub.latest(TEST_CAMERA_NODE_ID) is None


def test_each_camera_is_isolated(client: TestClient) -> None:
    """A viewer watching one camera never receives another camera's frames."""
    hub._latest["other-cam"] = OTHER_FRAME
    with client.websocket_connect(f"/api/v1/ws/publish/{TEST_CAMERA_NODE_ID}") as publisher:
        publisher.send_text(TEST_API_KEY)
        with client.websocket_connect(f"/api/v1/ws/watch/{TEST_CAMERA_NODE_ID}") as viewer:
            publisher.send_bytes(FRAME)
            assert viewer.receive_bytes() == FRAME
