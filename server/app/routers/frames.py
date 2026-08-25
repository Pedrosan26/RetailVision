"""
frames.py

Live camera preview over WebSockets.

Camera nodes hold one publisher socket open and push JPEG frames down it;
dashboard viewers hold a watcher socket open and receive them. Frames are
kept in memory only -- one per node, overwritten as each arrives, never
written to disk or the database. They are a visualization convenience and
are deliberately not part of the anonymized detection record.

WebSockets rather than an HTTP POST per frame: the previous design paid a
full request round-trip for every image, which capped the practical rate
at a few frames a second and made most of the cost protocol overhead
rather than picture. One long-lived connection removes that per-frame
cost, and lets the server push to viewers instead of having them poll.

The hub never queues. A frame that cannot be delivered to a viewer
immediately is dropped for that viewer, because in a live preview a late
frame has no value -- the next one is already more accurate. That keeps
one slow viewer from applying backpressure to the camera node or to
anyone else watching.
"""

import asyncio
import contextlib

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from ..config import Settings, get_settings

router = APIRouter(prefix="/api/v1", tags=["frames"])


class FrameHub:
    """Routes each camera node's latest frame to whoever is currently watching it."""

    def __init__(self) -> None:
        """Start with no publishers, no viewers, and no frames."""
        self._viewers: dict[str, set[WebSocket]] = {}
        self._latest: dict[str, bytes] = {}
        self._publishers: set[str] = set()

    @property
    def live_cameras(self) -> list[str]:
        """Camera nodes with a publisher connected right now, in a stable order."""
        return sorted(self._publishers)

    def latest(self, camera_node_id: str) -> bytes | None:
        """The most recent frame from a camera node, or None if it has sent nothing yet."""
        return self._latest.get(camera_node_id)

    def add_publisher(self, camera_node_id: str) -> None:
        """Record that a camera node has opened its publisher socket."""
        self._publishers.add(camera_node_id)

    def remove_publisher(self, camera_node_id: str) -> None:
        """Forget a camera node's publisher, and its last frame -- a stale image is worse than none."""
        self._publishers.discard(camera_node_id)
        self._latest.pop(camera_node_id, None)

    def add_viewer(self, camera_node_id: str, socket: WebSocket) -> None:
        """Register a dashboard viewer for one camera node."""
        self._viewers.setdefault(camera_node_id, set()).add(socket)

    def remove_viewer(self, camera_node_id: str, socket: WebSocket) -> None:
        """Deregister a viewer, dropping the camera's entry once nobody is left watching it."""
        watching = self._viewers.get(camera_node_id)
        if watching is None:
            return
        watching.discard(socket)
        if not watching:
            self._viewers.pop(camera_node_id, None)

    async def publish(self, camera_node_id: str, frame: bytes) -> None:
        """Store a frame as this camera's latest and push it to everyone watching."""
        self._latest[camera_node_id] = frame
        watching = self._viewers.get(camera_node_id)
        if not watching:
            return
        # send_bytes on a viewer whose socket has already gone raises rather
        # than returning an error, and one such viewer must not abort delivery
        # to the others -- so failures are gathered and the sockets dropped.
        results = await asyncio.gather(
            *(viewer.send_bytes(frame) for viewer in list(watching)),
            return_exceptions=True,
        )
        for viewer, result in zip(list(watching), results):
            if isinstance(result, Exception):
                watching.discard(viewer)


hub = FrameHub()


@router.get("/frames/cameras")
async def list_live_cameras() -> dict[str, list[str]]:
    """List the camera nodes currently streaming, so the dashboard knows what it can show."""
    return {"cameras": hub.live_cameras}


@router.websocket("/ws/publish/{camera_node_id}")
async def publish_frames(
    websocket: WebSocket,
    camera_node_id: str,
    settings: Settings = Depends(get_settings),
) -> None:
    """Accept a camera node's frame stream, authenticated by the same per-node key as ingest.

    The key is the first message on the socket rather than a query
    parameter: a URL is logged in full by servers and proxies, so a key
    carried there ends up written to disk in plaintext on every connection.
    The cost is that the socket must be accepted before the key can be
    read, so an unauthenticated peer can hold a connection open for the
    moment it takes to fail the check.
    """
    await websocket.accept()
    try:
        presented = await websocket.receive_text()
    except WebSocketDisconnect:
        return

    expected = settings.camera_node_api_key_map().get(camera_node_id)
    if expected is None or presented != expected:
        # 1008 is "policy violation", the closest close code to a refusal.
        await websocket.close(code=1008, reason="Invalid API key for camera node")
        return

    hub.add_publisher(camera_node_id)
    try:
        while True:
            await hub.publish(camera_node_id, await websocket.receive_bytes())
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove_publisher(camera_node_id)


@router.websocket("/ws/watch/{camera_node_id}")
async def watch_frames(websocket: WebSocket, camera_node_id: str) -> None:
    """Stream one camera node's frames to a dashboard viewer until it disconnects."""
    await websocket.accept()
    hub.add_viewer(camera_node_id, websocket)
    try:
        # Send whatever is already in hand so a viewer joining between frames
        # sees a picture immediately instead of an empty panel.
        first = hub.latest(camera_node_id)
        if first is not None:
            await websocket.send_bytes(first)
        # The viewer sends nothing; this read exists only to notice it leaving.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            hub.remove_viewer(camera_node_id, websocket)
