"""
frames.py

Ephemeral live-preview endpoint: camera nodes optionally POST their most
recent annotated frame here so the dashboard can show what each camera
currently sees. Held in memory only -- one JPEG per node, overwritten on
each POST, never written to disk or the database -- since this is a
visualization convenience, not part of the frozen anonymized detection
record. This is a deliberate, temporary trade-off against the project's
edge-inference privacy design (raw frames otherwise never leave a camera
node's machine) -- expected to be gated or removed once real privacy
policy work lands, see remote_log.py.
"""

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..config import get_settings

router = APIRouter(prefix="/api/v1", tags=["frames"])

_latest_frames: dict[str, bytes] = {}

STREAM_FPS = 10
STREAM_BOUNDARY = b"frame"


@router.post("/frames/{camera_node_id}", status_code=204)
async def post_frame(camera_node_id: str, request: Request, x_api_key: str | None = Header(None)) -> None:
    """Store the latest JPEG frame for a camera node, overwriting whatever was there before."""
    expected = get_settings().camera_node_api_key_map().get(camera_node_id)
    if expected is None or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key for camera node")
    _latest_frames[camera_node_id] = await request.body()


@router.get("/frames/{camera_node_id}")
async def get_frame(camera_node_id: str) -> Response:
    """Return the most recently posted JPEG frame for a camera node, or 404 if none has arrived yet."""
    frame = _latest_frames.get(camera_node_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame received yet for this camera node")
    return Response(content=frame, media_type="image/jpeg")


async def _mjpeg_generator(camera_node_id: str) -> AsyncIterator[bytes]:
    """Re-serve whatever frame is currently latest for camera_node_id at a steady STREAM_FPS, for a smooth-looking feed even when the underlying source updates less often."""
    while True:
        frame = _latest_frames.get(camera_node_id)
        if frame is not None:
            yield (
                b"--" + STREAM_BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(1 / STREAM_FPS)


@router.get("/frames/{camera_node_id}/stream")
async def stream_frame(camera_node_id: str) -> StreamingResponse:
    """Serve a continuously-updating MJPEG stream of the camera node's latest frame, browser-renderable directly in an <img> tag."""
    if camera_node_id not in get_settings().camera_node_api_key_map():
        raise HTTPException(status_code=404, detail="Unknown camera node")
    return StreamingResponse(
        _mjpeg_generator(camera_node_id),
        media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY.decode()}",
    )
