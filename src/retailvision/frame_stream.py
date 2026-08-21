"""
frame_stream.py

Optional live-preview sink: periodically ships the current annotated
frame to the central server's ephemeral frame endpoint (see
server/app/routers/frames.py) so the dashboard can show what this camera
node currently sees. Unlike output_log.py/remote_log.py, frames sent
here are held in memory server-side only, never persisted -- this is a
deliberate, temporary trade-off against the project's edge-inference
privacy design, not a new permanent sink. Throttled to min_interval
regardless of capture frame rate, since sending every frame would swamp
the network for no visual benefit at typical browser refresh rates.
Runs on a background thread, same pattern as RemoteLogShipper, so a
slow or unreachable server never stalls frame capture. min_interval
defaults low (0.3s) since this only bounds how often a frame leaves this
process -- the server re-serves whatever's latest at a steady rate to
each dashboard viewer regardless (see frames.py's /stream endpoint), so
this value mostly just trades local network bandwidth for freshness on a
trusted LAN, not viewer-side smoothness.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import requests


class FrameStreamer:
    """Throttles and ships JPEG-encoded frames to the server's live-preview endpoint, off the capture thread."""

    def __init__(
        self,
        server_url: str,
        camera_node_id: str,
        api_key: str,
        min_interval: float = 0.3,
        request_timeout: float = 3.0,
    ) -> None:
        """Configure the target server, this node's identity, and the send-rate throttle."""
        self._frame_url = server_url.rstrip("/") + f"/api/v1/frames/{camera_node_id}"
        self._api_key = api_key
        self._min_interval = min_interval
        self._request_timeout = request_timeout
        self._last_sent = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1)

    def send(self, frame) -> None:
        """Encode and ship the given frame as JPEG, throttled to at most one send per min_interval."""
        now = time.monotonic()
        if now - self._last_sent < self._min_interval:
            return
        self._last_sent = now
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return
        self._executor.submit(self._send, buffer.tobytes())

    def _send(self, jpeg_bytes: bytes) -> None:
        """POST one JPEG frame to the server; warn and drop on any failure, without retrying."""
        headers = {"X-API-Key": self._api_key, "Content-Type": "image/jpeg"}
        try:
            response = requests.post(self._frame_url, data=jpeg_bytes, headers=headers, timeout=self._request_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Warning: failed to stream frame to {self._frame_url}: {exc}")

    def close(self) -> None:
        """Wait for any in-flight frame upload to finish."""
        self._executor.shutdown(wait=True)
