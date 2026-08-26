"""
frame_stream.py

Optional live-preview sink: streams the annotated frame this node is
currently showing to the central server over a WebSocket, so the
dashboard can display what the camera sees.

Frames sent here live in the server's memory only -- never written to
disk or the database, and never part of the anonymized detection record
that output_log.py and remote_log.py produce. It is opt-in behind
--stream-frames and off by default, because it is the one path in this
system that moves pixels off the machine that captured them.

A WebSocket rather than one HTTP POST per frame. The request-per-frame
approach paid a full round-trip for every image, which put a ceiling of
a few frames a second on the result and spent most of the cost on
protocol rather than picture. One long-lived connection removes that
per-frame overhead.

Nothing is queued. The capture thread drops its newest frame into a
single slot, replacing whatever was there; the sender thread encodes
whatever it finds when it next looks. If encoding or the network cannot
keep up, intermediate frames are discarded rather than backing up --
in a live preview a late frame is worthless, because a newer one already
exists. That also means capture never blocks on the network, which is
the property that matters most: the pipeline's own frame rate is
unaffected by how the stream is doing.
"""

import threading
import time

import cv2
import numpy as np
import websocket

# Downscaled before encoding: the preview is for watching, not for
# analysis, and a full-resolution JPEG costs bandwidth that buys nothing
# at the size a browser shows it.
DEFAULT_MAX_WIDTH = 960
DEFAULT_JPEG_QUALITY = 70
DEFAULT_MAX_FPS = 15.0
# Reconnect backoff, in seconds, capped so a server that comes back is
# picked up promptly without hammering one that is still down.
_RETRY_BASE = 1.0
_RETRY_MAX = 15.0


def _websocket_url(server_url: str, camera_node_id: str) -> str:
    """Build the publisher WebSocket URL, converting an http(s) base to ws(s).

    The API key is deliberately not in the URL: servers and proxies log
    request lines in full, so a key here would be written to disk in
    plaintext on every connection. It is sent as the first message instead.
    """
    base = server_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return f"{base}/api/v1/ws/publish/{camera_node_id}"


class FrameStreamer:
    """Streams JPEG-encoded frames to the server over a WebSocket, off the capture thread."""

    def __init__(
        self,
        server_url: str,
        camera_node_id: str,
        api_key: str,
        max_fps: float = DEFAULT_MAX_FPS,
        max_width: int = DEFAULT_MAX_WIDTH,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> None:
        """Configure the target server and start the background sender thread."""
        self._url = _websocket_url(server_url, camera_node_id)
        self._api_key = api_key
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._max_width = max_width
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        self._pending: np.ndarray | None = None
        self._condition = threading.Condition()
        self._closed = False
        self._warned = False

        self._thread = threading.Thread(target=self._run, name="frame-streamer", daemon=True)
        self._thread.start()

    def send(self, frame: np.ndarray) -> None:
        """Offer a frame for streaming, replacing any frame not yet sent.

        Copies because the caller goes on drawing into its frame buffer
        while the sender thread is still reading this one.
        """
        with self._condition:
            if self._closed:
                return
            self._pending = frame.copy()
            self._condition.notify()

    def close(self) -> None:
        """Stop the sender thread and wait briefly for it to finish."""
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _take_pending(self) -> np.ndarray | None:
        """Block until a frame is waiting or the streamer closes, then take it."""
        with self._condition:
            while self._pending is None and not self._closed:
                self._condition.wait()
            frame, self._pending = self._pending, None
            return None if self._closed else frame

    def _encode(self, frame: np.ndarray) -> bytes | None:
        """Downscale to the preview width and JPEG-encode, or None if encoding fails."""
        height, width = frame.shape[:2]
        if width > self._max_width:
            scale = self._max_width / width
            frame = cv2.resize(frame, (self._max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", frame, self._encode_params)
        return buffer.tobytes() if ok else None

    def _run(self) -> None:
        """Hold a connection open, sending the newest frame available, reconnecting as needed."""
        attempt = 0
        while not self._closed:
            try:
                connection = websocket.create_connection(self._url, timeout=5)
            except Exception as error:
                if not self._warned:
                    print(f"Warning: frame streaming could not connect ({error}); retrying in the background.")
                    self._warned = True
                # Sleep in slices so close() is not left waiting on a long backoff.
                delay = min(_RETRY_BASE * (2**attempt), _RETRY_MAX)
                attempt += 1
                deadline = time.monotonic() + delay
                while not self._closed and time.monotonic() < deadline:
                    time.sleep(0.1)
                continue

            attempt = 0
            self._warned = False
            try:
                connection.send(self._api_key)
            except Exception as error:
                print(f"Warning: frame streaming could not authenticate ({error}).")
                connection.close()
                continue
            print("Frame streaming connected.")
            try:
                self._pump(connection)
            except Exception as error:
                # Never silent: a stream that stops working looks identical to
                # one nobody is watching, and the reconnect below would other-
                # wise hide a repeating failure behind a healthy-looking log.
                print(f"Warning: frame streaming interrupted ({type(error).__name__}: {error}); reconnecting.")
            finally:
                try:
                    connection.close()
                except Exception:
                    pass

    def _pump(self, connection: websocket.WebSocket) -> None:
        """Send frames over one open connection until it fails or the streamer closes."""
        next_send = 0.0
        while not self._closed:
            frame = self._take_pending()
            if frame is None:
                return
            now = time.monotonic()
            if now < next_send:
                continue
            payload = self._encode(frame)
            if payload is None:
                continue
            connection.send_binary(payload)
            next_send = now + self._min_interval
