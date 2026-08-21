"""
camera_check.py

Opens every connected camera at once and shows them tiled in a single
window, to find out how many a machine can actually run simultaneously.

Each camera is read on its own thread. That is not an optimization, it is
required for the measurement to mean anything: reading several cameras in
turn on one thread makes every read wait for the one before it, so the
loop rate collapses to the sum of their latencies and every camera appears
equally slow no matter how healthy the bus is. Threaded reads let each
camera run at its own pace, so a genuinely starved one stands out from
its neighbours.

Two different failures are worth telling apart, and both are quiet:
a camera that opens and then never delivers a frame, and a set of cameras
that all deliver at a reduced rate because the bus cannot carry them.
The first shows as dropped reads with no frames, the second as a low
capture rate on several cameras at once.

With --detect, the face detector is run on every stream from the display
loop, which answers the heavier question of whether the machine can carry
inference across that many cameras. Capture rates are still measured on
the camera threads, so the two effects stay separable.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/setup/camera_check.py
       PYTHONPATH=. ./venv/bin/python3 scripts/setup/camera_check.py --source 0 1 2 3
       PYTHONPATH=. ./venv/bin/python3 scripts/setup/camera_check.py --detect
Press 'q' to quit.
"""

import argparse
import threading
import time

import cv2
import numpy as np

MAX_PROBE_INDEX = 8
TILE_WIDTH = 640
# Every tile is rendered at one fixed shape and frames are letterboxed into it,
# since cameras on the same machine often differ in aspect ratio.
TILE_ASPECT = 9 / 16
FPS_SMOOTHING = 0.9
HEALTHY_FPS = 10.0
# Variance of the Laplacian: a standard focus measure. A sharp image has strong
# local intensity changes, a blurred one does not. The absolute number depends on
# scene content, so it is useful for comparing one camera against itself while
# moving a target, not for comparing different cameras to each other.
SHARP_ENOUGH = 100.0
# Below this the sample is too short for rates or failures to mean anything.
MIN_SAMPLE_SECONDS = 8.0


def parse_args() -> argparse.Namespace:
    """Parse which cameras to open, the tile size, the requested resolution, and whether to run detection."""
    parser = argparse.ArgumentParser(description="Check how many cameras can run at once")
    parser.add_argument("--source", nargs="+", default=None, help="Camera indices to open (default: probe and use all)")
    parser.add_argument("--detect", action="store_true", help="Also run the face detector on every stream")
    parser.add_argument("--tile-width", type=int, default=TILE_WIDTH, help=f"Width of each tile in pixels (default: {TILE_WIDTH})")
    parser.add_argument("--width", type=int, default=None, help="Request this capture width from every camera")
    parser.add_argument("--height", type=int, default=None, help="Request this capture height from every camera")
    parser.add_argument("--fourcc", default=None, help="Request a pixel format, e.g. MJPG -- far less USB bandwidth than uncompressed")
    parser.add_argument("--fps", type=float, default=None, help="Request this capture frame rate from every camera")
    return parser.parse_args()


def probe_cameras() -> list[int]:
    """Find camera indices that both open and deliver a frame, probing one at a time."""
    found = []
    for index in range(MAX_PROBE_INDEX):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                found.append(index)
        capture.release()
    return found


class Stream:
    """One camera, read continuously on its own thread so its rate is independent of the others."""

    def __init__(
        self,
        index: int,
        width: int | None,
        height: int | None,
        fourcc: str | None = None,
        fps: float | None = None,
    ) -> None:
        """Open the camera, optionally forcing format, resolution and rate, and start its reader thread."""
        self.index = index
        self.capture = cv2.VideoCapture(index)
        # Pixel format is set before resolution deliberately: a camera picks which
        # resolutions it can offer based on the format, and several USB cameras on
        # one bus usually only fit at all in a compressed format like MJPG.
        if fourcc is not None:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.opened = self.capture.isOpened()

        self.frames = 0
        self.failures = 0
        self.fps = 0.0
        # Measured from when this camera's reader thread starts, not from when the
        # display loop begins, so warmup frames are not divided by a shorter window.
        self._started = time.monotonic()
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = self.opened
        self._last_time: float | None = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        if self.opened:
            self._thread.start()

    @property
    def resolution(self) -> tuple[int, int]:
        """The capture resolution the camera actually settled on, which may not be what was requested."""
        return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def elapsed(self) -> float:
        """Seconds this camera has been capturing, used for its honest average rate."""
        return max(time.monotonic() - self._started, 1e-9)

    @property
    def average_fps(self) -> float:
        """Frames delivered per second over this camera's whole run."""
        return self.frames / self.elapsed

    @property
    def fourcc(self) -> str:
        """The pixel format the camera actually settled on, which drives how much bus bandwidth it reserves."""
        code = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        if code <= 0:
            return "?"
        return "".join(chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)).strip()

    def _read_loop(self) -> None:
        """Continuously pull frames, keeping only the newest and tracking this camera's own capture rate."""
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                self.failures += 1
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
            self.frames += 1

            now = time.monotonic()
            if self._last_time is not None:
                elapsed = now - self._last_time
                if elapsed > 0:
                    instant = 1.0 / elapsed
                    self.fps = instant if self.fps == 0 else self.fps * FPS_SMOOTHING + instant * (1 - FPS_SMOOTHING)
            self._last_time = now

    def latest(self) -> np.ndarray | None:
        """Return the most recently captured frame, or None if this camera has not delivered one."""
        with self._lock:
            return self._frame

    def release(self) -> None:
        """Stop the reader thread and release the capture device."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.capture.release()


def sharpness(frame: np.ndarray) -> float:
    """Measure how sharply focused the centre of a frame is, for finding a camera's in-focus range."""
    height, width = frame.shape[:2]
    centre = frame[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    grey = cv2.cvtColor(centre, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def tile_size(width: int) -> tuple[int, int]:
    """Return the fixed (width, height) every tile is rendered at, so the grid can be stacked."""
    return (width, int(width * TILE_ASPECT))


def fit_into(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox a frame into a fixed-size tile, returning it with the scale and offset used.

    Cameras on one machine routinely differ in aspect ratio, so scaling each
    frame to a common width alone leaves tiles of different heights that cannot
    be stacked into a grid. Padding to a fixed size keeps the grid uniform
    without stretching any camera's image.
    """
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))))
    canvas = np.full((height, width, 3), 25, np.uint8)
    offset_x, offset_y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return canvas, scale, (offset_x, offset_y)


def make_tile(stream: Stream, frame: np.ndarray | None, width: int, faces: list | None) -> np.ndarray:
    """Render one camera's latest frame into a fixed-size tile, with its status drawn on top."""
    width, height = tile_size(width)
    if frame is None:
        tile = np.full((height, width, 3), 40, np.uint8)
        cv2.putText(tile, f"cam {stream.index}: NO FRAMES", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(tile, f"{stream.failures} failed reads", (10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return tile

    tile, scale, (offset_x, offset_y) = fit_into(frame, width, height)
    if faces:
        for x, y, w, h in faces:
            top_left = (int(x * scale) + offset_x, int(y * scale) + offset_y)
            bottom_right = (int((x + w) * scale) + offset_x, int((y + h) * scale) + offset_y)
            cv2.rectangle(tile, top_left, bottom_right, (0, 255, 0), 2)

    native = stream.resolution
    healthy = stream.fps >= HEALTHY_FPS
    focus = sharpness(frame)
    label = f"cam {stream.index}  {native[0]}x{native[1]}  {stream.fps:4.1f} fps"
    if faces is not None:
        label += f"  faces={len(faces)}"
    cv2.putText(tile, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if healthy else (0, 165, 255), 2)
    focus_colour = (0, 255, 0) if focus >= SHARP_ENOUGH else (0, 165, 255)
    cv2.putText(tile, f"sharpness {focus:6.0f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_colour, 2)
    if stream.failures:
        cv2.putText(tile, f"failed reads: {stream.failures}", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return tile


def mosaic(tiles: list[np.ndarray]) -> np.ndarray:
    """Arrange tiles into as square a grid as their count allows, padding any empty cell."""
    columns = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / columns))
    height, width = tiles[0].shape[:2]
    blank = np.full((height, width, 3), 25, np.uint8)
    padded = tiles + [blank] * (rows * columns - len(tiles))
    return np.vstack([np.hstack(padded[row * columns : (row + 1) * columns]) for row in range(rows)])


def summarize(streams: list[Stream], display_fps: float, elapsed: float) -> None:
    """Print each camera's outcome, separating a camera that never delivered from one the bus cannot keep up with."""
    print(f"\nRan {elapsed:.0f}s; display loop at {display_fps:.1f} fps. Per-camera capture rates:")
    if elapsed < MIN_SAMPLE_SECONDS:
        print(f"  WARNING: only {elapsed:.1f}s of data -- let it run at least {MIN_SAMPLE_SECONDS:.0f}s before trusting this.")
    working = [s for s in streams if s.frames > 0]
    slow = [s for s in working if s.average_fps < HEALTHY_FPS]

    for stream in streams:
        if not stream.opened:
            print(f"  camera {stream.index}: never opened")
            continue
        if stream.frames == 0:
            print(f"  camera {stream.index}: 0 frames, {stream.failures} failed reads -- opened but delivered nothing")
            continue

        # The smoothed rate reacts to bursts of buffered or duplicate frames, so
        # the average over the whole run is the one to trust; a large gap between
        # them means the camera is delivering unevenly rather than steadily.
        average = stream.average_fps
        verdict = "ok" if average >= HEALTHY_FPS else "slow"
        if stream.fps > average * 2:
            verdict += " (bursty -- recent rate spiked, likely duplicate frames)"
        print(f"  camera {stream.index}: {stream.frames} frames, {stream.failures} failed reads, {average:.1f} fps average ({stream.fps:.1f} recent) -- {verdict}")

    print()
    if not working:
        print("No camera delivered frames.")
    elif not slow:
        print(f"All {len(working)} camera(s) captured at a healthy rate simultaneously.")
    elif len(slow) == len(working):
        print(f"Every camera is slow ({len(slow)}/{len(working)}) -- consistent with the bus being saturated.")
        print("Try a lower resolution: --width 1280 --height 720, or --width 640 --height 480")
    else:
        print(f"{len(slow)} of {len(working)} camera(s) are slow while the rest are fine.")
        print("That points at those specific cameras or their port, not overall bus bandwidth.")


def main() -> None:
    """Open every requested camera at once, each on its own reader thread, and display them tiled."""
    args = parse_args()

    indices = [int(s) for s in args.source] if args.source else probe_cameras()
    if not indices:
        raise SystemExit("No cameras detected.")
    print(f"Opening cameras {indices} simultaneously...")

    detector = None
    if args.detect:
        from src.retailvision.detection import FaceDetector

        print("Loading face detector...")
        detector = FaceDetector()

    streams = [Stream(index, args.width, args.height, args.fourcc, args.fps) for index in indices]
    time.sleep(1.0)  # let each reader thread deliver a first frame before reporting
    for stream in streams:
        state = f"opened at {stream.resolution[0]}x{stream.resolution[1]} {stream.fourcc}" if stream.opened else "FAILED TO OPEN"
        print(f"  camera {stream.index}: {state}")

    live = [s for s in streams if s.opened]
    if not live:
        raise SystemExit("No cameras could be opened simultaneously.")
    print(f"\n{len(live)}/{len(streams)} opened. Press 'q' to quit.\n")

    display_frames, started = 0, time.monotonic()
    try:
        while True:
            tiles = []
            for stream in live:
                frame = stream.latest()
                faces = detector.detect(frame) if detector and frame is not None else None
                tiles.append(make_tile(stream, frame, args.tile_width, faces))

            cv2.imshow("Camera check", mosaic(tiles))
            display_frames += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        elapsed = max(time.monotonic() - started, 1e-9)
        for stream in streams:
            stream.release()
        cv2.destroyAllWindows()
        summarize(streams, display_frames / elapsed, elapsed)


if __name__ == "__main__":
    main()
