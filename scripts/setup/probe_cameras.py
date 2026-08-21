"""
probe_cameras.py

Throwaway helper: reports what resolution each attached camera actually
delivers, so a calibration can be captured at the same size the camera
will be run at.

For each index it prints the default negotiated resolution, the maximum
the backend will clamp to, and which common sizes are honored exactly --
a camera silently substituting a nearby size is the failure mode worth
catching before spending a calibration session on it.

Usage: ./venv/bin/python3 scripts/setup/probe_cameras.py
"""

import cv2

INDICES = range(4)
CANDIDATES = [(640, 480), (800, 600), (1280, 720), (1600, 1200), (1920, 1080)]


def actual_size(capture: cv2.VideoCapture) -> tuple[int, int]:
    """Return the resolution the camera is currently delivering, read back from a real frame."""
    ok, frame = capture.read()
    if ok:
        return (frame.shape[1], frame.shape[0])
    return (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def probe(index: int) -> None:
    """Print one camera's default, maximum, and exactly-supported resolutions."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(f"camera {index}: could not open")
        return

    default = actual_size(capture)

    # Backends clamp an over-large request down to the sensor's maximum, which
    # is the only portable way to discover that maximum.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    maximum = actual_size(capture)

    honored = []
    for width, height in CANDIDATES:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        got = actual_size(capture)
        honored.append(f"{width}x{height}->{got[0]}x{got[1]}{'' if got == (width, height) else ' (substituted)'}")

    capture.release()
    print(f"camera {index}: default {default[0]}x{default[1]} | max {maximum[0]}x{maximum[1]}")
    for line in honored:
        print(f"    {line}")


def main() -> None:
    """Probe every candidate camera index in turn."""
    for index in INDICES:
        probe(index)


if __name__ == "__main__":
    main()
