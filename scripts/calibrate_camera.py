"""
calibrate_camera.py

One-off intrinsic calibration for a single physical camera, the
prerequisite for marker pose estimation. Shows a live view, captures a
chessboard pattern each time SPACE is pressed, and solves for the lens
model once enough views are collected.

Hold a printed chessboard at clearly different distances, tilts and
positions across the frame -- views that are all head-on and centered
constrain the lens model poorly no matter how many are captured. Include
some near the frame's corners, where distortion is strongest.

The result describes the camera itself, not the room, so it stays valid
across every zone and deployment that camera is used in. Rerun only if the
lens, resolution or zoom changes.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/calibrate_camera.py --source 0 --out calibration/camera_0.json
SPACE captures a view, 'c' calibrates and saves, 'q' quits.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.retailvision.calibration import calibrate, find_chessboard_corners, round_trip_error, view_reprojection_errors

DEFAULT_PATTERN = "9x6"
DEFAULT_MIN_SAMPLES = 15

# A view is treated as an outlier when it fits several times worse than the
# typical view, but never when it is already accurate in absolute terms.
OUTLIER_RATIO = 3.0
OUTLIER_FLOOR_PX = 1.0
WINDOW_NAME = "Camera calibration"

# Requesting a resolution makes the backend tear down and rebuild the capture
# session, and on macOS the camera can take the better part of ten seconds to
# deliver its first frame afterwards. Reading immediately looks exactly like a
# dead or busy camera, so wait the restart out before concluding anything.
WARMUP_TIMEOUT = 25.0
MIN_VIEWS_AFTER_DROP = 8

# A single dropped read is normal; a long unbroken run of them means the camera
# has stopped delivering, usually because another process already holds it.
MAX_CONSECUTIVE_READ_FAILURES = 30


def wait_for_first_frame(capture: cv2.VideoCapture, timeout: float) -> bool:
    """Block until the camera delivers a frame or the timeout expires, returning whether one arrived."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, _ = capture.read()
        if ok:
            return True
        time.sleep(0.1)
    return False


def parse_pattern(text: str) -> tuple[int, int]:
    """Parse a 'COLUMNSxROWS' inner-corner count, e.g. '9x6' for a 10x7 square chessboard."""
    columns, _, rows = text.partition("x")
    return (int(columns), int(rows))


def parse_args() -> argparse.Namespace:
    """Parse the camera source, output path, chessboard geometry, and minimum sample count."""
    parser = argparse.ArgumentParser(description="Calibrate a camera's intrinsics from a chessboard pattern")
    parser.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    parser.add_argument("--out", type=Path, default=Path("calibration/camera_0.json"), help="Where to write the calibration JSON")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"Inner corners as COLUMNSxROWS (default: {DEFAULT_PATTERN})")
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="Chessboard square size in meters; affects only reported scale, not the intrinsics (default: 0.025)",
    )
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help=f"Views required before calibrating (default: {DEFAULT_MIN_SAMPLES})")
    # Focal length is measured in pixels, so a calibration only describes the
    # camera at the resolution it was captured at. When several cameras have to
    # share a USB bus and are therefore run below their native resolution, they
    # must be calibrated at that same reduced resolution.
    parser.add_argument("--width", type=int, default=None, help="Capture at this width -- must match the width the camera will be run at")
    parser.add_argument("--height", type=int, default=None, help="Capture at this height -- must match the height the camera will be run at")
    return parser.parse_args()


def main() -> None:
    """Capture chessboard views interactively, then solve and save the camera's intrinsic calibration."""
    args = parse_args()
    pattern = parse_pattern(args.pattern)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera at source: {args.source}")
    if args.width is not None or args.height is not None:
        if args.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Waiting for camera {args.source} to restart at the requested resolution...")
        if not wait_for_first_frame(capture, WARMUP_TIMEOUT):
            capture.release()
            raise SystemExit(
                f"Camera {args.source} delivered no frame within {WARMUP_TIMEOUT:.0f}s of being set to "
                f"{args.width}x{args.height}.\nIt may not support that resolution, or may be held by "
                "another process (close any camera_check.py window)."
            )
    actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if (args.width, args.height) != (None, None) and actual != (args.width, args.height):
        print(f"Note: camera settled on {actual[0]}x{actual[1]}, not the requested {args.width}x{args.height}.")
    print(f"Calibrating at {actual[0]}x{actual[1]} -- run this camera at the same resolution.")

    corner_sets: list = []
    image_size: tuple[int, int] | None = None
    print(f"Looking for a {pattern[0]}x{pattern[1]} inner-corner chessboard. SPACE captures, 'c' calibrates, 'q' quits.")

    consecutive_failures = 0
    camera_failed = False
    quit_early = False
    # Create the window before the first frame arrives, so a camera that is slow
    # to start still gives visible feedback rather than an empty screen.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    camera_failed = True
                    break
                # Keep pumping the GUI while reads are failing, so the window stays
                # responsive and 'q' still works instead of the app looking hung.
                waiting = np.full((240, 640, 3), 40, np.uint8)
                cv2.putText(waiting, f"waiting for camera {args.source}...", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow(WINDOW_NAME, waiting)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    quit_early = True
                    break
                continue
            consecutive_failures = 0
            image_size = (frame.shape[1], frame.shape[0])

            corners = find_chessboard_corners(frame, pattern)
            display = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, pattern, corners, True)

            status = "board found" if corners is not None else "no board"
            color = (0, 255, 0) if corners is not None else (0, 0, 255)
            ready = len(corner_sets) >= args.min_samples
            cv2.putText(display, f"{status} | captured {len(corner_sets)}/{args.min_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(
                display,
                "SPACE captures | 'c' calibrates" if ready else f"SPACE captures | {args.min_samples - len(corner_sets)} more needed",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if ready else (0, 200, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                quit_early = True
                break
            if key == ord(" ") and corners is not None:
                corner_sets.append(corners)
                print(f"Captured view {len(corner_sets)}")
            if key == ord("c"):
                # Keep capturing rather than exiting on an early 'c'; throwing away
                # views already collected would mean starting the whole session over.
                if len(corner_sets) >= args.min_samples:
                    break
                print(f"Only {len(corner_sets)} views so far, need {args.min_samples}. Keep capturing.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if camera_failed:
        raise SystemExit(
            f"Camera {args.source} stopped delivering frames after {len(corner_sets)} view(s).\n"
            "Another process is probably holding it -- close any other camera_check.py or "
            "pipeline window, then try again."
        )
    if quit_early:
        raise SystemExit(f"Quit with {len(corner_sets)} view(s) captured; nothing was saved.")
    if image_size is None:
        raise SystemExit(
            f"Camera {args.source} never delivered a frame. It may be in use by another process, "
            "or unable to run at the requested resolution."
        )
    if len(corner_sets) < args.min_samples:
        raise SystemExit(f"Only {len(corner_sets)} views captured, need at least {args.min_samples}")

    calibration = calibrate(corner_sets, pattern, image_size, args.square_size)
    calibration, corner_sets = drop_outlier_views(calibration, corner_sets, pattern, image_size, args.square_size)
    report(calibration, corner_sets, pattern, args.square_size)

    calibration.save(args.out)
    print(f"\nWrote {args.out}")


def drop_outlier_views(
    calibration, corner_sets: list, pattern: tuple[int, int], image_size: tuple[int, int], square_size: float
):
    """Re-solve without any views that fit far worse than the rest, since one bad board corrupts the whole model."""
    errors = view_reprojection_errors(corner_sets, pattern, calibration, square_size)
    threshold = max(OUTLIER_FLOOR_PX, OUTLIER_RATIO * float(np.median(errors)))
    kept = [corners for corners, error in zip(corner_sets, errors) if error <= threshold]

    dropped = len(corner_sets) - len(kept)
    if dropped == 0 or len(kept) < MIN_VIEWS_AFTER_DROP:
        return calibration, corner_sets

    retried = calibrate(kept, pattern, image_size, square_size)
    if retried.reprojection_error >= calibration.reprojection_error:
        return calibration, corner_sets
    print(f"Dropped {dropped} view(s) fitting far worse than the rest; error {calibration.reprojection_error:.3f} -> {retried.reprojection_error:.3f} px")
    return retried, kept


def report(calibration, corner_sets: list, pattern: tuple[int, int], square_size: float) -> None:
    """Print the calibration's quality, the worst-fitting views, and what to do if it is not good enough."""
    width, height = calibration.image_size
    matrix = calibration.camera_matrix
    offset_x, offset_y = abs(matrix[0, 2] - width / 2), abs(matrix[1, 2] - height / 2)
    tangential = float(np.max(np.abs(calibration.dist_coeffs.ravel()[2:4])))

    print(f"\nReprojection error: {calibration.reprojection_error:.4f} px (below ~0.5 is good, above ~1.0 is not usable)")

    # Reprojection error only says the model fits the views it was solved from.
    # This says whether the model is self-consistent at all, which is a different
    # and more basic question -- and one a low reprojection error can hide.
    round_trip = round_trip_error(calibration)
    verdict = "ok" if round_trip < 1.0 else "BROKEN -- do not use this calibration"
    print(f"Lens-model self-consistency: {round_trip:.2f} px ({verdict})")
    print(f"Optical center off image center by {offset_x:.0f}px, {offset_y:.0f}px (a large offset means the views did not cover the frame)")
    print(f"Largest tangential distortion term: {tangential:.4f} (above ~0.01 usually means overfitting)")

    errors = view_reprojection_errors(corner_sets, pattern, calibration, square_size)
    ranked = sorted(enumerate(errors, start=1), key=lambda pair: pair[1], reverse=True)
    print("Worst-fitting views: " + ", ".join(f"#{index} ({error:.2f}px)" for index, error in ranked[:5]))

    if round_trip > 1.0:
        print("\nThe lens model does not round-trip: poses solved with it will not reproject correctly,")
        print("so distances and marker positions will be wrong even though the error above looks fine.")
        print("Recapture with more spread -- especially the four frame corners and the steep tilts.")
        return

    if calibration.reprojection_error > 1.0:
        spread = max(errors) / max(min(errors), 1e-9)
        if spread > 5:
            print("\nA few views fit far worse than the rest -- likely motion blur or a partly mis-detected board.")
            print("Recapture, holding still for a beat before each SPACE.")
        else:
            print("\nEvery view fits poorly, so this is not one bad capture.")
            print("Most likely: the board is not rigid (tape it to cardboard), or the views were too alike.")
            print("Vary tilt and position much more, and make sure the board reaches all four frame corners.")


if __name__ == "__main__":
    main()
