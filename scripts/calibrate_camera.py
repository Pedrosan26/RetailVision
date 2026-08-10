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
from pathlib import Path

import cv2
import numpy as np

from src.retailvision.calibration import calibrate, find_chessboard_corners, view_reprojection_errors

DEFAULT_PATTERN = "9x6"
DEFAULT_MIN_SAMPLES = 15

# A view is treated as an outlier when it fits several times worse than the
# typical view, but never when it is already accurate in absolute terms.
OUTLIER_RATIO = 3.0
OUTLIER_FLOOR_PX = 1.0
MIN_VIEWS_AFTER_DROP = 8


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
    return parser.parse_args()


def main() -> None:
    """Capture chessboard views interactively, then solve and save the camera's intrinsic calibration."""
    args = parse_args()
    pattern = parse_pattern(args.pattern)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera at source: {args.source}")

    corner_sets: list = []
    image_size: tuple[int, int] | None = None
    print(f"Looking for a {pattern[0]}x{pattern[1]} inner-corner chessboard. SPACE captures, 'c' calibrates, 'q' quits.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
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
            cv2.imshow("Camera calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
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

    if len(corner_sets) < args.min_samples:
        raise SystemExit(f"Only {len(corner_sets)} views captured, need at least {args.min_samples}")
    if image_size is None:
        raise SystemExit("No frames were read from the camera")

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
    print(f"Optical center off image center by {offset_x:.0f}px, {offset_y:.0f}px (a large offset means the views did not cover the frame)")
    print(f"Largest tangential distortion term: {tangential:.4f} (above ~0.01 usually means overfitting)")

    errors = view_reprojection_errors(corner_sets, pattern, calibration, square_size)
    ranked = sorted(enumerate(errors, start=1), key=lambda pair: pair[1], reverse=True)
    print("Worst-fitting views: " + ", ".join(f"#{index} ({error:.2f}px)" for index, error in ranked[:5]))

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
