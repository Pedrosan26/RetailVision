"""
calibration.py

Camera intrinsic calibration: the per-camera lens model (focal length,
optical center, distortion coefficients) that pose estimation needs in
order to turn a marker's apparent size and shape in the image into a real
3D distance and orientation.

Intrinsics are a property of the physical camera and lens, not of the room
it is pointed at -- calibrate a camera once and the result stays valid for
every deployment, so this cost is paid per camera, not per zone. That is
what makes marker pose estimation practical here: no measuring of the room
or of distances between markers is ever required, only the marker's own
printed size.

Calibration itself is a one-off capture of a chessboard pattern held at
many angles (see scripts/calibrate_camera.py); this module holds the
resulting model, the solver that produces it, and the chessboard geometry
helpers that feed it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Sub-pixel corner refinement settings: chessboard corners found at whole-pixel
# resolution are too coarse for a stable lens model, so each is refined until it
# moves less than this epsilon or the iteration cap is hit.
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CORNER_WINDOW = (11, 11)

# The third radial coefficient is rarely identifiable from a handheld calibration
# session, and left free it reliably absorbs noise as an enormous value that
# cancels within the captured views and diverges outside them. Such a model keeps
# a low reprojection error while being physically absurd: solvePnP undistorts and
# projectPoints re-distorts, and the two stop agreeing, so a pose can fit its own
# corners and still land hundreds of pixels away when projected back. Fixing k3
# at zero costs nothing for ordinary lenses and removes the failure entirely.
# Distortion models are tried richest-first and the first self-consistent one is
# kept. A handheld session rarely constrains the higher radial terms, and left
# free they absorb noise as huge values that cancel within the captured views and
# diverge towards the frame corners -- precisely where a marker most needs to be
# trusted. Dropping a term the data cannot support loses nothing real.
CALIBRATION_FLAG_LADDER = (
    ("k1+k2, tangential", cv2.CALIB_FIX_K3),
    ("k1 only, tangential", cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2),
    ("k1 only, no tangential", cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2 | cv2.CALIB_ZERO_TANGENT_DIST),
)

# Above this the undistort/distort round trip disagrees enough that solved poses
# will not reproject correctly, whatever the reprojection error says.
MAX_ROUND_TRIP_PX = 1.0


@dataclass(frozen=True)
class CameraCalibration:
    """A single camera's intrinsic lens model, loadable from and savable to disk."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float

    def save(self, path: str | Path) -> None:
        """Write the calibration to a JSON file, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "image_size": list(self.image_size),
            "reprojection_error": self.reprojection_error,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibration:
        """Read a calibration back from a JSON file written by save()."""
        payload = json.loads(Path(path).read_text())
        return cls(
            camera_matrix=np.array(payload["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(payload["dist_coeffs"], dtype=np.float64).reshape(1, -1),
            image_size=tuple(payload["image_size"]),
            reprojection_error=float(payload["reprojection_error"]),
        )


def chessboard_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    """Build the flat 3D grid of a chessboard's inner corners, in the board's own coordinate frame."""
    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points * square_size


def find_chessboard_corners(frame: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Locate and sub-pixel refine a chessboard's inner corners in a frame, or return None if not found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, CORNER_WINDOW, (-1, -1), CORNER_CRITERIA)


def view_reprojection_errors(
    corner_sets: list[np.ndarray],
    pattern_size: tuple[int, int],
    calibration: CameraCalibration,
    square_size: float = 0.025,
) -> list[float]:
    """Score each captured view separately against a calibration, to find which ones are dragging the fit down.

    The overall error is an average, so it cannot distinguish a few badly
    detected or motion-blurred boards from a uniformly mediocre session. Those
    two cases need opposite fixes -- drop the bad views, or recapture
    everything -- so it is worth being able to tell them apart.
    """
    object_points = chessboard_object_points(pattern_size, square_size)
    errors = []
    for corners in corner_sets:
        _, rvec, tvec = cv2.solvePnP(object_points, corners, calibration.camera_matrix, calibration.dist_coeffs)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, calibration.camera_matrix, calibration.dist_coeffs
        )
        deviations = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        errors.append(float(deviations.mean()))
    return errors


def round_trip_error(calibration: CameraCalibration) -> float:
    """Measure whether the lens model is self-consistent, sampled densely across the whole frame.

    A calibration can fit its own capture session well and still be unusable,
    because undistortion and distortion are separate operations that only agree
    when the coefficients are physically sensible. Sampling only the middle of
    the frame hides the problem: an overfitted radial model is typically well
    behaved near the optical centre and diverges towards the corners, which is
    where a marker most needs to be trusted. So the grid spans the frame, and the
    score is the 90th percentile rather than the median, which a mostly-fine
    model would otherwise flatter.
    """
    width, height = calibration.image_size
    xs = np.linspace(width * 0.05, width * 0.95, 25)
    ys = np.linspace(height * 0.05, height * 0.95, 25)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64).reshape(-1, 1, 2)

    normalized = cv2.undistortPoints(grid, calibration.camera_matrix, calibration.dist_coeffs)
    rays = np.hstack([normalized.reshape(-1, 2), np.ones((len(normalized), 1))])
    back, _ = cv2.projectPoints(
        rays, np.zeros(3), np.zeros(3), calibration.camera_matrix, calibration.dist_coeffs
    )
    deviations = np.linalg.norm(back.reshape(-1, 2) - grid.reshape(-1, 2), axis=1)
    return float(np.percentile(deviations, 90))


def _solve(corner_sets, pattern_size, image_size, square_size, flags) -> CameraCalibration:
    """Run the calibration solver once with a given distortion model."""
    object_points = [chessboard_object_points(pattern_size, square_size)] * len(corner_sets)
    error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, corner_sets, image_size, None, None, flags=flags
    )
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_error=float(error),
    )


def calibrate(
    corner_sets: list[np.ndarray],
    pattern_size: tuple[int, int],
    image_size: tuple[int, int],
    square_size: float = 0.025,
) -> CameraCalibration:
    """Solve for the camera's intrinsics, simplifying the distortion model until the result is self-consistent."""
    if not corner_sets:
        raise ValueError("Need at least one chessboard observation to calibrate")

    fallback = None
    for label, flags in CALIBRATION_FLAG_LADDER:
        calibration = _solve(corner_sets, pattern_size, image_size, square_size, flags)
        deviation = round_trip_error(calibration)
        if deviation <= MAX_ROUND_TRIP_PX:
            if fallback is not None:
                print(f"Distortion model simplified to '{label}' -- richer models did not round-trip.")
            return calibration
        print(f"Distortion model '{label}' is not self-consistent ({deviation:.1f}px); trying a simpler one.")
        fallback = calibration
    return fallback


