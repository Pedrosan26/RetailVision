"""
marker_pose.py

Per-marker 3D pose estimation: given a calibrated camera and markers all
printed at one known physical size, recovers how far away each visible
marker is and how it is oriented, from that single marker alone.

This is what replaces the earlier homography approach. A homography maps
image pixels to a flat zone plane, but is only solvable from four point
correspondences at once, so a camera had to see all four of a zone's
corner markers simultaneously -- unworkable in a real room where no single
viewpoint covers every corner. Pose estimation instead solves each marker
independently, because the marker's four corners plus its known printed
size are already a complete 3D-to-2D correspondence set on their own. One
visible marker is enough, which is what allows several cameras to
collectively cover a zone that none of them can see in full.

The pose returned places the marker in the camera's own 3D frame; turning
several such per-camera readings into one shared frame is marker_map.py's
job.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CameraCalibration

# DICT_4X4_50 matches scripts/generate_aruco_markers.py: a small, sparse pattern
# that stays detectable further from the camera than a denser dictionary would.
DEFAULT_DICTIONARY = cv2.aruco.DICT_4X4_50


@dataclass(frozen=True)
class MarkerPose:
    """One detected marker's 3D position and orientation relative to the camera that saw it."""

    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    reprojection_error: float = 0.0

    @property
    def distance_meters(self) -> float:
        """Straight-line distance from the camera to the marker's center."""
        return float(np.linalg.norm(self.tvec))

    @property
    def pixel_area(self) -> float:
        """Apparent area of the marker in the image, a proxy for how reliable this pose reading is."""
        return float(abs(cv2.contourArea(self.corners.astype(np.float32))))

    @property
    def matrix(self) -> np.ndarray:
        """The 4x4 transform taking points from this marker's frame into the observing camera's frame."""
        return pose_matrix(self.rvec, self.tvec)


def _detector_parameters() -> cv2.aruco.DetectorParameters:
    """Build detector parameters tuned for pose estimation rather than plain detection."""
    parameters = cv2.aruco.DetectorParameters()
    # The detector defaults to whole-pixel corners, which is ample for reading a
    # marker's ID but not for solving its pose: a marker several meters away spans
    # only tens of pixels, so a one-pixel corner error becomes centimeters of
    # position error. Sub-pixel refinement is the difference between a usable and
    # an unusable pose at realistic ceiling-camera distances.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return parameters


def pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Combine a rotation vector and translation vector into a single 4x4 homogeneous transform."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3], _ = cv2.Rodrigues(rvec)
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return matrix


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform, reversing which frame it maps from and to."""
    rotation = matrix[:3, :3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverted


def marker_object_points(marker_size: float) -> np.ndarray:
    """Build a marker's four corners in its own frame, ordered to match cv2.aruco's detected corner order."""
    half = marker_size / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )


class MarkerPoseEstimator:
    """Detects ArUco markers in a frame and solves each one's 3D pose relative to the camera."""

    def __init__(
        self,
        calibration: CameraCalibration,
        marker_size: float,
        dictionary: int = DEFAULT_DICTIONARY,
        allowed_ids: set[int] | None = None,
    ) -> None:
        """Bind a calibrated camera to the marker size every marker is printed at, and optionally to the IDs in use."""
        if marker_size <= 0:
            raise ValueError("marker_size must be a positive length in meters")
        self.calibration = calibration
        self.marker_size = marker_size
        # A 4x4 marker carries little redundancy, so noise, motion blur and
        # incidental rectangles in a scene are regularly decoded as valid IDs that
        # were never printed. Such a phantom is indistinguishable from a real
        # marker once mapped, and because poses are first-observation-wins it
        # corrupts the frame permanently. Restricting detection to the IDs
        # actually deployed removes the whole failure mode.
        self.allowed_ids = allowed_ids
        self._object_points = marker_object_points(marker_size)
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary), _detector_parameters()
        )

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """Return {marker_id: 4x2 pixel corners} for every marker visible in the frame."""
        corners, ids, _ = self._detector.detectMarkers(frame)
        if ids is None:
            return {}
        detected = {int(marker_id): c[0] for c, marker_id in zip(corners, ids.flatten())}
        if self.allowed_ids is None:
            return detected
        return {marker_id: c for marker_id, c in detected.items() if marker_id in self.allowed_ids}

    def estimate(self, frame: np.ndarray) -> dict[int, MarkerPose]:
        """Detect every visible marker and return each one's best-fitting pose, keyed by marker ID."""
        return {marker_id: poses[0] for marker_id, poses in self.estimate_candidates(frame).items()}

    def estimate_candidates(self, frame: np.ndarray) -> dict[int, list[MarkerPose]]:
        """Detect every visible marker and return all of its candidate poses, best reprojection first."""
        return {
            marker_id: self.solve_candidates(marker_id, corners)
            for marker_id, corners in self.detect(frame).items()
        }

    def solve(self, marker_id: int, corners: np.ndarray) -> MarkerPose:
        """Solve one marker's pose from its pixel corners, taking the lowest-reprojection candidate."""
        return self.solve_candidates(marker_id, corners)[0]

    def solve_candidates(self, marker_id: int, corners: np.ndarray) -> list[MarkerPose]:
        """Return every pose consistent with a marker's pixel corners, ordered by reprojection error.

        A flat square viewed through a camera is genuinely ambiguous: two
        different orientations, roughly mirrored through the marker's plane,
        project to almost the same four pixels. The solver returns both, and
        their reprojection errors can be near-identical, so picking the lowest
        one alone will sometimes choose an orientation flipped by more than 90
        degrees while the position stays about right. Callers that have outside
        information -- other markers in view, a known mounting plane, the
        previous frame's pose -- must use it to choose between these; see
        marker_map.CameraLocalizer.
        """
        _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            self._object_points,
            corners.astype(np.float32),
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        candidates = [
            MarkerPose(
                marker_id=marker_id,
                rvec=rvec,
                tvec=tvec,
                corners=corners,
                reprojection_error=float(np.asarray(error).ravel()[0]),
            )
            for rvec, tvec, error in zip(rvecs, tvecs, errors)
        ]
        return sorted(candidates, key=lambda pose: pose.reprojection_error)
