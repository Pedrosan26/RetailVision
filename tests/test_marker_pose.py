"""
test_marker_pose.py

Unit tests for marker pose estimation: the transform helpers, and a
round-trip check that a marker placed at a known 3D pose and projected
through a synthetic camera is recovered by the solver, which is what
justifies trusting a single marker to localize a camera.
"""

import unittest

import cv2
import numpy as np

from src.retailvision.calibration import CameraCalibration
from src.retailvision.marker_pose import (
    MarkerPoseEstimator,
    invert_pose,
    marker_object_points,
    pose_matrix,
)

MARKER_SIZE = 0.10


def synthetic_calibration() -> CameraCalibration:
    """Build a distortion-free calibration for a plausible 640x480 camera, for exact synthetic tests."""
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros((1, 5)),
        image_size=(640, 480),
        reprojection_error=0.0,
    )


class TestPoseHelpers(unittest.TestCase):
    """Verify the rotation/translation to 4x4 transform helpers."""

    def test_pose_matrix_places_translation_and_stays_homogeneous(self) -> None:
        """A pose matrix carries the translation in its last column and a homogeneous bottom row."""
        matrix = pose_matrix(np.zeros((3, 1)), np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])

    def test_invert_pose_round_trips_to_identity(self) -> None:
        """Composing a pose with its inverse yields the identity transform."""
        matrix = pose_matrix(np.array([[0.3], [-0.2], [0.1]]), np.array([[0.5], [-1.0], [2.0]]))
        np.testing.assert_allclose(matrix @ invert_pose(matrix), np.eye(4), atol=1e-12)

    def test_marker_object_points_span_the_printed_size(self) -> None:
        """The four object points form a flat square whose side equals the printed marker size."""
        points = marker_object_points(MARKER_SIZE)
        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose(points[:, 2], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(points[0] - points[1])), MARKER_SIZE, places=6)


class TestMarkerPoseEstimator(unittest.TestCase):
    """Verify a single marker's pose is recovered from its projected corners alone."""

    def setUp(self) -> None:
        """Bind an estimator to the synthetic camera used by every test here."""
        self.calibration = synthetic_calibration()
        self.estimator = MarkerPoseEstimator(self.calibration, MARKER_SIZE)

    def project(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project a marker at the given pose into synthetic pixel corners."""
        corners, _ = cv2.projectPoints(
            marker_object_points(MARKER_SIZE), rvec, tvec, self.calibration.camera_matrix, self.calibration.dist_coeffs
        )
        return corners.reshape(4, 2)

    def test_rejects_non_positive_marker_size(self) -> None:
        """A marker size of zero or less has no physical meaning and is refused up front."""
        with self.assertRaises(ValueError):
            MarkerPoseEstimator(self.calibration, 0.0)

    def test_recovers_known_pose_from_one_marker(self) -> None:
        """A marker projected from a known pose is solved back to that same rotation and translation."""
        rvec = np.array([[0.3], [-0.2], [0.1]])
        tvec = np.array([[0.05], [-0.10], [1.50]])
        pose = self.estimator.solve(7, self.project(rvec, tvec))
        self.assertEqual(pose.marker_id, 7)
        np.testing.assert_allclose(pose.rvec, rvec, atol=1e-4)
        np.testing.assert_allclose(pose.tvec, tvec, atol=1e-4)

    def test_reports_true_distance_from_a_single_marker(self) -> None:
        """Distance comes from the marker's known printed size, with no measurement of the scene."""
        tvec = np.array([[0.0], [0.0], [2.40]])
        pose = self.estimator.solve(0, self.project(np.zeros((3, 1)), tvec))
        self.assertAlmostEqual(pose.distance_meters, 2.40, places=3)

    def test_nearer_marker_has_larger_pixel_area(self) -> None:
        """Apparent area grows as a marker gets closer, so it ranks pose reliability sensibly."""
        near = self.estimator.solve(0, self.project(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]])))
        far = self.estimator.solve(0, self.project(np.zeros((3, 1)), np.array([[0.0], [0.0], [3.0]])))
        self.assertGreater(near.pixel_area, far.pixel_area)


if __name__ == "__main__":
    unittest.main()
