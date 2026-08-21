"""
test_marker_map.py

Unit tests for the shared marker map and the localizer that populates it:
that a camera can be placed from a single mapped marker, that a second
camera sharing only one marker with the first still lands in the same
world frame, that the map grows to cover markers no single camera ever saw
together, and that the two-fold orientation ambiguity of a lone square
marker is resolved rather than picked at random. These are the claims the
whole multi-camera zone design rests on, so they run against synthetic
views rendered from hand-built ground-truth poses.
"""

import unittest

import cv2
import numpy as np

from src.retailvision.calibration import CameraCalibration
from src.retailvision.marker_map import CameraLocalizer, MarkerMap, project_to_plane
from src.retailvision.marker_pose import MarkerPoseEstimator, marker_object_points
from tests.test_marker_pose import synthetic_calibration

MARKER_SIZE = 0.20
QUIET_ZONE_RATIO = 0.25
# A wider view than the pose tests use, so a camera set back far enough to see
# two floor markers at once still frames both of them.
IMAGE_SIZE = (1280, 960)


def wide_calibration() -> CameraCalibration:
    """Distortion-free calibration for the wide synthetic camera these map tests render through."""
    width, height = IMAGE_SIZE
    camera_matrix = np.array([[900.0, 0.0, width / 2], [0.0, 900.0, height / 2], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        camera_matrix=camera_matrix, dist_coeffs=np.zeros((1, 5)), image_size=IMAGE_SIZE, reprojection_error=0.0
    )

# Four markers lying flat on the floor at the corners of a 2m x 3m area.
FLOOR_MARKERS = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (2.0, 3.0), 3: (0.0, 3.0)}


def marker_at(x: float, y: float) -> np.ndarray:
    """World pose of a marker lying flat on the floor at (x, y), facing up."""
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, 0.0]
    return matrix


def camera_at(eye: tuple[float, float, float], target: tuple[float, float, float]) -> np.ndarray:
    """World pose of a camera at eye looking towards target, with z forward and y down."""
    eye_array, target_array = np.array(eye, float), np.array(target, float)
    forward = target_array - eye_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, down, forward])
    matrix[:3, 3] = eye_array
    return matrix


def marker_image(marker_id: int, pixels: int = 240) -> np.ndarray:
    """Render one marker's printable image, framed by the same white quiet zone as the real printouts."""
    image = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, pixels)
    margin = int(pixels * QUIET_ZONE_RATIO)
    framed = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(framed, cv2.COLOR_GRAY2BGR)


def render(camera_pose: np.ndarray, marker_ids: list[int], calibration) -> np.ndarray:
    """Render the given floor markers as a camera at camera_pose would see them."""
    width, height = IMAGE_SIZE
    frame = np.full((height, width, 3), 110, np.uint8)
    camera_from_world = np.linalg.inv(camera_pose)

    for marker_id in marker_ids:
        image = marker_image(marker_id)
        side = image.shape[0]
        relative = camera_from_world @ marker_at(*FLOOR_MARKERS[marker_id])
        rvec, _ = cv2.Rodrigues(relative[:3, :3])
        outer = marker_object_points(MARKER_SIZE * (1 + 2 * QUIET_ZONE_RATIO))
        projected, _ = cv2.projectPoints(
            outer, rvec, relative[:3, 3], calibration.camera_matrix, calibration.dist_coeffs
        )
        source = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
        transform = cv2.getPerspectiveTransform(source, projected.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(image, transform, (width, height))
        mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), transform, (width, height))
        frame[mask > 0] = warped[mask > 0]
    return frame


class TestMarkerMapData(unittest.TestCase):
    """Verify the map's plain accessors, independent of any camera."""

    def test_unmapped_marker_has_no_position(self) -> None:
        """Asking for a marker that was never observed returns nothing rather than guessing."""
        marker_map = MarkerMap()
        self.assertIsNone(marker_map.position(11))
        self.assertIsNone(marker_map.floor_position(11))
        self.assertIsNone(marker_map.normal(11))
        self.assertIsNone(marker_map.world_corners(11, MARKER_SIZE))
        self.assertNotIn(11, marker_map)

    def test_floor_position_drops_the_height_component(self) -> None:
        """A mapped marker's floor position is its world x/y."""
        marker_map = MarkerMap()
        marker_map.poses[4] = marker_at(1.25, -0.75)
        self.assertEqual(marker_map.floor_position(4), (1.25, -0.75))

    def test_world_corners_span_the_printed_size_around_the_marker(self) -> None:
        """A mapped marker's world corners form its physical square at its mapped position."""
        marker_map = MarkerMap()
        marker_map.poses[0] = marker_at(1.0, 1.0)
        corners = marker_map.world_corners(0, MARKER_SIZE)
        self.assertEqual(corners.shape, (4, 3))
        np.testing.assert_allclose(corners.mean(axis=0), [1.0, 1.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(corners[0] - corners[1])), MARKER_SIZE, places=6)

    def test_clear_forgets_every_marker(self) -> None:
        """Resetting the map drops all mapped markers so the next observation re-anchors it."""
        marker_map = MarkerMap()
        marker_map.poses[0] = marker_at(0.0, 0.0)
        marker_map.clear()
        self.assertEqual(marker_map.marker_ids, [])


class TestCameraLocalizer(unittest.TestCase):
    """Verify camera placement, ambiguity resolution and cross-camera map growth from rendered views."""

    def setUp(self) -> None:
        """Build the shared calibration and estimator used to render and solve every view here."""
        self.calibration = wide_calibration()
        self.estimator = MarkerPoseEstimator(self.calibration, MARKER_SIZE)

    def localizer(self, marker_map: MarkerMap) -> CameraLocalizer:
        """Build a localizer for this test's camera against the given map."""
        return CameraLocalizer(self.estimator, marker_map)

    def test_first_view_anchors_the_world_origin(self) -> None:
        """With no anchor configured, the largest marker in the first view becomes the world origin."""
        marker_map = MarkerMap()
        camera = camera_at((1.0, -1.2, 2.0), (1.0, 0.4, 0.0))
        self.localizer(marker_map).update(render(camera, [0, 1], self.calibration))
        self.assertIn(marker_map.anchor_id, (0, 1))
        np.testing.assert_allclose(marker_map.position(marker_map.anchor_id), [0.0, 0.0, 0.0], atol=1e-9)

    def test_configured_anchor_defines_the_origin(self) -> None:
        """An explicitly configured anchor sits at the origin and other markers are placed relative to it."""
        marker_map = MarkerMap(anchor_id=0)
        camera = camera_at((1.0, -1.2, 2.0), (1.0, 0.4, 0.0))
        self.localizer(marker_map).update(render(camera, [0, 1], self.calibration))
        np.testing.assert_allclose(marker_map.position(0), [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(marker_map.floor_position(1), FLOOR_MARKERS[1], atol=0.05)

    def test_camera_pose_recovers_where_the_camera_stands(self) -> None:
        """A camera is placed in the world frame close to where it physically is."""
        marker_map = MarkerMap(anchor_id=0)
        camera = camera_at((1.0, -1.2, 2.0), (1.0, 0.4, 0.0))
        result = self.localizer(marker_map).update(render(camera, [0, 1], self.calibration))
        self.assertIsNotNone(result.camera_pose)
        self.assertLess(float(np.linalg.norm(result.camera_pose[:3, 3] - camera[:3, 3])), 0.10)

    def test_no_markers_visible_yields_no_localization(self) -> None:
        """An empty view localizes nothing and reports no markers."""
        marker_map = MarkerMap(anchor_id=0)
        width, height = IMAGE_SIZE
        result = self.localizer(marker_map).update(np.full((height, width, 3), 110, np.uint8))
        self.assertIsNone(result.camera_pose)
        self.assertEqual(result.marker_poses, {})
        self.assertEqual(result.learned, [])

    def test_view_without_the_configured_anchor_maps_nothing(self) -> None:
        """Until the configured anchor is seen there is no world frame to place anything in."""
        marker_map = MarkerMap(anchor_id=9)
        camera = camera_at((1.0, -1.2, 2.0), (1.0, 0.4, 0.0))
        result = self.localizer(marker_map).update(render(camera, [0, 1], self.calibration))
        self.assertIsNone(result.camera_pose)
        self.assertEqual(marker_map.marker_ids, [])

    def test_markers_are_mapped_facing_the_same_way_as_the_anchor(self) -> None:
        """The orientation ambiguity is resolved towards the plane the markers share, never flipped through it."""
        marker_map = MarkerMap(anchor_id=0)
        camera = camera_at((1.0, -1.2, 2.2), (1.0, 0.6, 0.0))
        self.localizer(marker_map).update(render(camera, [0, 1], self.calibration))
        for marker_id in marker_map.marker_ids:
            self.assertGreater(float(marker_map.normal(marker_id)[2]), 0.9)

    def test_second_camera_sharing_one_marker_extends_the_same_world_frame(self) -> None:
        """Two cameras overlapping on a single marker map corners neither of them saw together."""
        marker_map = MarkerMap(anchor_id=0)
        camera_a = camera_at((1.0, -1.5, 2.2), (1.0, 0.2, 0.0))
        camera_b = camera_at((1.0, 4.5, 2.2), (1.2, 1.6, 0.0))

        self.localizer(marker_map).update(render(camera_a, [0, 1], self.calibration))
        self.assertEqual(marker_map.marker_ids, [0, 1])

        result = self.localizer(marker_map).update(render(camera_b, [1, 2, 3], self.calibration))
        self.assertEqual(result.learned, [2, 3])
        self.assertEqual(marker_map.marker_ids, [0, 1, 2, 3])

        for marker_id, expected in FLOOR_MARKERS.items():
            got = marker_map.floor_position(marker_id)
            self.assertLess(float(np.hypot(got[0] - expected[0], got[1] - expected[1])), 0.25)
            self.assertGreater(float(marker_map.normal(marker_id)[2]), 0.9)

    def test_already_mapped_markers_are_not_relearned(self) -> None:
        """A marker's world pose is fixed on first mapping, so a repeat view reports nothing new."""
        marker_map = MarkerMap(anchor_id=0)
        camera = camera_at((1.0, -1.2, 2.0), (1.0, 0.4, 0.0))
        localizer = self.localizer(marker_map)
        frame = render(camera, [0, 1], self.calibration)
        localizer.update(frame)
        self.assertEqual(localizer.update(frame).learned, [])

    def test_localization_reports_a_small_reprojection_error(self) -> None:
        """A healthy fit projects the mapped markers back onto their detected corners within about a pixel."""
        marker_map = MarkerMap(anchor_id=0)
        camera = camera_at((1.0, -1.5, 2.2), (1.0, 0.2, 0.0))
        result = self.localizer(marker_map).update(render(camera, [0, 1], self.calibration))
        self.assertLess(result.reprojection_error, 1.0)


class TestProjectToPlane(unittest.TestCase):
    """Verify back-projection of a pixel onto a horizontal world plane."""

    def setUp(self) -> None:
        """Place a camera 2m up looking straight down, so expected intersections are exact by hand."""
        self.calibration = synthetic_calibration()
        self.camera_pose = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
        )

    def test_principal_point_lands_directly_below_the_camera(self) -> None:
        """The image center back-projects to the point on the plane straight under the camera."""
        world = project_to_plane((320.0, 240.0), self.camera_pose, self.calibration)
        np.testing.assert_allclose(world, (0.0, 0.0), atol=1e-9)

    def test_offset_pixel_scales_with_height_above_the_plane(self) -> None:
        """A pixel one focal length off center lands one camera-height away on the plane."""
        world = project_to_plane((1120.0, 240.0), self.camera_pose, self.calibration)
        np.testing.assert_allclose(world, (2.0, 0.0), atol=1e-9)

    def test_plane_above_the_camera_is_unreachable(self) -> None:
        """A downward-looking camera never meets a plane above it, so no position is reported."""
        self.assertIsNone(project_to_plane((320.0, 240.0), self.camera_pose, self.calibration, plane_z=5.0))

    def test_ray_parallel_to_the_plane_has_no_intersection(self) -> None:
        """A camera looking along the plane produces a ray that never meets it."""
        horizontal = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertIsNone(project_to_plane((320.0, 240.0), horizontal, self.calibration))


if __name__ == "__main__":
    unittest.main()


class TestMarkerMapPersistence(unittest.TestCase):
    """Verify a surveyed map round-trips to disk and is authoritative once loaded."""

    def build(self) -> MarkerMap:
        """A small hand-built map with a non-default mounting and anchor height."""
        marker_map = MarkerMap(anchor_id=3, mounting="wall", anchor_height=1.7)
        for marker_id, (x, y) in FLOOR_MARKERS.items():
            matrix = np.eye(4)
            matrix[:3, 3] = [x, y, 1.7]
            marker_map.poses[marker_id] = matrix
        return marker_map

    def test_round_trips_through_disk(self) -> None:
        """Saving and loading preserves every marker pose and the world-frame settings."""
        import tempfile
        from pathlib import Path

        original = self.build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            original.save(path)
            loaded = MarkerMap.load(path)

        self.assertEqual(loaded.marker_ids, original.marker_ids)
        self.assertEqual(loaded.anchor_id, original.anchor_id)
        self.assertEqual(loaded.mounting, original.mounting)
        self.assertEqual(loaded.anchor_height, original.anchor_height)
        for marker_id in original.poses:
            np.testing.assert_allclose(loaded.poses[marker_id], original.poses[marker_id])

    def test_a_loaded_map_is_frozen(self) -> None:
        """Nodes load a survey rather than extending it, so their world frames cannot drift apart."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            self.build().save(path)
            loaded = MarkerMap.load(path)

        self.assertTrue(loaded.frozen)
        calibration = wide_calibration()
        estimator = MarkerPoseEstimator(calibration, MARKER_SIZE, allowed_ids=set(FLOOR_MARKERS))
        before = list(loaded.marker_ids)
        CameraLocalizer(estimator, loaded).update(
            render(camera_at((1.0, -1.5, 2.2), (1.0, 0.4, 0.0)), [0, 1], calibration)
        )
        self.assertEqual(loaded.marker_ids, before)

    def test_a_freshly_built_map_is_not_frozen(self) -> None:
        """A map being surveyed must still be able to grow."""
        self.assertFalse(MarkerMap().frozen)
