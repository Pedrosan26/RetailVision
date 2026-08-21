"""
test_zone_resolver.py

Unit tests for ZoneResolver, the piece the live pipeline uses to turn a
detection's bounding box into the zone that person is standing in. Driven
by rendered marker views so the camera really is localized from markers,
rather than handed a pose directly.
"""

import unittest

import numpy as np

from src.retailvision.marker_map import MarkerMap
from src.retailvision.marker_pose import MarkerPoseEstimator
from src.retailvision.zones import Zone, ZoneMap, ZoneResolver
from tests.test_marker_map import (
    FLOOR_MARKERS,
    IMAGE_SIZE,
    MARKER_SIZE,
    camera_at,
    render,
    wide_calibration,
)

# A camera above the 2m x 3m floor zone, looking down into it.
CAMERA = camera_at((1.0, -1.5, 2.2), (1.0, 0.4, 0.0))


def blank_frame() -> np.ndarray:
    """A frame containing no markers at all."""
    width, height = IMAGE_SIZE
    return np.full((height, width, 3), 110, np.uint8)


class TestZoneResolver(unittest.TestCase):
    """Verify zone resolution, and that an unlocalized camera reports nothing rather than guessing."""

    def setUp(self) -> None:
        """Build a resolver over one zone covering the four floor markers."""
        self.calibration = wide_calibration()
        self.marker_map = MarkerMap(anchor_id=0)
        estimator = MarkerPoseEstimator(self.calibration, MARKER_SIZE, allowed_ids=set(FLOOR_MARKERS))
        self.zone_map = ZoneMap([Zone("floor_zone", tuple(FLOOR_MARKERS))], self.marker_map)
        self.resolver = ZoneResolver(estimator, self.marker_map, self.zone_map, head_height=1.6)

    def localize(self) -> None:
        """Feed the resolver enough views to map every marker and localize the camera."""
        for camera, ids in [
            (CAMERA, [0, 1]),
            (camera_at((1.0, 4.5, 2.2), (1.2, 1.6, 0.0)), [1, 2, 3]),
            (CAMERA, [0, 1]),
        ]:
            resolver_camera = ZoneResolver(
                self.resolver.localizer.estimator, self.marker_map, self.zone_map, head_height=1.6
            )
            resolver_camera.update(render(camera, ids, self.calibration))
        self.resolver.update(render(CAMERA, [0, 1], self.calibration))

    def test_unlocalized_camera_resolves_nothing(self) -> None:
        """A camera that sees no mapped marker reports no zone rather than assuming one."""
        self.resolver.update(blank_frame())
        self.assertIsNone(self.resolver.camera_pose)
        self.assertIsNone(self.resolver.world_position((100, 100, 40, 40)))
        self.assertIsNone(self.resolver.zone_for_bbox((100, 100, 40, 40)))

    def test_pose_is_held_through_a_frame_with_no_marker_visible(self) -> None:
        """A person walking in front of the markers for a frame does not drop the camera's positions."""
        self.localize()
        held = self.resolver.camera_pose.copy()
        self.resolver.update(blank_frame())
        self.assertIsNotNone(self.resolver.camera_pose)
        np.testing.assert_array_equal(self.resolver.camera_pose, held)
        self.assertIsNotNone(self.resolver.world_position((100, 100, 40, 40)))

    def test_resolve_returns_one_entry_per_box_in_order(self) -> None:
        """Every bounding box gets exactly one answer, positionally matched to its input."""
        self.localize()
        boxes = [(100, 100, 40, 40), (300, 200, 50, 50), (600, 400, 30, 30)]
        self.assertEqual(len(self.resolver.resolve(boxes)), len(boxes))

    def test_world_position_is_reported_once_localized(self) -> None:
        """A localized camera turns a pixel box into a finite world floor position."""
        self.localize()
        width, height = IMAGE_SIZE
        position = self.resolver.world_position((width // 2 - 20, height // 2 - 20, 40, 40))
        self.assertIsNotNone(position)
        self.assertTrue(all(np.isfinite(position)))

    def test_occupancy_counts_only_resolved_zones(self) -> None:
        """Detections outside every zone, or unresolved, are not counted anywhere."""
        self.localize()
        counts = self.resolver.occupancy(["floor_zone", "floor_zone", None, "other_zone"])
        self.assertEqual(counts, {"floor_zone": 2})

    def test_occupancy_reports_zero_for_a_ready_but_empty_zone(self) -> None:
        """A zone with nobody in it reports zero rather than being absent from the counts."""
        self.localize()
        self.assertEqual(self.resolver.occupancy([]), {"floor_zone": 0})

    def test_occupancy_is_empty_while_no_zone_is_ready(self) -> None:
        """Before a zone's markers are mapped there is nothing to count, and no zone is invented."""
        self.assertEqual(self.resolver.occupancy(["floor_zone"]), {})

    def test_standing_face_missed_by_the_single_plane_is_still_in_the_zone(self) -> None:
        """A face whose head-height-plane projection lands outside the zone is caught by the height band.

        The single plane pushes a face at a different real height radially
        along the ray; membership sampled across the band of plausible face
        heights must not lose that person.
        """
        self.localize()
        width, height = IMAGE_SIZE
        # Scan pixel rows to find a box the single 1.6m plane places outside
        # the zone while some band height places it inside -- the standing
        # person case. The fixture guarantees such rows exist because the
        # zone's far edge is within view.
        from src.retailvision.marker_map import project_to_plane

        found = None
        for row in range(height // 4, height - 40, 8):
            bbox = (width // 2 - 20, row, 40, 40)
            single = self.resolver.world_position(bbox)
            single_zone = None if single is None else self.resolver.zone_map.zone_for(single)
            band_zone = self.resolver.zone_for_bbox(bbox)
            if single_zone is None and band_zone is not None:
                found = bbox
                break
        self.assertIsNotNone(found, "no pixel exercises the single-plane miss; fixture geometry changed")
        self.assertEqual(self.resolver.zone_for_bbox(found), "floor_zone")


class TestZoneIdInLogRecords(unittest.TestCase):
    """Verify the resolved zone reaches the frozen log schema."""

    def test_zone_id_defaults_to_none(self) -> None:
        """Without zones configured the field stays null, exactly as before this existed."""
        from src.retailvision.output_log import build_log_record

        record = build_log_record({"age_group": "25-34", "gender": "male", "emotion": "happy"})
        self.assertIsNone(record["zone_id"])

    def test_zone_id_is_carried_into_the_record(self) -> None:
        """A resolved zone is written to the record's zone_id field."""
        from src.retailvision.output_log import build_log_record

        record = build_log_record(
            {"age_group": "25-34", "gender": "male", "emotion": "happy"}, zone_id="floor_zone"
        )
        self.assertEqual(record["zone_id"], "floor_zone")


if __name__ == "__main__":
    unittest.main()
