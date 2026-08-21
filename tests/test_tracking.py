"""
test_tracking.py

Unit tests for CentroidTracker: track ID stability across frames, correct
handling of new/disappearing objects, and correct nearest-centroid
assignment when multiple objects are present.
"""

import unittest

from src.retailvision.tracking import CentroidTracker, bbox_centroid


class TestBboxCentroid(unittest.TestCase):
    """Verify bbox_centroid() computes the correct center point."""

    def test_centroid_of_bbox(self) -> None:
        """The centroid of a 100x50 box at (0, 0) is its midpoint."""
        self.assertEqual(bbox_centroid((0, 0, 100, 50)), (50.0, 25.0))


class TestCentroidTracker(unittest.TestCase):
    """Verify CentroidTracker assigns and maintains track IDs correctly."""

    def test_first_frame_registers_new_tracks(self) -> None:
        """Every bbox in the first frame gets a distinct new track ID."""
        tracker = CentroidTracker()
        track_ids = tracker.update([(0, 0, 10, 10), (100, 100, 10, 10)])
        self.assertEqual(len(track_ids), 2)
        self.assertEqual(len(set(track_ids)), 2)

    def test_same_object_keeps_same_id_across_frames(self) -> None:
        """A single object moving a small amount frame-to-frame keeps its track ID."""
        tracker = CentroidTracker()
        (first_id,) = tracker.update([(0, 0, 10, 10)])
        (second_id,) = tracker.update([(2, 2, 10, 10)])
        (third_id,) = tracker.update([(4, 4, 10, 10)])
        self.assertEqual(first_id, second_id)
        self.assertEqual(second_id, third_id)

    def test_two_objects_do_not_swap_ids(self) -> None:
        """Two well-separated objects each keep their own ID as both move slightly."""
        tracker = CentroidTracker()
        left_id, right_id = tracker.update([(0, 0, 10, 10), (500, 0, 10, 10)])
        left_id_2, right_id_2 = tracker.update([(5, 0, 10, 10), (505, 0, 10, 10)])
        self.assertEqual(left_id, left_id_2)
        self.assertEqual(right_id, right_id_2)

    def test_track_deregistered_after_max_disappeared_frames(self) -> None:
        """A track not seen for more than max_disappeared frames is dropped, freeing its ID for reuse."""
        tracker = CentroidTracker(max_disappeared=2)
        (original_id,) = tracker.update([(0, 0, 10, 10)])

        tracker.update([])
        tracker.update([])
        tracker.update([])  # 3rd consecutive empty frame: exceeds max_disappeared=2

        (new_id,) = tracker.update([(0, 0, 10, 10)])
        self.assertNotEqual(original_id, new_id)

    def test_track_survives_brief_disappearance(self) -> None:
        """A track missing for fewer than max_disappeared frames keeps its ID when it reappears."""
        tracker = CentroidTracker(max_disappeared=3)
        (original_id,) = tracker.update([(0, 0, 10, 10)])

        tracker.update([])
        tracker.update([])

        (reappeared_id,) = tracker.update([(1, 1, 10, 10)])
        self.assertEqual(original_id, reappeared_id)

    def test_far_apart_new_object_gets_new_id_not_matched(self) -> None:
        """A new object far beyond max_distance from an existing track gets its own new ID, not matched to it."""
        tracker = CentroidTracker(max_distance=50.0)
        (existing_id,) = tracker.update([(0, 0, 10, 10)])
        track_ids = tracker.update([(0, 0, 10, 10), (1000, 1000, 10, 10)])
        self.assertIn(existing_id, track_ids)
        self.assertEqual(len(set(track_ids)), 2)


if __name__ == "__main__":
    unittest.main()
