"""
test_counter.py

Unit tests for LineCounter: entry/exit detection, net occupancy counting,
and dwell time tracking, driven by synthetic centroid sequences.
"""

import unittest

from src.retailvision.counter import LineCounter


class TestLineCounter(unittest.TestCase):
    """Verify LineCounter correctly detects crossings and tracks occupancy/dwell."""

    def test_first_sighting_emits_no_event(self) -> None:
        """A track's first appearance has no prior side to compare against, so no crossing is reported."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        events = counter.update({1: (50.0, 0.0)}, timestamp=0.0)
        self.assertEqual(events, [])
        self.assertEqual(counter.count, 0)

    def test_crossing_increasing_direction_is_entry(self) -> None:
        """A track moving from below the line to above it registers an entry and increments occupancy."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        counter.update({1: (50.0, 0.0)}, timestamp=0.0)
        events = counter.update({1: (150.0, 0.0)}, timestamp=1.0)
        self.assertEqual(events, [(1, "entry")])
        self.assertEqual(counter.count, 1)

    def test_crossing_back_is_exit(self) -> None:
        """A track that re-crosses back the other way registers an exit and decrements occupancy."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        counter.update({1: (50.0, 0.0)}, timestamp=0.0)
        counter.update({1: (150.0, 0.0)}, timestamp=1.0)
        events = counter.update({1: (50.0, 0.0)}, timestamp=2.0)
        self.assertEqual(events, [(1, "exit")])
        self.assertEqual(counter.count, 0)

    def test_decreasing_entry_direction(self) -> None:
        """With entry_direction='decreasing', crossing from above to below the line is the entry."""
        counter = LineCounter(axis="y", position=100, entry_direction="decreasing")
        counter.update({1: (0.0, 150.0)}, timestamp=0.0)
        events = counter.update({1: (0.0, 50.0)}, timestamp=1.0)
        self.assertEqual(events, [(1, "entry")])
        self.assertEqual(counter.count, 1)

    def test_staying_on_same_side_emits_no_event(self) -> None:
        """Small movement that doesn't cross the line produces no entry/exit event."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        counter.update({1: (50.0, 0.0)}, timestamp=0.0)
        events = counter.update({1: (60.0, 0.0)}, timestamp=1.0)
        self.assertEqual(events, [])
        self.assertEqual(counter.count, 0)

    def test_count_does_not_go_negative(self) -> None:
        """Occupancy is floored at 0 even if an exit is somehow processed without a matching entry."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        counter.count = 0
        counter.update({1: (150.0, 0.0)}, timestamp=0.0)
        counter.update({1: (50.0, 0.0)}, timestamp=1.0)
        self.assertEqual(counter.count, 0)

    def test_dwell_seconds_none_before_entry(self) -> None:
        """dwell_seconds() is None for a track that has never entered."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        self.assertIsNone(counter.dwell_seconds(1, timestamp=5.0))

    def test_dwell_seconds_after_entry(self) -> None:
        """dwell_seconds() returns elapsed time since entry while the track remains inside."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        counter.update({1: (50.0, 0.0)}, timestamp=0.0)
        counter.update({1: (150.0, 0.0)}, timestamp=10.0)
        self.assertEqual(counter.dwell_seconds(1, timestamp=25.0), 15.0)

    def test_dwell_seconds_none_after_exit(self) -> None:
        """dwell_seconds() reverts to None once a track has exited."""
        counter = LineCounter(axis="x", position=100, entry_direction="increasing")
        counter.update({1: (50.0, 0.0)}, timestamp=0.0)
        counter.update({1: (150.0, 0.0)}, timestamp=10.0)
        counter.update({1: (50.0, 0.0)}, timestamp=20.0)
        self.assertIsNone(counter.dwell_seconds(1, timestamp=25.0))


if __name__ == "__main__":
    unittest.main()
