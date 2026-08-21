"""
test_zones.py

Unit tests for zone polygons and occupancy: corner ordering, the partial
state where a zone's markers are not all mapped yet, and point-in-zone
counting. Zones are driven off a stub marker map so the geometry under
test is exact and independent of pose estimation.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retailvision.marker_map import MarkerMap
from src.retailvision.zones import Zone, ZoneMap, load_zones, order_polygon


def marker_map_at(positions: dict[int, tuple[float, float]]) -> MarkerMap:
    """Build a marker map with markers placed directly at the given floor positions."""
    marker_map = MarkerMap(anchor_id=0)
    for marker_id, (x, y) in positions.items():
        matrix = np.eye(4)
        matrix[:3, 3] = [x, y, 0.0]
        marker_map.poses[marker_id] = matrix
    return marker_map


SQUARE = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (2.0, 2.0), 3: (0.0, 2.0)}


class TestOrderPolygon(unittest.TestCase):
    """Verify corners are sorted into a valid ring regardless of input order."""

    def test_scrambled_corners_form_a_simple_ring(self) -> None:
        """Corners given in a crossing order are reordered so consecutive points are adjacent."""
        scrambled = [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)]
        ordered = order_polygon(scrambled)
        sides = [float(np.linalg.norm(ordered[i] - ordered[(i + 1) % 4])) for i in range(4)]
        for side in sides:
            self.assertAlmostEqual(side, 2.0, places=5)

    def test_interior_point_does_not_dent_the_perimeter(self) -> None:
        """A marker whose floor position lands inside the others still yields the outer fence, not a folded ring."""
        with_interior = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
        ordered = order_polygon(with_interior)
        self.assertEqual(ordered.shape, (4, 2))
        self.assertNotIn([1.0, 1.0], ordered.tolist())


class TestZoneMap(unittest.TestCase):
    """Verify zone readiness, polygon construction and occupancy counting."""

    def setUp(self) -> None:
        """Configure a single 2x2 meter zone over the four square markers."""
        self.zone = Zone(zone_id="entrance", marker_ids=(0, 1, 2, 3))

    def test_zone_needs_at_least_three_corners(self) -> None:
        """Two markers cannot bound an area, so such a zone is rejected at construction."""
        with self.assertRaises(ValueError):
            ZoneMap([Zone(zone_id="bad", marker_ids=(0, 1))], marker_map_at(SQUARE))

    def test_polygon_is_unavailable_until_every_corner_is_mapped(self) -> None:
        """A partially mapped zone reports which corners are missing and yields no polygon."""
        zone_map = ZoneMap([self.zone], marker_map_at({0: (0.0, 0.0), 1: (2.0, 0.0)}))
        self.assertEqual(zone_map.missing_markers("entrance"), [2, 3])
        self.assertIsNone(zone_map.polygon("entrance"))
        self.assertEqual(zone_map.ready_zone_ids(), [])

    def test_zone_becomes_ready_once_all_corners_are_mapped(self) -> None:
        """Corners contributed from different cameras still complete the same zone."""
        zone_map = ZoneMap([self.zone], marker_map_at(SQUARE))
        self.assertEqual(zone_map.missing_markers("entrance"), [])
        self.assertEqual(zone_map.ready_zone_ids(), ["entrance"])
        self.assertEqual(zone_map.polygon("entrance").shape, (4, 2))

    def test_contains_distinguishes_inside_from_outside(self) -> None:
        """A point within the marked square is inside the zone and one beyond it is not."""
        zone_map = ZoneMap([self.zone], marker_map_at(SQUARE))
        self.assertTrue(zone_map.contains("entrance", (1.0, 1.0)))
        self.assertFalse(zone_map.contains("entrance", (3.0, 1.0)))

    def test_unready_zone_contains_nothing(self) -> None:
        """Before its corners are known a zone has no extent, so nothing is reported inside it."""
        zone_map = ZoneMap([self.zone], marker_map_at({0: (0.0, 0.0)}))
        self.assertFalse(zone_map.contains("entrance", (1.0, 1.0)))

    def test_zone_for_returns_none_outside_every_zone(self) -> None:
        """A position outside all configured zones maps to no zone rather than the nearest one."""
        zone_map = ZoneMap([self.zone], marker_map_at(SQUARE))
        self.assertEqual(zone_map.zone_for((1.0, 1.0)), "entrance")
        self.assertIsNone(zone_map.zone_for((-5.0, -5.0)))

    def test_occupancy_counts_only_positions_inside_each_zone(self) -> None:
        """Occupancy is a live headcount per zone, ignoring tracks positioned outside them."""
        aisle = Zone(zone_id="aisle", marker_ids=(4, 5, 6, 7))
        positions = dict(SQUARE)
        positions.update({4: (5.0, 0.0), 5: (7.0, 0.0), 6: (7.0, 2.0), 7: (5.0, 2.0)})
        zone_map = ZoneMap([self.zone, aisle], marker_map_at(positions))
        counts = zone_map.occupancy({1: (0.5, 0.5), 2: (1.5, 1.5), 3: (6.0, 1.0), 4: (20.0, 20.0)})
        self.assertEqual(counts, {"entrance": 2, "aisle": 1})


class TestLoadZones(unittest.TestCase):
    """Verify zone definitions round-trip from their JSON config format."""

    def test_loads_zone_ids_and_corner_markers(self) -> None:
        """Each configured entry becomes a Zone with its marker IDs as a tuple."""
        payload = {"zones": [{"zone_id": "entrance", "marker_ids": [0, 1, 2, 3]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            path.write_text(json.dumps(payload))
            zones = load_zones(path)
        self.assertEqual(zones, [Zone(zone_id="entrance", marker_ids=(0, 1, 2, 3))])


if __name__ == "__main__":
    unittest.main()
