"""
zones.py

Zone definition and occupancy testing on top of a MarkerMap.

A zone is just a set of marker IDs physically placed at the corners of a
floor area. Because those markers' world positions come from the shared
map, a zone's polygon is assembled from wherever each corner was mapped
from -- corners seen only by camera A and corners seen only by camera B
still form one polygon. No camera ever needs a view of the whole zone,
which is the practical problem the marker map exists to solve.

Occupancy is a point-in-polygon test against a person's floor position, so
it reports who is inside right now rather than accumulating crossings.
That removes the drift the virtual-line counter is prone to, where a
missed crossing permanently skews the running total (see
docs/people_counter.md).

Corner ordering is derived, not configured: corners are sorted by angle
around their centroid so that markers can be listed in any order, which
assumes the zone is convex. Retail floor zones are effectively rectangles,
so this trades a shape restriction that does not bite for a setup step
that is easy to get wrong.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .marker_map import MarkerMap

MIN_ZONE_MARKERS = 3


@dataclass(frozen=True)
class Zone:
    """A named floor area, defined by the markers physically placed at its corners."""

    zone_id: str
    marker_ids: tuple[int, ...]


def load_zones(path: str | Path) -> list[Zone]:
    """Read zone definitions from a JSON file of {"zones": [{"zone_id": ..., "marker_ids": [...]}]}."""
    payload = json.loads(Path(path).read_text())
    return [Zone(zone_id=z["zone_id"], marker_ids=tuple(z["marker_ids"])) for z in payload["zones"]]


def order_polygon(points: Iterable[tuple[float, float]]) -> np.ndarray:
    """Sort points into a non-self-intersecting ring by angle around their centroid, assuming a convex shape."""
    array = np.array(list(points), dtype=np.float32)
    centroid = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - centroid[1], array[:, 0] - centroid[0])
    return array[np.argsort(angles)]


class ZoneMap:
    """Turns mapped marker positions into zone polygons and answers which zone a floor position falls in."""

    def __init__(self, zones: Sequence[Zone], marker_map: MarkerMap) -> None:
        """Bind zone definitions to the marker map that supplies their corner positions."""
        for zone in zones:
            if len(zone.marker_ids) < MIN_ZONE_MARKERS:
                raise ValueError(f"Zone {zone.zone_id!r} needs at least {MIN_ZONE_MARKERS} corner markers")
        self.zones = list(zones)
        self.marker_map = marker_map

    @property
    def zone_ids(self) -> list[str]:
        """Every configured zone's ID, in configuration order."""
        return [zone.zone_id for zone in self.zones]

    def _zone(self, zone_id: str) -> Zone | None:
        """Look up a configured zone by ID."""
        return next((zone for zone in self.zones if zone.zone_id == zone_id), None)

    def missing_markers(self, zone_id: str) -> list[int]:
        """List the zone's corner markers not yet placed in the map -- what still needs to be shown to a camera."""
        zone = self._zone(zone_id)
        if zone is None:
            return []
        return sorted(m for m in zone.marker_ids if m not in self.marker_map)

    def polygon(self, zone_id: str) -> np.ndarray | None:
        """Return the zone's floor polygon in world meters, or None until all its corner markers are mapped."""
        zone = self._zone(zone_id)
        if zone is None or self.missing_markers(zone_id):
            return None
        corners = [self.marker_map.floor_position(m) for m in zone.marker_ids]
        return order_polygon(corners)

    def ready_zone_ids(self) -> list[str]:
        """Every zone whose corner markers are all mapped, and which can therefore be tested against."""
        return [zone.zone_id for zone in self.zones if not self.missing_markers(zone.zone_id)]

    def contains(self, zone_id: str, point: tuple[float, float]) -> bool:
        """True if a world floor position lies inside the zone; False if outside or the zone isn't ready."""
        polygon = self.polygon(zone_id)
        if polygon is None:
            return False
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def zone_for(self, point: tuple[float, float]) -> str | None:
        """Return the first configured zone containing a world floor position, or None if it is outside them all."""
        return next((zone.zone_id for zone in self.zones if self.contains(zone.zone_id, point)), None)

    def occupancy(self, points: dict[int, tuple[float, float]]) -> dict[str, int]:
        """Count how many of the given per-track floor positions fall inside each ready zone."""
        counts = {zone_id: 0 for zone_id in self.ready_zone_ids()}
        for point in points.values():
            zone_id = self.zone_for(point)
            if zone_id in counts:
                counts[zone_id] += 1
        return counts
