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
missed crossing permanently skews the running total (see counter.py).

The polygon is the convex hull of the markers' floor positions, not a
ring visited in any configured order. Markers get taped up in whatever
sequence is convenient, so neither their IDs nor their listing order says
anything about which are physically adjacent -- and a marker whose floor
position lands inside the shape the others form (an interior pillar, or
survey noise pulling a wall marker inward) must not fold a dent into the
perimeter. The hull is exactly "the fence around all the markers", which
is what placing markers at a zone's edges means. The cost is that a
deliberately concave zone cannot be expressed; retail floor areas are
effectively rectangles, so that restriction does not bite.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .marker_map import (
    DEFAULT_HEAD_HEIGHT_METERS,
    CameraLocalizer,
    Localization,
    MarkerMap,
    project_to_plane,
)
from .marker_pose import MarkerPoseEstimator

MIN_ZONE_MARKERS = 3

# Faces are not all at one height: a seated face sits around 1.1-1.3m, a
# standing one around 1.5-1.8m. Zone membership therefore tests the viewing
# ray against the zone's vertical prism across this whole band, sampled at
# several heights, rather than against a single plane -- a single plane
# pushes anyone whose real face height differs from it radially along the
# ray, far enough to land outside a zone they are standing inside.
FACE_HEIGHT_BAND = (0.95, 1.85)
FACE_HEIGHT_SAMPLES = 7


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
    """Return the convex hull ring of the points -- the perimeter around them all, in whatever order they came."""
    array = np.array(list(points), dtype=np.float32)
    return cv2.convexHull(array).reshape(-1, 2)


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


class ZoneResolver:
    """Answers which zone a detected person is standing in, for one camera.

    Owns the whole chain a live pipeline needs -- localizing the camera
    against the shared marker map each frame, back-projecting a detection onto
    a horizontal plane at head height, and testing that position against the
    zone polygons -- so callers hand over a frame and bounding boxes and get
    zone identities back.

    A camera that has never seen a mapped marker resolves nothing rather than
    guessing. Once localized, though, the pose is held through frames where no
    marker is visible: these cameras are fixed installs, and the usual reason a
    marker vanishes for a frame is a person walking in front of it -- exactly
    the moment dropping every position would hurt most. The cost is that a
    camera physically moved without a marker in view keeps reporting from its
    old pose until it sees one again.
    """

    def __init__(
        self,
        estimator: MarkerPoseEstimator,
        marker_map: MarkerMap,
        zone_map: "ZoneMap",
        head_height: float = DEFAULT_HEAD_HEIGHT_METERS,
    ) -> None:
        """Bind a camera's pose estimator and the shared marker map to the zones being measured."""
        self.localizer = CameraLocalizer(estimator, marker_map)
        self.zone_map = zone_map
        self.head_height = head_height
        self.camera_pose: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> Localization:
        """Re-localize the camera from the markers visible in this frame, holding the last good pose otherwise."""
        result = self.localizer.update(frame)
        if result.camera_pose is not None:
            self.camera_pose = result.camera_pose
        return result

    def world_position(self, bbox: tuple[int, int, int, int]) -> tuple[float, float] | None:
        """Return the world floor position of a detection's bounding box, or None if the camera is not localized.

        The box centre is used rather than its base: this pipeline detects
        faces, so the bottom edge is a chin rather than a pair of feet, and the
        ray is intersected with a plane at head height instead of the floor.
        """
        if self.camera_pose is None:
            return None
        x, y, width, height = bbox
        return project_to_plane(
            (x + width / 2.0, y + height / 2.0),
            self.camera_pose,
            self.localizer.estimator.calibration,
            plane_z=self.head_height,
        )

    def floor_world_position(self, pixel: tuple[float, float]) -> tuple[float, float] | None:
        """Return the world position of a pixel known to be where someone meets the floor.

        This is the accurate path, and it exists because a body detection
        supplies something a face never could: a point that is on the floor
        by definition. The ray through it is intersected with z = 0, so
        there is no assumed height to be wrong about -- unlike
        world_position() below, whose error grows with the difference
        between a person's real head height and the configured one.

        Callers get None when the camera is not localized, and are expected
        to have already established that the feet are actually visible; see
        person_detection.floor_pixel(), which returns None when they are not.
        """
        if self.camera_pose is None:
            return None
        return project_to_plane(
            pixel,
            self.camera_pose,
            self.localizer.estimator.calibration,
            plane_z=0.0,
        )

    def zone_for_floor(self, pixel: tuple[float, float]) -> str | None:
        """Return the zone containing a floor-contact pixel, or None if outside them all.

        No height band is sampled here. The face path has to try a range of
        plausible head heights because it cannot tell a seated person from a
        standing one; a floor contact is the same point either way.
        """
        position = self.floor_world_position(pixel)
        return None if position is None else self.zone_map.zone_for(position)

    def zone_for_bbox(self, bbox: tuple[int, int, int, int]) -> str | None:
        """Return the zone whose vertical prism the detection's viewing ray crosses at a plausible face height.

        The face lies somewhere on the ray through its pixel; where exactly
        depends on how tall the person is and whether they are seated, which a
        single camera cannot tell. Sampling the ray across the plausible band
        of face heights and accepting any sample inside the zone counts seated
        and standing people alike. The reported world position stays the
        configured head-height estimate: membership needs only "is the ray in
        the prism", but a coordinate has to commit to one plane.
        """
        if self.camera_pose is None:
            return None
        x, y, width, height = bbox
        pixel = (x + width / 2.0, y + height / 2.0)
        low, high = FACE_HEIGHT_BAND
        for step in range(FACE_HEIGHT_SAMPLES):
            plane = low + (high - low) * step / (FACE_HEIGHT_SAMPLES - 1)
            position = project_to_plane(pixel, self.camera_pose, self.localizer.estimator.calibration, plane_z=plane)
            if position is None:
                continue
            zone_id = self.zone_map.zone_for(position)
            if zone_id is not None:
                return zone_id
        return None

    def resolve(self, bboxes: list[tuple[int, int, int, int]]) -> list[str | None]:
        """Return one zone ID (or None) per bounding box, in the same order."""
        return [self.zone_for_bbox(bbox) for bbox in bboxes]

    def occupancy(self, zone_ids: list[str | None]) -> dict[str, int]:
        """Count how many of the resolved detections fall in each zone, for every ready zone."""
        counts = {zone_id: 0 for zone_id in self.zone_map.ready_zone_ids()}
        for zone_id in zone_ids:
            if zone_id in counts:
                counts[zone_id] += 1
        return counts
