"""
marker_map.py

The shared coordinate frame that lets several cameras describe the same
room in the same numbers, plus the localizer that keeps it trustworthy.

MarkerPoseEstimator gives each camera marker positions relative to itself,
which are not comparable between cameras. MarkerMap fixes one marker as
the origin of a single world frame and holds every other marker's pose in
it. A camera that can see any one already-mapped marker can work out where
it itself is standing, and from there any other marker it sees can be
added -- so the map grows outward from the anchor one overlapping view at
a time. Two cameras need one marker in common, not four, and a zone larger
than any single camera's view can still be described completely.

CameraLocalizer exists because a single marker's pose is ambiguous: two
orientations project to nearly the same pixels, and their reprojection
errors can be too close to separate. Position survives that ambiguity but
orientation does not, and since the map chains marker to camera to marker
through full orientations, one wrong pick corrupts everything mapped
afterwards. The localizer resolves it with information a lone marker does
not have -- how well a candidate explains the other markers in view, the
plane the markers are known to lie in, and the previous frame's pose.

Positions are in meters, taken purely from the markers' known printed
size; the room is never measured. The world frame is the anchor marker's
own frame, so with markers laid flat on the floor the world's z=0 plane is
the floor and world x/y are floor coordinates.

Marker poses are first-observation-wins: fixed when first seen from a
known vantage point, never refined afterwards. Averaging repeated
observations, or a full pose graph optimization, is future work -- worth
doing once drift across a long chain of markers is measured to be a
problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import CameraCalibration
from .marker_pose import MarkerPose, MarkerPoseEstimator, invert_pose, marker_object_points

# Faces, not whole bodies, are what this pipeline detects, so a face's pixel
# position back-projects onto a plane at roughly head height rather than onto the
# floor -- intersecting a face ray with the floor would place the person well
# beyond where they actually stand. The resulting x/y still reads as that
# person's floor footprint, which is what zone tests need.
DEFAULT_HEAD_HEIGHT_METERS = 1.6

# Below this, a viewing ray runs so nearly parallel to the target plane that the
# intersection point is numerically meaningless.
_MIN_RAY_PLANE_ANGLE = 1e-6

# A candidate camera pose within this factor of the best one is not meaningfully
# better explained by the image alone, so a tie-break on continuity with the
# previous frame decides it rather than sub-pixel noise.
_TIE_BREAK_RATIO = 1.5

# Candidate poses whose off-plane penalties are within this much of each other
# are not separated by the mounting plane either, and fall through to continuity.
_PLANARITY_TOLERANCE = 0.02


class MarkerMap:
    """Holds every known marker's pose in one shared world frame, growing as cameras observe overlapping markers."""

    def __init__(self, anchor_id: int | None = None) -> None:
        """Optionally fix which marker defines the world origin; if unset, the first marker mapped becomes it."""
        self.anchor_id = anchor_id
        self.poses: dict[int, np.ndarray] = {}

    def __contains__(self, marker_id: int) -> bool:
        """True if this marker's world pose is already known."""
        return marker_id in self.poses

    @property
    def marker_ids(self) -> list[int]:
        """Every marker ID currently placed in the world frame, sorted."""
        return sorted(self.poses)

    def clear(self) -> None:
        """Forget every mapped marker, so the next observation re-anchors the world frame."""
        self.poses.clear()

    def position(self, marker_id: int) -> np.ndarray | None:
        """Return a mapped marker's (x, y, z) world position in meters, or None if it isn't mapped yet."""
        matrix = self.poses.get(marker_id)
        return None if matrix is None else matrix[:3, 3].copy()

    def floor_position(self, marker_id: int) -> tuple[float, float] | None:
        """Return a mapped marker's world position projected to floor (x, y), or None if it isn't mapped yet."""
        position = self.position(marker_id)
        return None if position is None else (float(position[0]), float(position[1]))

    def normal(self, marker_id: int) -> np.ndarray | None:
        """Return the direction a mapped marker faces, in world coordinates, or None if it isn't mapped yet."""
        matrix = self.poses.get(marker_id)
        return None if matrix is None else matrix[:3, 2].copy()

    def world_corners(self, marker_id: int, marker_size: float) -> np.ndarray | None:
        """Return a mapped marker's four physical corners as world points, for reprojection against an image."""
        matrix = self.poses.get(marker_id)
        if matrix is None:
            return None
        local = marker_object_points(marker_size).astype(np.float64)
        return (matrix[:3, :3] @ local.T).T + matrix[:3, 3]


@dataclass
class Localization:
    """The outcome of localizing one camera against the shared map for a single frame."""

    camera_pose: np.ndarray | None = None
    marker_poses: dict[int, MarkerPose] = field(default_factory=dict)
    learned: list[int] = field(default_factory=list)
    reprojection_error: float | None = None


class CameraLocalizer:
    """Places one camera in the shared world frame each frame, resolving single-marker pose ambiguity."""

    def __init__(self, estimator: MarkerPoseEstimator, marker_map: MarkerMap) -> None:
        """Bind a camera's pose estimator to the marker map it contributes to and localizes against."""
        self.estimator = estimator
        self.marker_map = marker_map
        self.camera_pose: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> Localization:
        """Detect markers, localize the camera, extend the map with anything new, and report what happened."""
        candidates = self.estimator.estimate_candidates(frame)
        if not candidates:
            return Localization()

        if not self.marker_map.poses:
            self._anchor(candidates)

        camera_pose, error = self._best_camera_pose(candidates)
        if camera_pose is None:
            self.camera_pose = None
            return Localization(marker_poses={i: c[0] for i, c in candidates.items()})

        camera_pose = self._refine(camera_pose, candidates)
        self.camera_pose = camera_pose
        learned = self._extend(camera_pose, candidates)
        chosen = self._chosen_poses(camera_pose, candidates)
        return Localization(camera_pose=camera_pose, marker_poses=chosen, learned=learned, reprojection_error=error)

    def _anchor(self, candidates: dict[int, list[MarkerPose]]) -> None:
        """Seed an empty map by placing its anchor marker at the world origin."""
        anchor_id = self.marker_map.anchor_id
        if anchor_id is None:
            anchor_id = max(candidates, key=lambda i: candidates[i][0].pixel_area)
        if anchor_id not in candidates:
            return
        self.marker_map.anchor_id = anchor_id
        self.marker_map.poses[anchor_id] = np.eye(4, dtype=np.float64)

    def _camera_pose_candidates(self, candidates: dict[int, list[MarkerPose]]) -> list[np.ndarray]:
        """Every camera pose implied by pairing a mapped marker with one of its ambiguous orientations."""
        return [
            self.marker_map.poses[marker_id] @ invert_pose(pose.matrix)
            for marker_id, poses in candidates.items()
            if marker_id in self.marker_map
            for pose in poses
        ]

    def _reprojection_error(self, camera_pose: np.ndarray, candidates: dict[int, list[MarkerPose]]) -> float:
        """Average pixel error when every mapped marker in view is projected through a candidate camera pose.

        This is what breaks the single-marker ambiguity: a flipped orientation
        still explains its own marker's corners, but predicts the other markers
        in view badly, so scoring across all of them separates the two.
        """
        world_to_camera = invert_pose(camera_pose)
        rvec, _ = cv2.Rodrigues(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]

        total, count = 0.0, 0
        for marker_id, poses in candidates.items():
            corners = self.marker_map.world_corners(marker_id, self.estimator.marker_size)
            if corners is None:
                continue
            projected, _ = cv2.projectPoints(
                corners, rvec, tvec, self.estimator.calibration.camera_matrix, self.estimator.calibration.dist_coeffs
            )
            total += float(np.linalg.norm(projected.reshape(4, 2) - poses[0].corners, axis=1).sum())
            count += 4
        return total / count if count else float("inf")

    def _planarity_penalty(self, camera_pose: np.ndarray, candidates: dict[int, list[MarkerPose]]) -> float:
        """Score how badly a candidate camera pose would place the not-yet-mapped markers off the shared plane.

        Reprojection cannot separate the two orientations while only the anchor
        is mapped, since both explain that one marker equally. The markers being
        mounted on a common plane is the extra fact that does separate them: the
        wrong orientation lifts every other marker in view off that plane and
        tilts it, which this measures directly in meters and alignment.
        """
        penalty = 0.0
        for marker_id, poses in candidates.items():
            if marker_id in self.marker_map:
                continue
            best = max(poses, key=lambda pose: (camera_pose @ pose.matrix)[2, 2])
            world = camera_pose @ best.matrix
            penalty += abs(float(world[2, 3])) + (1.0 - float(world[2, 2]))
        return penalty

    def _best_camera_pose(self, candidates: dict[int, list[MarkerPose]]) -> tuple[np.ndarray | None, float | None]:
        """Choose the camera pose that best explains the mapped markers, then the plane, then frame-to-frame continuity."""
        options = self._camera_pose_candidates(candidates)
        if not options:
            return None, None

        scored = sorted(((self._reprojection_error(pose, candidates), pose) for pose in options), key=lambda s: s[0])
        best_error = scored[0][0]
        if not np.isfinite(best_error):
            return None, None

        close = [pose for error, pose in scored if error <= best_error * _TIE_BREAK_RATIO]
        if len(close) == 1:
            return close[0], best_error

        # Reprojection could not separate these, so fall back to the mounting
        # plane, and only then to continuity -- a fixed camera does not jump
        # between frames, but that is a weaker signal than the geometry.
        penalties = [self._planarity_penalty(pose, candidates) for pose in close]
        best_penalty = min(penalties)
        close = [pose for pose, penalty in zip(close, penalties) if penalty <= best_penalty + _PLANARITY_TOLERANCE]
        if self.camera_pose is not None and len(close) > 1:
            previous = self.camera_pose
            return min(close, key=lambda pose: _pose_distance(pose, previous)), best_error
        return close[0], best_error

    def _refine(self, camera_pose: np.ndarray, candidates: dict[int, list[MarkerPose]]) -> np.ndarray:
        """Re-solve the camera pose against all mapped corners at once, which is tighter than any single marker."""
        object_points, image_points = [], []
        for marker_id, poses in candidates.items():
            corners = self.marker_map.world_corners(marker_id, self.estimator.marker_size)
            if corners is None:
                continue
            object_points.append(corners)
            image_points.append(poses[0].corners)
        if len(object_points) < 2:
            return camera_pose

        world_to_camera = invert_pose(camera_pose)
        rvec, _ = cv2.Rodrigues(world_to_camera[:3, :3])
        ok, rvec, tvec = cv2.solvePnP(
            np.vstack(object_points).astype(np.float64),
            np.vstack(image_points).astype(np.float64),
            self.estimator.calibration.camera_matrix,
            self.estimator.calibration.dist_coeffs,
            rvec.copy(),
            world_to_camera[:3, 3].copy().reshape(3, 1),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return camera_pose
        refined = invert_pose(_matrix(rvec, tvec))
        return refined if self._reprojection_error(refined, candidates) <= self._reprojection_error(camera_pose, candidates) else camera_pose

    def _extend(self, camera_pose: np.ndarray, candidates: dict[int, list[MarkerPose]]) -> list[int]:
        """Map any newly visible markers, picking the orientation that lies in the plane the mapped markers share."""
        learned = []
        for marker_id, poses in candidates.items():
            if marker_id in self.marker_map:
                continue
            # Zone markers are all mounted flat on one surface, so of the two
            # orientations the correct one is whichever ends up facing the same
            # way as the anchor rather than back through the floor.
            best = max(poses, key=lambda pose: (camera_pose @ pose.matrix)[2, 2])
            self.marker_map.poses[marker_id] = camera_pose @ best.matrix
            learned.append(marker_id)
        return sorted(learned)

    def _chosen_poses(self, camera_pose: np.ndarray, candidates: dict[int, list[MarkerPose]]) -> dict[int, MarkerPose]:
        """Report each visible marker under the orientation consistent with the chosen camera pose."""
        world_to_camera = invert_pose(camera_pose)
        chosen = {}
        for marker_id, poses in candidates.items():
            world = self.marker_map.poses.get(marker_id)
            if world is None:
                chosen[marker_id] = poses[0]
                continue
            expected = world_to_camera @ world
            chosen[marker_id] = min(poses, key=lambda pose: float(np.linalg.norm(pose.matrix - expected)))
        return chosen


def _matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a solver's rotation and translation vectors."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3], _ = cv2.Rodrigues(rvec)
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return matrix


def _pose_distance(pose: np.ndarray, other: np.ndarray) -> float:
    """Combined positional and rotational difference between two poses, for continuity tie-breaks."""
    translation = float(np.linalg.norm(pose[:3, 3] - other[:3, 3]))
    rotation = float(np.linalg.norm(pose[:3, :3] - other[:3, :3]))
    return translation + rotation


def project_to_plane(
    pixel: tuple[float, float],
    camera_pose: np.ndarray,
    calibration: CameraCalibration,
    plane_z: float = 0.0,
) -> tuple[float, float] | None:
    """Trace the viewing ray through an image pixel out to a horizontal world plane, returning its (x, y) there."""
    point = np.array([[list(pixel)]], dtype=np.float64)
    normalized = cv2.undistortPoints(point, calibration.camera_matrix, calibration.dist_coeffs)
    direction_camera = np.array([normalized[0][0][0], normalized[0][0][1], 1.0])

    origin = camera_pose[:3, 3]
    direction_world = camera_pose[:3, :3] @ direction_camera
    if abs(direction_world[2]) < _MIN_RAY_PLANE_ANGLE:
        return None

    distance = (plane_z - origin[2]) / direction_world[2]
    if distance <= 0:
        return None

    intersection = origin + distance * direction_world
    return (float(intersection[0]), float(intersection[1]))
