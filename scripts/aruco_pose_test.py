"""
aruco_pose_test.py

Standalone spike for the marker-pose zone approach: opens one or more
calibrated cameras, solves every visible marker's 3D pose, and folds all
of them into a single shared MarkerMap.

Passing several sources is the real test of the design. Each camera is
localized independently from whichever mapped marker it can see, so point
two cameras at a room where neither sees every corner, give them one
marker in common, and the map should still end up holding all of them --
that is the property the homography approach could not provide.

With --zones, each ready zone's floor polygon is projected back into every
camera view, which is the visual check that the zone landed where the
markers were physically placed. Clicking anywhere in a view back-projects
that pixel onto a horizontal plane at head height and reports the world
position and containing zone, standing in for a real face detection.

Requires a calibration per camera from scripts/calibrate_camera.py, and
all markers printed at the same physical size passed as --marker-size.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/aruco_pose_test.py \
    --source 0 --calibration calibration/camera_0.json --marker-size 0.10
Press 'q' to quit, 'r' to reset the map.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.retailvision.calibration import CameraCalibration
from src.retailvision.marker_map import (
    DEFAULT_HEAD_HEIGHT_METERS,
    FLOOR_MOUNTING,
    WALL_MOUNTING,
    CameraLocalizer,
    MarkerMap,
    project_to_plane,
)
from src.retailvision.marker_pose import MarkerPoseEstimator, invert_pose
from src.retailvision.zones import ZoneMap, load_zones

# Overlay palette (BGR). Every piece of text is drawn twice -- a thick dark
# underlay, then the colour -- because a single thin stroke disappears against
# whatever the camera happens to be pointed at; see draw_label().
MARKER_COLOR = (0, 255, 255)  # yellow: per-marker ID + distance labels
ZONE_COLOR = (0, 165, 255)  # orange: zone outline + name
TEXT_COLOR = (0, 255, 0)  # green: healthy status lines
WARN_COLOR = (0, 0, 255)  # red: not-localized / error states
OUTLINE_COLOR = (0, 0, 0)


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.6) -> None:
    """Draw text with a dark outline so it stays legible over any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, OUTLINE_COLOR, 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

# Requesting a resolution makes the backend rebuild the capture session, and on
# macOS the camera can take several seconds to deliver its first frame after
# that. Reading immediately is indistinguishable from a dead camera.
WARMUP_TIMEOUT = 25.0

# How often the consolidated map status is printed to the console. The per-window
# overlays only show one camera each; whether the cameras are actually linked to
# one another is a property of the whole set, so it needs reporting in one place.
STATUS_INTERVAL = 2.0

# Pose accuracy collapses when a marker is either too small in the image or seen
# too near edge-on, and both look like ordinary detections until the numbers are
# compared. Below roughly 50 pixels a side there are too few pixels to localize
# corners precisely; beyond about 60 degrees off face-on the square degenerates
# towards a line and its corners stop being well separated.
MIN_MARKER_SIDE_PX = 50.0
MAX_MARKER_TILT_DEGREES = 60.0


def parse_args() -> argparse.Namespace:
    """Parse camera sources, their calibrations, the printed marker size, and optional zone config."""
    parser = argparse.ArgumentParser(description="Live test of marker pose estimation and the shared marker map")
    parser.add_argument("--source", nargs="+", default=["0"], help="One or more camera indices or video paths")
    parser.add_argument("--calibration", nargs="+", required=True, help="Calibration JSON per source, in the same order")
    parser.add_argument("--marker-size", type=float, default=0.10, help="Printed marker side length in meters (default: 0.10)")
    parser.add_argument("--anchor", type=int, default=None, help="Marker ID to fix as the world origin (default: first seen)")
    parser.add_argument(
        "--marker-mounting",
        choices=[FLOOR_MOUNTING, WALL_MOUNTING],
        default=FLOOR_MOUNTING,
        help="How markers are physically mounted: flat on the floor, or upright on walls (default: floor)",
    )
    parser.add_argument(
        "--marker-height",
        type=float,
        default=0.0,
        help="Height in meters of the ANCHOR marker's centre above the floor; puts the world's z=0 plane on the real floor",
    )
    parser.add_argument("--zones", type=Path, default=None, help="Optional zone definition JSON")
    parser.add_argument(
        "--save-map",
        type=Path,
        default=None,
        help="Write the surveyed marker map here on exit, for camera nodes to load with pipeline_demo --marker-map",
    )
    parser.add_argument(
        "--markers",
        type=int,
        nargs="+",
        default=None,
        help="Marker IDs actually deployed; anything else is ignored as a misread (default: those in --zones plus --anchor)",
    )
    parser.add_argument(
        "--head-height",
        type=float,
        default=DEFAULT_HEAD_HEIGHT_METERS,
        help=f"Plane height in meters that clicks back-project onto (default: {DEFAULT_HEAD_HEIGHT_METERS})",
    )
    args = parser.parse_args()
    if len(args.calibration) != len(args.source):
        parser.error("Pass one --calibration per --source, in the same order")
    return args


class CameraView:
    """One open camera contributing observations to the shared marker map."""

    def __init__(
        self,
        source: str,
        calibration_path: str,
        marker_size: float,
        marker_map: MarkerMap,
        allowed_ids: set[int] | None = None,
    ) -> None:
        """Open the camera and bind it to its own calibration, pose estimator and localizer."""
        self.name = f"Camera {source}"
        self.calibration = CameraCalibration.load(calibration_path)
        self.estimator = MarkerPoseEstimator(self.calibration, marker_size, allowed_ids=allowed_ids)
        self.localizer = CameraLocalizer(self.estimator, marker_map)
        self.capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera at source: {source}")
        # Ask the camera for exactly the resolution its calibration was captured
        # at, rather than leaving the two to be matched by hand. Focal length is
        # in pixels, so a mismatch rescales every distance without failing.
        if self._resolution() != tuple(self.calibration.image_size):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.calibration.image_size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calibration.image_size[1])
            if not self._wait_for_first_frame():
                self.capture.release()
                raise RuntimeError(
                    f"{self.name} delivered no frame within {WARMUP_TIMEOUT:.0f}s of being set to "
                    f"{self.calibration.image_size[0]}x{self.calibration.image_size[1]}. "
                    "Some cameras only run at their native resolution -- recalibrate this one there."
                )
        self._check_resolution()
        self.camera_pose: np.ndarray | None = None
        self.reprojection_error: float | None = None
        self.visible_markers: set[int] = set()
        self.localization = None
        self.click: tuple[int, int] | None = None
        self.click_label = ""

    def _resolution(self) -> tuple[int, int]:
        """The resolution the camera is currently delivering."""
        return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _wait_for_first_frame(self) -> bool:
        """Block until the camera delivers a frame after a resolution change, or the timeout expires."""
        deadline = time.monotonic() + WARMUP_TIMEOUT
        while time.monotonic() < deadline:
            ok, _ = self.capture.read()
            if ok:
                return True
            time.sleep(0.1)
        return False

    def _check_resolution(self) -> None:
        """Warn if the camera is running at a different resolution than it was calibrated at.

        Focal length is measured in pixels, so it only describes the camera at
        the resolution the calibration was captured at. Running at another
        resolution silently rescales every distance rather than failing, which
        is far harder to notice than a wrong-looking number.
        """
        live = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if live != tuple(self.calibration.image_size):
            print(
                f"WARNING: {self.name} is running at {live[0]}x{live[1]} but was calibrated at "
                f"{self.calibration.image_size[0]}x{self.calibration.image_size[1]}. "
                "Distances will be wrong -- recalibrate at this resolution."
            )

    def release(self) -> None:
        """Release the underlying capture device."""
        self.capture.release()


def on_click(event: int, x: int, y: int, flags: int, view: CameraView) -> None:
    """Record a left-click's pixel position on the view it happened in."""
    if event == cv2.EVENT_LBUTTONDOWN:
        view.click = (x, y)


def draw_markers(frame: np.ndarray, view: CameraView, poses: dict) -> None:
    """Draw each detected marker's 3D axes and its measured distance from the camera."""
    for pose in poses.values():
        cv2.drawFrameAxes(
            frame, view.calibration.camera_matrix, view.calibration.dist_coeffs, pose.rvec, pose.tvec, view.estimator.marker_size / 2
        )
        center = pose.corners.mean(axis=0).astype(int)
        draw_label(frame, f"{pose.marker_id}: {pose.distance_meters:.2f}m", (center[0] + 8, center[1]), MARKER_COLOR, scale=0.7)


# A camera standing inside the zone -- the normal deployment -- always has
# some polygon corners behind it, so the polygon is clipped to what the lens
# can see rather than skipped whenever any corner is out of view. The lateral
# bounds come from the calibration itself: the distortion polynomial is only
# meaningful within the field of view it was fitted over, and points past it
# fold back into the frame at meaningless positions, which draws as stray
# lines crossing the image. A guessed constant is wrong per-lens; the camera
# matrix says exactly where each frame edge is.
ZONE_FILL_ALPHA = 0.22
_NEAR_METERS = 0.05
_FRUSTUM_MARGIN = 1.15  # keep slightly past the frame edge so edges leave the image cleanly


def _clip(points: np.ndarray, signed_distance) -> np.ndarray:
    """Sutherland-Hodgman: keep the part of a polygon where signed_distance >= 0."""
    result = []
    for index in range(len(points)):
        a, b = points[index], points[(index + 1) % len(points)]
        da, db = signed_distance(a), signed_distance(b)
        if da >= 0:
            result.append(a)
        if (da >= 0) != (db >= 0):
            result.append(a + (da / (da - db)) * (b - a))
    return np.array(result)


def visible_zone_pixels(view: CameraView, polygon: np.ndarray) -> np.ndarray | None:
    """Project a zone's floor polygon into this view, clipped to the part the lens can actually see."""
    world_to_camera = invert_pose(view.camera_pose)
    corners = np.hstack([polygon, np.zeros((len(polygon), 1), dtype=np.float32)]).astype(np.float64)
    in_camera = (world_to_camera[:3, :3] @ corners.T + world_to_camera[:3, 3].reshape(3, 1)).T

    matrix = view.calibration.camera_matrix
    width, height = view.calibration.image_size
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    tan_left = (cx / fx) * _FRUSTUM_MARGIN
    tan_right = ((width - cx) / fx) * _FRUSTUM_MARGIN
    tan_up = (cy / fy) * _FRUSTUM_MARGIN
    tan_down = ((height - cy) / fy) * _FRUSTUM_MARGIN

    for bound in (
        lambda p: p[2] - _NEAR_METERS,
        lambda p: tan_right * p[2] - p[0],
        lambda p: tan_left * p[2] + p[0],
        lambda p: tan_down * p[2] - p[1],
        lambda p: tan_up * p[2] + p[1],
    ):
        if len(in_camera) < 3:
            return None
        in_camera = _clip(in_camera, bound)
    if len(in_camera) < 3:
        return None

    projected, _ = cv2.projectPoints(
        in_camera.astype(np.float64), np.zeros(3), np.zeros(3),
        view.calibration.camera_matrix, view.calibration.dist_coeffs,
    )
    return projected.reshape(-1, 2).astype(np.int32)


def draw_zones(frame: np.ndarray, view: CameraView, zone_map: ZoneMap) -> None:
    """Shade every ready zone's visible floor area in this camera's view and outline it."""
    if view.camera_pose is None:
        return
    for zone_id in zone_map.ready_zone_ids():
        pixels = visible_zone_pixels(view, zone_map.polygon(zone_id))
        if pixels is None:
            continue
        # A translucent fill reads as "this floor area" where a bare outline
        # reads as unrelated lines crossing the room.
        shaded = frame.copy()
        cv2.fillPoly(shaded, [pixels], ZONE_COLOR)
        cv2.addWeighted(shaded, ZONE_FILL_ALPHA, frame, 1 - ZONE_FILL_ALPHA, 0, frame)
        cv2.polylines(frame, [pixels], True, ZONE_COLOR, 2)
        height, width = frame.shape[:2]
        label_at = pixels.mean(axis=0).astype(int)
        draw_label(
            frame, zone_id,
            (int(np.clip(label_at[0], 10, width - 160)), int(np.clip(label_at[1], 20, height - 10))),
            ZONE_COLOR,
        )


def resolve_click(view: CameraView, zone_map: ZoneMap | None, head_height: float) -> None:
    """Back-project a pending click onto the head-height plane and record its world position and zone."""
    if view.click is None:
        return
    if view.camera_pose is None:
        view.click_label = "click: camera not localized"
        view.click = None
        return

    world = project_to_plane(view.click, view.camera_pose, view.calibration, plane_z=head_height)
    if world is None:
        view.click_label = "click: ray misses the plane"
    else:
        zone_id = zone_map.zone_for(world) if zone_map else None
        view.click_label = f"click: ({world[0]:.2f}, {world[1]:.2f})m zone={zone_id}"
        print(f"{view.name} {view.click_label}")
    view.click = None


def draw_status(frame: np.ndarray, view: CameraView, marker_map: MarkerMap, zone_map: ZoneMap | None) -> None:
    """Overlay this camera's world position, the shared map's contents, and zone readiness."""
    if view.camera_pose is None:
        lines = ["camera not localized -- show it a mapped marker"]
        color = WARN_COLOR
    else:
        position = view.camera_pose[:3, 3]
        lines = [f"camera at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})m"]
        color = TEXT_COLOR
        # A camera below z=0 is below the floor, which is impossible: it means
        # the world's datum is off, almost always a survey run without
        # --marker-height, leaving z=0 at the anchor instead of the floor.
        if position[2] < 0:
            lines.append("camera is BELOW the floor -- re-run with --marker-height <anchor centre height>")
            color = WARN_COLOR

    if view.reprojection_error is not None:
        lines.append(f"fit: {view.reprojection_error:.2f}px (under ~1px is healthy)")
    lines.append(f"mapped markers: {marker_map.marker_ids} anchor={marker_map.anchor_id}")
    if zone_map is not None:
        pending = {z: zone_map.missing_markers(z) for z in zone_map.zone_ids if zone_map.missing_markers(z)}
        lines.append(f"ready zones: {zone_map.ready_zone_ids()}")
        if pending:
            lines.append(f"waiting on: {pending}")
    if view.click_label:
        lines.append(view.click_label)

    for index, line in enumerate(lines):
        draw_label(frame, line, (10, 30 + index * 26), color if index == 0 else TEXT_COLOR)


def print_marker_detail(view: "CameraView", marker_map: MarkerMap) -> None:
    """Print each visible marker's measured distance and, when mapped, how far its own reprojection is off.

    A single aggregate fit says the frame is inconsistent but not which
    measurement disagrees. Per-marker numbers separate the usual causes: a wrong
    printed size shows as every distance being scaled alike, a duplicated or
    misplaced marker shows as one marker's reprojection dwarfing the rest.
    """
    result = view.localization
    if result is None or not result.marker_poses:
        return
    for marker_id in sorted(result.marker_poses):
        pose = result.marker_poses[marker_id]
        # A square cannot look larger than face-on, so comparing its apparent size
        # against the size its own reported distance implies recovers how far off
        # face-on it is -- and catches a distance that cannot be right at all.
        apparent = float(np.sqrt(max(pose.pixel_area, 1.0)))
        implied = view.calibration.camera_matrix[0, 0] * view.estimator.marker_size / max(pose.distance_meters, 1e-6)
        ratio = min(apparent / implied, 1.0) if implied > 0 else 0.0
        tilt = float(np.degrees(np.arccos(np.clip(ratio, 0.0, 1.0))))
        detail = f"distance {pose.distance_meters:5.2f}m, {apparent:4.0f}px side, {tilt:2.0f}deg off face-on"

        warnings = []
        if apparent < MIN_MARKER_SIDE_PX:
            warnings.append("TOO SMALL")
        if tilt > MAX_MARKER_TILT_DEGREES:
            warnings.append("TOO OBLIQUE")
        if warnings:
            detail += "  <<< " + " + ".join(warnings) + " -- pose unreliable"
        if view.camera_pose is not None and marker_id in marker_map:
            corners = marker_map.world_corners(marker_id, view.estimator.marker_size)
            world_to_camera = invert_pose(view.camera_pose)
            rvec, _ = cv2.Rodrigues(world_to_camera[:3, :3])
            projected, _ = cv2.projectPoints(
                corners, rvec, world_to_camera[:3, 3],
                view.calibration.camera_matrix, view.calibration.dist_coeffs,
            )
            off = float(np.linalg.norm(projected.reshape(4, 2) - pose.corners, axis=1).mean())
            detail += f", reprojects {off:8.2f}px off"
        else:
            detail += ", unmapped"
        print(f"      marker {marker_id}: {detail}")


def print_status(views: list["CameraView"], marker_map: MarkerMap, zone_map: ZoneMap | None) -> None:
    """Print whether every camera is localized and which markers link them, for the whole set at once."""
    print("\n--- marker map ---")
    print(f"  mapped markers: {marker_map.marker_ids}   anchor: {marker_map.anchor_id}")

    for view in views:
        seen = sorted(view.visible_markers)
        linked = sorted(m for m in seen if m in marker_map)
        if view.camera_pose is None:
            state = "NOT LOCALIZED"
        else:
            position = view.camera_pose[:3, 3]
            state = f"at ({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f})m  fit {view.reprojection_error:.2f}px"
        print(f"  {view.name}: sees {seen or '[]'}, of which mapped {linked or '[]'}  -> {state}")
        print_marker_detail(view, marker_map)

    # A marker seen by two cameras at once is what ties their coordinate frames
    # together; without at least one such marker the cameras cannot be related.
    bridges = []
    for first in range(len(views)):
        for second in range(first + 1, len(views)):
            shared = sorted(views[first].visible_markers & views[second].visible_markers)
            if shared:
                bridges.append(f"{views[first].name} <-> {views[second].name} via {shared}")
    if bridges:
        print("  shared markers (camera links):")
        for bridge in bridges:
            print(f"    {bridge}")
    else:
        print("  shared markers: NONE -- no camera pair sees a common marker right now")

    if zone_map is not None:
        for zone_id in zone_map.zone_ids:
            missing = zone_map.missing_markers(zone_id)
            print(f"  zone {zone_id}: {'READY' if not missing else f'waiting on {missing}'}")

    localized = sum(1 for v in views if v.camera_pose is not None)
    print(f"  => {localized}/{len(views)} camera(s) localized in the shared frame")


def main() -> None:
    """Run every camera against one shared marker map, drawing poses, zones and click read-outs live."""
    args = parse_args()
    marker_map = MarkerMap(anchor_id=args.anchor, mounting=args.marker_mounting, anchor_height=args.marker_height)
    zones = load_zones(args.zones) if args.zones else []
    zone_map = ZoneMap(zones, marker_map) if zones else None

    # Restrict detection to the markers actually deployed. A 4x4 marker is easy to
    # misread out of noise or blur, and a phantom ID that reaches the map is
    # permanent, so an explicit allow-list is worth far more than it costs.
    allowed_ids: set[int] | None = None
    if args.markers is not None:
        allowed_ids = set(args.markers)
    elif zones:
        allowed_ids = {m for zone in zones for m in zone.marker_ids}
    if allowed_ids is not None and args.anchor is not None:
        allowed_ids.add(args.anchor)
    if allowed_ids is not None:
        print(f"Accepting only marker IDs {sorted(allowed_ids)}; any other detection is treated as a misread.")

    views = [
        CameraView(source, calibration, args.marker_size, marker_map, allowed_ids)
        for source, calibration in zip(args.source, args.calibration)
    ]
    for view in views:
        cv2.namedWindow(view.name)
        cv2.setMouseCallback(view.name, on_click, view)

    print(f"{len(views)} camera(s) open, sharing one marker map. Click to probe a position, 'r' resets, 'q' quits.")
    last_status = 0.0
    try:
        while True:
            for view in views:
                ok, frame = view.capture.read()
                if not ok:
                    return

                result = view.localizer.update(frame)
                if result.learned:
                    print(f"{view.name} mapped new markers: {result.learned}")
                view.camera_pose = result.camera_pose
                view.reprojection_error = result.reprojection_error
                view.visible_markers = set(result.marker_poses)
                view.localization = result

                draw_markers(frame, view, result.marker_poses)
                if zone_map is not None:
                    draw_zones(frame, view, zone_map)
                resolve_click(view, zone_map, args.head_height)
                draw_status(frame, view, marker_map, zone_map)
                cv2.imshow(view.name, frame)

            if time.monotonic() - last_status >= STATUS_INTERVAL:
                print_status(views, marker_map, zone_map)
                last_status = time.monotonic()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                marker_map.clear()
                marker_map.anchor_id = args.anchor
                for view in views:
                    view.localizer.camera_pose = None
                print("Marker map reset")
    finally:
        for view in views:
            view.release()
        cv2.destroyAllWindows()
        if args.save_map is not None and marker_map.poses:
            marker_map.save(args.save_map)
            print(f"Wrote {args.save_map} with markers {marker_map.marker_ids} (anchor {marker_map.anchor_id}).")
            if zone_map is not None:
                pending = {z: zone_map.missing_markers(z) for z in zone_map.zone_ids if zone_map.missing_markers(z)}
                if pending:
                    print(f"WARNING: saved while zones are still incomplete: {pending}")


if __name__ == "__main__":
    main()
