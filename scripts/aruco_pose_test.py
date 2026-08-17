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
    CameraLocalizer,
    MarkerMap,
    project_to_plane,
)
from src.retailvision.marker_pose import MarkerPoseEstimator, invert_pose
from src.retailvision.zones import ZoneMap, load_zones

AXIS_COLOR = (255, 200, 0)
TEXT_COLOR = (0, 255, 0)
WARN_COLOR = (0, 0, 255)

# Requesting a resolution makes the backend rebuild the capture session, and on
# macOS the camera can take several seconds to deliver its first frame after
# that. Reading immediately is indistinguishable from a dead camera.
WARMUP_TIMEOUT = 25.0

# How often the consolidated map status is printed to the console. The per-window
# overlays only show one camera each; whether the cameras are actually linked to
# one another is a property of the whole set, so it needs reporting in one place.
STATUS_INTERVAL = 2.0


def parse_args() -> argparse.Namespace:
    """Parse camera sources, their calibrations, the printed marker size, and optional zone config."""
    parser = argparse.ArgumentParser(description="Live test of marker pose estimation and the shared marker map")
    parser.add_argument("--source", nargs="+", default=["0"], help="One or more camera indices or video paths")
    parser.add_argument("--calibration", nargs="+", required=True, help="Calibration JSON per source, in the same order")
    parser.add_argument("--marker-size", type=float, default=0.10, help="Printed marker side length in meters (default: 0.10)")
    parser.add_argument("--anchor", type=int, default=None, help="Marker ID to fix as the world origin (default: first seen)")
    parser.add_argument("--zones", type=Path, default=None, help="Optional zone definition JSON")
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
        cv2.putText(frame, f"{pose.marker_id}: {pose.distance_meters:.2f}m", (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AXIS_COLOR, 2)


def draw_zones(frame: np.ndarray, view: CameraView, zone_map: ZoneMap) -> None:
    """Project every ready zone's world floor polygon back into this camera's view and outline it."""
    if view.camera_pose is None:
        return
    world_to_camera = invert_pose(view.camera_pose)
    rvec, _ = cv2.Rodrigues(world_to_camera[:3, :3])
    tvec = world_to_camera[:3, 3]

    for zone_id in zone_map.ready_zone_ids():
        polygon = zone_map.polygon(zone_id)
        corners = np.hstack([polygon, np.zeros((len(polygon), 1), dtype=np.float32)]).astype(np.float64)
        # Points behind the camera project to meaningless pixels, so only draw a
        # zone this view is actually looking at.
        if np.any((world_to_camera[:3, :3] @ corners.T + tvec.reshape(3, 1))[2] <= 0):
            continue
        projected, _ = cv2.projectPoints(corners, rvec, tvec, view.calibration.camera_matrix, view.calibration.dist_coeffs)
        cv2.polylines(frame, [projected.reshape(-1, 2).astype(np.int32)], True, AXIS_COLOR, 2)
        label_at = projected.reshape(-1, 2).astype(np.int32)[0]
        cv2.putText(frame, zone_id, tuple(label_at), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AXIS_COLOR, 2)


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
        cv2.putText(frame, line, (10, 30 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color if index == 0 else TEXT_COLOR, 2)


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
    marker_map = MarkerMap(anchor_id=args.anchor)
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


if __name__ == "__main__":
    main()
