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


def parse_args() -> argparse.Namespace:
    """Parse camera sources, their calibrations, the printed marker size, and optional zone config."""
    parser = argparse.ArgumentParser(description="Live test of marker pose estimation and the shared marker map")
    parser.add_argument("--source", nargs="+", default=["0"], help="One or more camera indices or video paths")
    parser.add_argument("--calibration", nargs="+", required=True, help="Calibration JSON per source, in the same order")
    parser.add_argument("--marker-size", type=float, default=0.10, help="Printed marker side length in meters (default: 0.10)")
    parser.add_argument("--anchor", type=int, default=None, help="Marker ID to fix as the world origin (default: first seen)")
    parser.add_argument("--zones", type=Path, default=None, help="Optional zone definition JSON")
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

    def __init__(self, source: str, calibration_path: str, marker_size: float, marker_map: MarkerMap) -> None:
        """Open the camera and bind it to its own calibration, pose estimator and localizer."""
        self.name = f"Camera {source}"
        self.calibration = CameraCalibration.load(calibration_path)
        self.estimator = MarkerPoseEstimator(self.calibration, marker_size)
        self.localizer = CameraLocalizer(self.estimator, marker_map)
        self.capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera at source: {source}")
        self._check_resolution()
        self.camera_pose: np.ndarray | None = None
        self.reprojection_error: float | None = None
        self.click: tuple[int, int] | None = None
        self.click_label = ""

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


def main() -> None:
    """Run every camera against one shared marker map, drawing poses, zones and click read-outs live."""
    args = parse_args()
    marker_map = MarkerMap(anchor_id=args.anchor)
    zones = load_zones(args.zones) if args.zones else []
    zone_map = ZoneMap(zones, marker_map) if zones else None

    views = [
        CameraView(source, calibration, args.marker_size, marker_map)
        for source, calibration in zip(args.source, args.calibration)
    ]
    for view in views:
        cv2.namedWindow(view.name)
        cv2.setMouseCallback(view.name, on_click, view)

    print(f"{len(views)} camera(s) open, sharing one marker map. Click to probe a position, 'r' resets, 'q' quits.")
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

                draw_markers(frame, view, result.marker_poses)
                if zone_map is not None:
                    draw_zones(frame, view, zone_map)
                resolve_click(view, zone_map, args.head_height)
                draw_status(frame, view, marker_map, zone_map)
                cv2.imshow(view.name, frame)

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
