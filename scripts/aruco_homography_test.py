"""
aruco_homography_test.py

Standalone spike: once all 4 configured zone markers are visible, computes
a homography from their pixel positions to a normalized zone-local unit
square (0,0)-(1,1) -- no physical measurement of the zone's real size or
shape, ever, since that won't stay constant across deployments. Draws a
crosshair at the frame's center and shows what zone-local coordinate it
maps to, as a live sanity check: pointing the camera so the center lands
near marker 0 should show a coordinate near (0,0), near marker 3 near
(1,1), and so on. This is the piece that makes cross-camera fusion
possible later -- once each camera has its own homography for a zone, a
detected point from any contributing camera maps into the same shared
zone-local space, directly comparable regardless of which camera saw it.

Marker-to-corner convention: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1).

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/aruco_homography_test.py
Press 'q' to quit.
"""

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())

ZONE_CORNERS = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}


def marker_centers(corners: list, ids) -> dict[int, np.ndarray]:
    """Return {marker_id: pixel center} for every detected marker."""
    return {int(marker_id): marker_corners[0].mean(axis=0) for marker_corners, marker_id in zip(corners, ids.flatten())}


def compute_homography(centers: dict[int, np.ndarray]) -> np.ndarray | None:
    """Return the pixel-to-zone-local homography if all 4 configured markers are visible, else None."""
    if not set(ZONE_CORNERS).issubset(centers):
        return None
    src = np.array([centers[marker_id] for marker_id in sorted(ZONE_CORNERS)], dtype=np.float32)
    dst = np.array([ZONE_CORNERS[marker_id] for marker_id in sorted(ZONE_CORNERS)], dtype=np.float32)
    matrix, _ = cv2.findHomography(src, dst)
    return matrix


def main() -> None:
    """Open the camera and show the live zone-local coordinate of the frame's center point."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    print("Camera opened. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            frame_center = (width / 2, height / 2)
            cv2.drawMarker(frame, (int(frame_center[0]), int(frame_center[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            corners, ids, _ = DETECTOR.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                centers = marker_centers(corners, ids)
                homography = compute_homography(centers)
                if homography is not None:
                    point = np.array([[frame_center]], dtype=np.float32)
                    zone_local = cv2.perspectiveTransform(point, homography)[0][0]
                    label = f"zone-local: ({zone_local[0]:.2f}, {zone_local[1]:.2f})"
                    color = (0, 255, 0)
                else:
                    missing = sorted(set(ZONE_CORNERS) - set(centers))
                    label = f"Need all 4 markers -- missing {missing}"
                    color = (0, 0, 255)
            else:
                label = "No markers detected"
                color = (0, 0, 255)

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("ArUco homography test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
