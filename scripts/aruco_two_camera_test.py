"""
aruco_two_camera_test.py

Standalone spike: opens two cameras on the same machine, both aimed at
the same physical 4-marker zone from different angles, each independently
computing its own homography (see aruco_homography_test.py). Prints both
cameras' zone-local reading for their own frame center side by side --
this is the core fusion hypothesis, stripped of all networking: if two
different camera views of the same real-world point converge to the same
(or very close) zone-local coordinate, cross-camera fusion is viable.
Two cameras on one machine need no server/network at all -- the
comparison happens entirely in this one process's memory.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/aruco_two_camera_test.py --source-a 0 --source-b 1
Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())

ZONE_CORNERS = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}


def parse_args() -> argparse.Namespace:
    """Parse --source-a and --source-b, the two camera indices to compare."""
    parser = argparse.ArgumentParser(description="Compare two cameras' homography readings of the same zone")
    parser.add_argument("--source-a", default="0", help="First camera index (default: 0)")
    parser.add_argument("--source-b", default="1", help="Second camera index (default: 1)")
    return parser.parse_args()


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


def read_zone_local(frame) -> tuple[np.ndarray | None, str]:
    """Detect markers in frame and return (zone-local coord of frame center, status label), coord is None if unavailable."""
    height, width = frame.shape[:2]
    frame_center = (width / 2, height / 2)
    cv2.drawMarker(frame, (int(frame_center[0]), int(frame_center[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    corners, ids, _ = DETECTOR.detectMarkers(frame)
    if ids is None:
        return None, "no markers"
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    centers = marker_centers(corners, ids)
    homography = compute_homography(centers)
    if homography is None:
        missing = sorted(set(ZONE_CORNERS) - set(centers))
        return None, f"missing {missing}"
    point = np.array([[frame_center]], dtype=np.float32)
    zone_local = cv2.perspectiveTransform(point, homography)[0][0]
    return zone_local, f"({zone_local[0]:.2f}, {zone_local[1]:.2f})"


def open_source(source: str) -> cv2.VideoCapture:
    """Open a camera by index."""
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def main() -> None:
    """Open both cameras and show each one's live zone-local reading side by side."""
    args = parse_args()
    cap_a = open_source(args.source_a)
    cap_b = open_source(args.source_b)
    if not cap_a.isOpened():
        raise RuntimeError(f"Could not open camera A at source: {args.source_a}")
    if not cap_b.isOpened():
        raise RuntimeError(f"Could not open camera B at source: {args.source_b}")

    print("Both cameras opened. Press 'q' to quit.")
    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break

            coord_a, label_a = read_zone_local(frame_a)
            coord_b, label_b = read_zone_local(frame_b)
            cv2.putText(frame_a, f"A: {label_a}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_b, f"B: {label_b}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if coord_a is not None and coord_b is not None:
                distance = float(np.linalg.norm(coord_a - coord_b))
                print(f"A={coord_a.round(3).tolist()}  B={coord_b.round(3).tolist()}  distance={distance:.3f}")

            cv2.imshow("Camera A", frame_a)
            cv2.imshow("Camera B", frame_b)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap_a.release()
        cap_b.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
