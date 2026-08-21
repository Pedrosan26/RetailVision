"""
aruco_zone_test.py

Standalone spike: opens the camera, detects ArUco markers every frame,
and draws each one's outline, ID, and center point live -- no zone
polygon or face-detection logic yet. This is deliberately the smallest
possible test to validate marker detection behavior (reliability at
distance, angle, lighting, partial occlusion) with real printed markers
in hand before building anything on top of it. Not part of the pipeline.

Usage: PYTHONPATH=. ./venv/bin/python3 scripts/aruco_zone_test.py
Press 'q' to quit.
"""

import cv2

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())


def draw_markers(frame, corners: list, ids) -> None:
    """Draw each detected marker's outline, ID label, and center point onto the frame."""
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        center = marker_corners[0].mean(axis=0).astype(int)
        cv2.circle(frame, tuple(center), 4, (0, 0, 255), -1)
        cv2.putText(
            frame, f"id={marker_id}", (center[0] + 8, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )


def main() -> None:
    """Open the camera and show live marker detections until 'q' is pressed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    print("Camera opened. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            corners, ids, _ = DETECTOR.detectMarkers(frame)
            if ids is not None:
                draw_markers(frame, corners, ids)
                detected = sorted(int(i) for i in ids.flatten())
                cv2.putText(
                    frame, f"Detected: {detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            else:
                cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("ArUco zone test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
