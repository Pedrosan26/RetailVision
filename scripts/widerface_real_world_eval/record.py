"""
record.py

Records a live camera session to a video file, one per condition, with
no model inference at capture time -- keeping the recording
model-agnostic so it can be replayed through both the baseline and
fine-tuned checkpoints afterward for a fair, frame-identical comparison.
"""

from pathlib import Path

import cv2

from .constants import VIDEO_DIR


def record_session(condition: str) -> Path:
    """Open the camera and save raw frames to a video file until 'q' is pressed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{condition}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    print(f"Recording condition '{condition}'. Press 'q' to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            writer.write(frame)
            frame_count += 1

            cv2.imshow(f"RetailVision - recording {condition}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Recorded {frame_count} frames to {out_path}")
    return out_path
