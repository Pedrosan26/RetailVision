"""
person_detection.py

Detects and tracks whole people, as the identity anchor the rest of the
pipeline hangs off.

Faces were the original anchor because they are what the demographic
classifiers need. They make a poor identity, though: a face exists only
while someone is looking roughly at the camera, so a track ends every
time a person turns their head and a new one begins when they turn back.
A body is visible continuously, from any angle, and from behind.

The bigger reason is geometric. A face gives no fixed height to measure
from -- projecting one onto the floor needs an assumed head height, and
how wrong that assumption is has been the dominant source of position
error in this system. A person's box has their feet at the bottom edge,
and feet are on the floor at z = 0 by definition. That removes the
assumption rather than improving it: see floor_position() below.

This uses the stock COCO-pretrained YOLOv8n rather than a fine-tuned
checkpoint. Person is one of the classes it was trained on and is by far
its best-represented one, so it works out of the box. It is not tuned for
these cameras the way the face detector is, and the honest expectation is
that a ceiling-mounted view of people at 2-5m is well within what it
handles, while unusual angles are where it would first need fine-tuning.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = REPO_ROOT / "models" / "yolov8n.pt"
BYTETRACK_CONFIG = REPO_ROOT / "config" / "bytetrack.yaml"

# COCO class 0. Everything else the model can find -- cars, chairs, handbags --
# is filtered out at inference rather than after, so the tracker never sees it.
PERSON_CLASS = 0

Bbox = tuple[int, int, int, int]


class PersonDetector:
    """Detects people in a frame and follows them across frames with ByteTrack."""

    def __init__(self, device: str | None = None, confidence: float = 0.4) -> None:
        """Load the COCO detector and fix the confidence floor for a person detection.

        The floor is above the Ultralytics default: a spurious person is not
        a harmless extra box here, it becomes a track, a dwell timer and a
        row of analytics.
        """
        self._model = YOLO(str(WEIGHTS_PATH))
        self._device = device
        self._confidence = confidence

    def track(self, frame: np.ndarray) -> list[tuple[Bbox, int | None]]:
        """Return each detected person as (box, track_id), with track_id None until confirmed.

        Tracker state lives on the model between calls, so frames must arrive
        in capture order and one detector instance must serve one camera.
        """
        result = self._model.track(
            source=frame,
            device=self._device,
            persist=True,
            tracker=str(BYTETRACK_CONFIG),
            classes=[PERSON_CLASS],
            conf=self._confidence,
            verbose=False,
        )[0]
        boxes = [
            (int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in result.boxes.xyxy.tolist()
        ]
        if result.boxes.id is None:
            return [(box, None) for box in boxes]
        return list(zip(boxes, (int(track_id) for track_id in result.boxes.id.tolist())))


# How close a box's bottom edge may come to the bottom of the frame before
# its floor contact is treated as cropped rather than real. A couple of pixels
# of slack, since a detector rarely puts the edge exactly on the boundary.
FRAME_EDGE_MARGIN_PIXELS = 4


def floor_pixel(person_bbox: Bbox, frame_height: int | None = None) -> tuple[float, float] | None:
    """Return the pixel where this person meets the floor, or None if their feet are not in shot.

    This is the whole reason to detect bodies rather than faces for
    position. A face has to be projected onto a plane at some assumed
    height, and every centimetre that assumption is wrong pushes the
    reported position along the viewing ray. Feet need no assumption --
    they are on the floor, so the ray is intersected with z = 0.

    That only holds while the feet are actually visible. A box whose
    bottom edge sits on the bottom of the frame has been cropped, and its
    lowest visible pixel is somebody's knees; projecting that onto the
    floor puts them metres further away than they are, confidently and
    silently. Passing frame_height turns that case into None, which a
    caller can drop, rather than into a plausible wrong coordinate that
    would go on to pollute the heatmap and the zone counts.

    Feet hidden behind a shelf are the same error and cannot be detected
    this way. That one is bounded by how much of the person is hidden,
    where the head-height error it replaces was bounded by nothing better
    than how much people differ in height.
    """
    x, y, width, height = person_bbox
    bottom = y + height
    if frame_height is not None and bottom >= frame_height - FRAME_EDGE_MARGIN_PIXELS:
        return None
    return (x + width / 2.0, float(bottom))


def face_owner(face_bbox: Bbox, people: list[tuple[Bbox, int | None]]) -> int | None:
    """Return the track_id of the person a face belongs to, or None if it sits in nobody.

    A face belongs to whichever person's box contains its centre. Where
    boxes overlap -- someone standing behind someone else -- the smaller
    box wins, because a face inside two boxes is far more likely to belong
    to the nearer, tighter one than to the larger box behind it.
    """
    fx, fy, fw, fh = face_bbox
    centre = (fx + fw / 2.0, fy + fh / 2.0)

    best_id, best_area = None, None
    for (px, py, pw, ph), track_id in people:
        if track_id is None:
            continue
        if not (px <= centre[0] <= px + pw and py <= centre[1] <= py + ph):
            continue
        area = pw * ph
        if best_area is None or area < best_area:
            best_id, best_area = track_id, area
    return best_id
