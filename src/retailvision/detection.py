"""
detection.py

Handles face detection for the RetailVision pipeline. This is the first
detection stage: it takes a single camera frame and returns the bounding
boxes of any faces found in it, using a YOLOv8 detection-mode model
trained from scratch on WIDER FACE and fine-tuned for retail camera
conditions (see RESULTS.md for the training results and for the
real-world evaluation that motivated
replacing this stage's original Haar cascade implementation).
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = REPO_ROOT / "models" / "face_detection" / "final.pt"
BYTETRACK_CONFIG = REPO_ROOT / "config" / "bytetrack.yaml"


# Ultralytics defaults to 0.25, which is too permissive here. At that floor the
# detector fires on the back of a head, and nothing downstream can tell that it
# should not have: the age, gender and emotion classifiers are handed a crop and
# will confidently describe whatever is in it, so a false face becomes a person
# with a mood. The classifiers cannot reject their input, which means this
# threshold is the only place that judgement can be made.
DEFAULT_CONFIDENCE = 0.5


class FaceDetector:
    def __init__(self, device: str | None = None, confidence: float = DEFAULT_CONFIDENCE) -> None:
        """Load the fine-tuned YOLOv8 face-detection checkpoint onto the given device."""
        self._model = YOLO(str(WEIGHTS_PATH))
        self._device = device
        self._confidence = confidence

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a BGR frame, returning (x, y, w, h) boxes."""
        result = self._model.predict(source=frame, device=self._device, conf=self._confidence, verbose=False)[0]
        return _boxes_from(result)

    def track(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], int | None, float]]:
        """Detect faces and track them across frames, returning (box, track_id, confidence) triples.

        Confidence is carried out rather than discarded because callers need
        it to choose between competing detections -- see the one-face-per-body
        rule in inference.py, which is the only thing that removes a confident
        detection of the back of somebody's head.

        Uses ByteTrack (see config/bytetrack.yaml), which unlike a
        detect-then-match tracker has the detector's confidence scores to
        work with: it matches confident detections first, then tries the
        low-confidence leftovers against tracks still unmatched rather than
        throwing them away. A face that turns or blurs loses confidence
        before it disappears, so that second pass is what keeps it on one
        track instead of ending one and starting another.

        track_id is None for a detection ByteTrack has not yet confirmed as
        a track. Callers decide what to do with those; this returns them
        rather than hiding a detection the model did make.

        State lives on the model between calls, so frames must be passed in
        capture order and one detector instance must serve one camera.
        """
        result = self._model.track(
            source=frame,
            device=self._device,
            persist=True,
            tracker=str(BYTETRACK_CONFIG),
            conf=self._confidence,
            verbose=False,
        )[0]
        boxes = _boxes_from(result)
        scores = _confidences_from(result)
        if result.boxes.id is None:
            return [(box, None, score) for box, score in zip(boxes, scores)]
        ids = (int(track_id) for track_id in result.boxes.id.tolist())
        return list(zip(boxes, ids, scores))


def _boxes_from(result) -> list[tuple[int, int, int, int]]:
    """Convert a YOLO result's xyxy boxes into (x, y, w, h) integer tuples."""
    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in result.boxes.xyxy.tolist()]


def _confidences_from(result) -> list[float]:
    """The detector's confidence for each box, in the same order."""
    if result.boxes.conf is None:
        return [1.0] * len(result.boxes.xyxy.tolist())
    return [float(c) for c in result.boxes.conf.tolist()]
