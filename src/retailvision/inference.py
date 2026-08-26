"""
inference.py

Unified real-time inference pipeline: for each camera frame, detects faces
with the fine-tuned YOLOv8 FaceDetector, then classifies age group,
gender, and emotion for every detected face using the fine-tuned YOLOv8
classifiers (models/age_gender/final_age.pt, final_gender.pt,
models/emotion/final.pt).

Design note on bounding-box matching: age/gender and emotion predictions
are generated from the exact same detected crop, since one shared detector
feeds all three classification heads. There is nothing to reconcile across
models because they never produce independent boxes for the same face --
IoU-based matching only becomes necessary if a future detector swap moves
to per-model detection passes.

Emotion output collapses the classifier's angry/disgust/fear/sad classes
into a single "negative" label -- happy/neutral/surprise pass through
unchanged. The classifier itself is untouched (still 7-way internally);
this is a post-hoc remap of its output. See
RESULTS.md for the data behind this decision.
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .detection import FaceDetector
from .person_detection import PersonDetector, face_owner, floor_pixel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

WEIGHTS = {
    "age": REPO_ROOT / "models" / "age_gender" / "final_age.pt",
    "gender": REPO_ROOT / "models" / "age_gender" / "final_gender.pt",
    "emotion": REPO_ROOT / "models" / "emotion" / "final.pt",
}

NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}


def resolve_device() -> str:
    """Pick the fastest available torch device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InferencePipeline:
    """Detects faces and classifies age group, gender, and emotion per frame."""

    def __init__(self, device: str | None = None, track: bool = False, bodies: bool = False) -> None:
        """Load the face detector, the classifiers, and optionally the person detector.

        With track=True faces are associated across frames by the detector
        itself (ByteTrack) and every detection carries a track_id. With
        track=False each frame is detected independently and track_id is
        None, leaving identity to whatever the caller wires up.

        With bodies=True a person detector runs alongside, and identity
        moves from the face to the body: each detection reports the track
        of the person it belongs to, plus the pixel where that person meets
        the floor. A body is visible from any angle where a face is not, so
        the track survives a head turn, and its feet give a floor position
        that needs no assumed head height.
        """
        self.track = track
        self.device = device or resolve_device()
        self.detector = FaceDetector(device=self.device)
        self.people = PersonDetector(device=self.device) if bodies else None
        self.models: dict[str, YOLO] = {task: YOLO(str(path)) for task, path in WEIGHTS.items()}

    def _classify(self, task: str, crop: np.ndarray) -> tuple[str, float]:
        """Run one fine-tuned classifier on a face crop, returning (label, confidence)."""
        result = self.models[task].predict(source=crop, device=self.device, verbose=False)[0]
        predicted_index = int(result.probs.top1)
        return result.names[predicted_index], round(float(result.probs.top1conf), 4)

    def _classify_emotion(self, crop: np.ndarray) -> tuple[str, float]:
        """Classify emotion, collapsing angry/disgust/fear/sad into "negative" -- see RESULTS.md."""
        result = self.models["emotion"].predict(source=crop, device=self.device, verbose=False)[0]
        predicted_index = int(result.probs.top1)
        label = result.names[predicted_index]
        if label not in NEGATIVE_EMOTIONS:
            return label, round(float(result.probs.top1conf), 4)

        negative_prob = sum(
            float(result.probs.data[index]) for index, name in result.names.items() if name in NEGATIVE_EMOTIONS
        )
        return "negative", round(negative_prob, 4)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a BGR frame and classify age group, gender, and emotion for each.

        Each detection carries a track_id when this pipeline was built with
        track=True, and None otherwise. A track_id of None while tracking is
        on means ByteTrack has seen the face but not yet confirmed it as a
        track.

        With bodies enabled, track_id is the *person's* track rather than
        the face's, and two more keys appear: body_bbox, and floor_pixel --
        the point where that person meets the floor, or None when their
        feet are out of shot. A face inside no detected body keeps its own
        face track, so a detection is never dropped for want of a body.
        """
        detections = []
        found = self.detector.track(frame) if self.track else [(box, None) for box in self.detector.detect(frame)]

        people = self.people.track(frame) if self.people is not None else []
        frame_height = frame.shape[0]
        bodies = {track_id: box for box, track_id in people if track_id is not None}

        for (x, y, w, h), track_id in found:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            # Identity moves to the body where one contains this face; a face
            # in no body keeps its own track rather than being discarded.
            body_bbox = None
            contact = None
            if people:
                owner = face_owner((x, y, w, h), people)
                if owner is not None:
                    track_id = owner
                    body_bbox = bodies[owner]
                    contact = floor_pixel(body_bbox, frame_height)

            age_group, age_conf = self._classify("age", crop)
            gender, gender_conf = self._classify("gender", crop)
            emotion, emotion_conf = self._classify_emotion(crop)

            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "body_bbox": body_bbox,
                    "floor_pixel": contact,
                    "track_id": track_id,
                    "age_group": age_group,
                    "gender": gender,
                    "emotion": emotion,
                    "confidence": {
                        "age": age_conf,
                        "gender": gender_conf,
                        "emotion": emotion_conf,
                    },
                }
            )
        return detections