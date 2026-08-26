"""
appearance.py

Turns a person's body crop into a vector that can be compared against
another camera's view of them.

This is what geometry cannot do. Two cameras watching the same room can
recognise one person from position alone, because both report into the
same world frame -- but only while they see them at the same moment.
Someone who leaves one camera's view and enters another's a few seconds
later shares no position to match on, and appearance is the only evidence
left.

What it encodes is clothing, build and colouring, not identity in any
durable sense: change your jacket and you are a new person to it. That
bounds what it can do -- continuity within a visit, not recognition of a
returning customer -- and it also bounds the harm, which is why body
appearance was chosen over face embeddings here.

On the backbone, because the first attempt failed and the failure is
worth keeping. A torchvision ResNet18 pretrained on ImageNet was tried
first and does not work for this at all: on real crops from the
evaluation clips it scored two different views of one person at 0.92,
indistinguishable from two views that genuinely matched. It was never
trained to tell people apart, and it does not.

The encoder used now is one Ultralytics ships for re-identification and
downloads on first use. On the same crops:

    same person, consistent view    cosine 0.704 - 0.973  (mean 0.893)
    same person, different footage  cosine -0.030 - 0.270 (mean 0.090)

That is a real gap where there was none. Note what it does *not* say: the
separation that finally matters is between two different people, and
every clip in this repository shows one person, so that has still never
been measured here. The threshold below is therefore a starting value,
not a tuned one, and the first footage containing several people should
be used to set it properly.

Embeddings are averaged over a track rather than taken from one frame. A
single crop catches whatever the person happened to be doing -- turned
away, half-occluded, motion-blurred -- and the running mean over a track
is both steadier and cheaper to compare than a set of vectors.
"""

from __future__ import annotations

import numpy as np

# The re-identification encoder Ultralytics publishes, fetched on first use.
# The "n" size is the smallest; the s/m/l/x variants are drop-in replacements
# if matching proves too loose and the frame budget can afford them.
DEFAULT_REID_MODEL = "yolo26n-reid.onnx"

# Cosine similarity above which two track appearances are treated as the same
# person. A starting value, chosen to sit well clear of the different-footage
# band measured above while staying below the consistent-view one. It has not
# been tuned against footage of several different people, because none exists
# yet -- doing so is the single most valuable thing that footage would buy.
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# Below this many pixels on a side, a crop carries no usable appearance and
# embedding it would add noise to a track's running mean rather than evidence.
MIN_CROP_PIXELS = 32


class AppearanceEmbedder:
    """Produces comparable appearance vectors for the people detected in a frame."""

    def __init__(self, model: str = DEFAULT_REID_MODEL, device: str | None = None) -> None:
        """Load the re-identification encoder once, downloading it if this is the first run."""
        from ultralytics.trackers.utils.reid import ReID

        self._encoder = ReID(model, device=device)

    def embed_boxes(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray | None]:
        """Return one L2-normalised appearance vector per box, or None where a box is unusable.

        Takes the whole frame and every box at once because the encoder
        batches them, which is a good deal cheaper than one call per person.
        Normalised so comparing two vectors is a dot product and averaging
        several keeps them on the same scale.
        """
        if not boxes:
            return []

        usable = [min(width, height) >= MIN_CROP_PIXELS for _, _, width, height in boxes]
        # The encoder wants centre-form boxes; detections here are top-left form.
        centred = np.array(
            [[x + width / 2, y + height / 2, width, height] for x, y, width, height in boxes],
            dtype=np.float32,
        )

        vectors: list[np.ndarray | None] = []
        for raw, is_usable in zip(self._encoder(frame, centred), usable):
            if raw is None or not is_usable:
                vectors.append(None)
                continue
            vector = np.asarray(raw, dtype=np.float32).ravel()
            norm = np.linalg.norm(vector)
            vectors.append(None if norm == 0 else vector / norm)
        return vectors


class TrackAppearance:
    """Keeps one running appearance vector per track, averaged over the frames seen so far."""

    def __init__(self) -> None:
        """Start with no tracks described."""
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

    def observe(self, track_id: int, vector: np.ndarray | None) -> None:
        """Fold one more view of a track into its running appearance, ignoring unusable views."""
        if vector is None:
            return
        if track_id in self._sums:
            self._sums[track_id] += vector
            self._counts[track_id] += 1
        else:
            self._sums[track_id] = vector.copy()
            self._counts[track_id] = 1

    def embedding(self, track_id: int) -> np.ndarray | None:
        """Return the track's mean appearance, re-normalised, or None if it has none yet."""
        total = self._sums.get(track_id)
        if total is None:
            return None
        mean = total / self._counts[track_id]
        norm = np.linalg.norm(mean)
        return None if norm == 0 else (mean / norm).astype(np.float32)

    def observations(self, track_id: int) -> int:
        """How many views went into this track's appearance, so a caller can weigh how settled it is."""
        return self._counts.get(track_id, 0)

    def retire(self, active: set[int]) -> None:
        """Forget tracks no longer being reported, so memory follows the room rather than the uptime."""
        for track_id in list(self._sums):
            if track_id not in active:
                del self._sums[track_id]
                del self._counts[track_id]


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity between two normalised appearance vectors."""
    return float(np.dot(left, right))
