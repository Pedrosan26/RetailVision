"""
tracking.py

Lightweight multi-object tracking for the people counter: assigns a stable
track ID to each detected face across frames using centroid distance,
matched optimally per frame with the Hungarian algorithm. This is
deliberately simpler than a full tracker like ByteTrack (no motion model,
no re-identification after a long occlusion) -- centroid tracking is
sufficient for a single virtual-line crossing counter; ByteTrack is
documented future work (see docs/future_work/) for when multi-zone
polygon tracking is actually needed.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

Bbox = tuple[int, int, int, int]
Centroid = tuple[float, float]


def bbox_centroid(bbox: Bbox) -> Centroid:
    """Compute the (x, y) center point of an (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


class CentroidTracker:
    """Assigns stable track IDs to bounding boxes across frames via nearest-centroid matching."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 75.0) -> None:
        """Set how many frames a track survives without a match, and the max match distance in pixels."""
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id = 0
        self.objects: dict[int, Centroid] = {}
        self._disappeared: dict[int, int] = {}

    def _register(self, centroid: Centroid) -> int:
        """Start tracking a new centroid under a fresh track ID, returning that ID."""
        track_id = self._next_id
        self._next_id += 1
        self.objects[track_id] = centroid
        self._disappeared[track_id] = 0
        return track_id

    def _deregister(self, track_id: int) -> None:
        """Stop tracking a track ID that has disappeared for too long."""
        del self.objects[track_id]
        del self._disappeared[track_id]

    def update(self, bboxes: list[Bbox]) -> list[int]:
        """Match this frame's bboxes against existing tracks, returning one track ID per input bbox (same order)."""
        centroids = [bbox_centroid(b) for b in bboxes]

        if not self.objects:
            return [self._register(c) for c in centroids]

        track_ids = list(self.objects.keys())

        if not centroids:
            for track_id in track_ids:
                self._disappeared[track_id] += 1
                if self._disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return []

        track_centroids = np.array([self.objects[t] for t in track_ids])
        input_centroids = np.array(centroids)
        distances = cdist(track_centroids, input_centroids)

        row_indices, col_indices = linear_sum_assignment(distances)

        assigned_tracks: set[int] = set()
        assigned_inputs: set[int] = set()
        result: list[int | None] = [None] * len(centroids)

        for row, col in zip(row_indices, col_indices):
            if distances[row, col] > self.max_distance:
                continue
            track_id = track_ids[row]
            self.objects[track_id] = centroids[col]
            self._disappeared[track_id] = 0
            result[col] = track_id
            assigned_tracks.add(track_id)
            assigned_inputs.add(col)

        for row, track_id in enumerate(track_ids):
            if track_id in assigned_tracks:
                continue
            self._disappeared[track_id] += 1
            if self._disappeared[track_id] > self.max_disappeared:
                self._deregister(track_id)

        for col, centroid in enumerate(centroids):
            if col not in assigned_inputs:
                result[col] = self._register(centroid)

        return result
