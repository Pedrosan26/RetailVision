"""
counter.py

Virtual line-crossing people counter: given per-track centroids from
CentroidTracker, detects when a track crosses a configurable line and
emits entry/exit events, maintains a net occupancy count, and tracks
per-track dwell time (elapsed time since that track's entry event).

Known limitation: a track that leaves the frame without recrossing the
line (e.g. exits out of camera view on the entry side) is never counted
as an exit, since there is only one line and no enclosing zone boundary
to detect that departure. This mirrors the ticket's explicit centroid
tracking scope -- a full zone-boundary tracker is documented future work
(see docs/future_work/).
"""

from typing import Literal

from .tracking import Centroid

Axis = Literal["x", "y"]
Direction = Literal["increasing", "decreasing"]


class LineCounter:
    """Detects virtual-line crossings from tracked centroids, maintaining net occupancy and per-track dwell time."""

    def __init__(self, axis: Axis, position: float, entry_direction: Direction) -> None:
        """Configure the virtual line's axis, position, and which crossing direction counts as an entry."""
        self.axis = axis
        self.position = position
        self.entry_direction = entry_direction
        self.count = 0
        self._sides: dict[int, bool] = {}
        self._entry_times: dict[int, float] = {}

    def _side(self, centroid: Centroid) -> bool:
        """True if the centroid is on the entry_direction's positive side of the line."""
        coordinate = centroid[0] if self.axis == "x" else centroid[1]
        if self.entry_direction == "increasing":
            return coordinate >= self.position
        return coordinate <= self.position

    def update(self, tracks: dict[int, Centroid], timestamp: float) -> list[tuple[int, str]]:
        """Check each active track for a line crossing, returning (track_id, 'entry' | 'exit') events."""
        events = []
        for track_id, centroid in tracks.items():
            side = self._side(centroid)
            previous_side = self._sides.get(track_id)
            self._sides[track_id] = side

            if previous_side is None:
                continue

            if side and not previous_side:
                self.count += 1
                self._entry_times[track_id] = timestamp
                events.append((track_id, "entry"))
            elif not side and previous_side:
                self.count = max(0, self.count - 1)
                self._entry_times.pop(track_id, None)
                events.append((track_id, "exit"))

        for track_id in set(self._sides) - set(tracks):
            del self._sides[track_id]

        return events

    def dwell_seconds(self, track_id: int, timestamp: float) -> float | None:
        """Return how long a track has been inside since its entry event, or None if it isn't currently inside."""
        entry_time = self._entry_times.get(track_id)
        if entry_time is None:
            return None
        return timestamp - entry_time
