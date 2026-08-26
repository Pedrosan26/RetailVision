"""
dedup.py

Turns detections reported by several cameras into a headcount of actual
people.

Cameras covering one area see each other's subjects, so summing their
counts double-counts anyone visible to more than one of them -- with three
cameras on a single room, that is most people. What makes the correct
answer reachable is that every camera watching a zone reports positions in
the same world frame (see the pipeline's marker map), so "is this the same
person" reduces to "are these two positions close together".

Two decisions worth stating, because both trade one kind of error for
another:

Only detections from *different* cameras are merged. Two people genuinely
standing close together are reported by the same camera as two separate
detections, and collapsing those would undercount a queue or a group --
the exact situations occupancy is most often used to measure. One camera's
own frame is therefore taken as ground truth for how many people it sees.

Each camera contributes only its most recent frame. Positions from a
second ago belong to a person who has since moved, so pooling several
frames would smear one person into a short trail and count them repeatedly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

# Two detections from different cameras closer than this are treated as one
# person. It has to exceed the position error, which is dominated by how far
# each subject's real head height differs from the assumed one -- around half a
# metre for a camera roughly a metre above the assumed plane. Set it too tight
# and one person is counted twice; too loose and two people standing together
# merge into one.
DEFAULT_MERGE_RADIUS_METERS = 1.0

# How far back a camera's "current frame" reaches. Detections from one frame
# share a timestamp to within a few milliseconds, but frames arrive batched and
# clocks differ slightly between nodes, so this is generous enough to catch a
# whole frame without pulling in the one before it.
DEFAULT_FRAME_WINDOW = timedelta(milliseconds=400)


# Track clustering, below, answers a different question to the headcount above:
# not "how many people are here now" but "which of these tracks are the same
# person", which is what every historical count needs and none of them had.

# Two tracks must overlap in time by at least this much before their positions
# are compared at all. A brief overlap is not evidence: two people passing each
# other are close for a moment, and a hundred milliseconds of proximity would
# merge them.
DEFAULT_MIN_OVERLAP = timedelta(seconds=2)

# Positions are compared at paired moments no further apart than this. Cameras
# do not sample on a shared clock, so a point from one is matched to the
# nearest in time from the other, and anything further apart is not a pair.
DEFAULT_PAIRING_TOLERANCE = timedelta(milliseconds=500)

# How many paired samples a comparison needs before its median means anything.
MIN_PAIRED_SAMPLES = 4

# Cosine similarity above which two tracks with no shared moment are treated
# as the same person. A starting value, not a tuned one: setting it properly
# needs footage containing several different people, and the separation that
# matters -- between two people rather than two views of one -- has not been
# measured yet. See src/retailvision/appearance.py for what has.
DEFAULT_APPEARANCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class Observation:
    """One camera's sighting of a person at a world floor position."""

    camera_node_id: str
    timestamp: datetime
    world_x: float
    world_y: float


def latest_frame_per_camera(
    observations: Iterable[Observation], frame_window: timedelta = DEFAULT_FRAME_WINDOW
) -> dict[str, list[Observation]]:
    """Keep only each camera's most recent frame, so one person is not counted once per frame."""
    by_camera: dict[str, list[Observation]] = {}
    for observation in observations:
        by_camera.setdefault(observation.camera_node_id, []).append(observation)

    latest: dict[str, list[Observation]] = {}
    for camera_node_id, sightings in by_camera.items():
        newest = max(sighting.timestamp for sighting in sightings)
        latest[camera_node_id] = [s for s in sightings if newest - s.timestamp <= frame_window]
    return latest


def merge_across_cameras(
    frames: dict[str, list[Observation]], merge_radius: float = DEFAULT_MERGE_RADIUS_METERS
) -> int:
    """Count distinct people across cameras, merging sightings of one person into one.

    Cameras are folded in one at a time. Each of a camera's sightings either
    matches an already-counted person -- the nearest one within the merge radius
    that this camera has not already claimed -- or becomes a new person. The
    per-camera claim is what stops two people standing close together from
    collapsing: one camera reporting two detections always yields two people,
    however near each other they are.
    """
    people: list[tuple[float, float, set[str]]] = []

    for camera_node_id in sorted(frames):
        for sighting in frames[camera_node_id]:
            best_index, best_distance = None, merge_radius
            for index, (x, y, claimed_by) in enumerate(people):
                if camera_node_id in claimed_by:
                    continue
                distance = math.hypot(sighting.world_x - x, sighting.world_y - y)
                if distance <= best_distance:
                    best_index, best_distance = index, distance

            if best_index is None:
                people.append((sighting.world_x, sighting.world_y, {camera_node_id}))
            else:
                x, y, claimed_by = people[best_index]
                claimed_by.add(camera_node_id)
                # Average the two views rather than keeping the first, so a third
                # camera compares against the better of the available estimates.
                people[best_index] = ((x + sighting.world_x) / 2, (y + sighting.world_y) / 2, claimed_by)

    return len(people)


def deduplicated_headcount(
    observations: Sequence[Observation],
    merge_radius: float = DEFAULT_MERGE_RADIUS_METERS,
    frame_window: timedelta = DEFAULT_FRAME_WINDOW,
) -> tuple[int, dict[str, int]]:
    """Return the zone's deduplicated headcount and each camera's own count."""
    frames = latest_frame_per_camera(observations, frame_window)
    per_camera = {camera_node_id: len(sightings) for camera_node_id, sightings in frames.items()}
    return merge_across_cameras(frames, merge_radius), per_camera


@dataclass(frozen=True)
class TrackKey:
    """Identifies one track: a camera node and the track id it assigned."""

    camera_node_id: str
    track_id: str


@dataclass
class TrackPath:
    """Where one track was, over time, in the zone's shared world frame."""

    key: TrackKey
    points: list[tuple[datetime, float, float]]

    def sorted_points(self) -> list[tuple[datetime, float, float]]:
        """The path in time order, which the overlap arithmetic below assumes."""
        return sorted(self.points, key=lambda point: point[0])

    @property
    def span(self) -> tuple[datetime, datetime]:
        """First and last moment this track was seen."""
        times = [point[0] for point in self.points]
        return min(times), max(times)


def _median(values: list[float]) -> float:
    """Middle value of a non-empty list, averaging the two middles when even."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def paired_distances(
    left: TrackPath, right: TrackPath, tolerance: timedelta = DEFAULT_PAIRING_TOLERANCE
) -> list[float]:
    """Distances between two tracks at the moments both were seen.

    Cameras do not share a sampling clock, so each point on the shorter
    path is matched to the nearest point in time on the other and dropped
    if the nearest is still too far away. Comparing positions recorded at
    different moments would measure how fast someone walks, not whether
    two tracks are the same person.
    """
    other = right.sorted_points()
    if not other:
        return []

    distances = []
    for when, x, y in left.sorted_points():
        nearest = min(other, key=lambda point: abs(point[0] - when))
        if abs(nearest[0] - when) > tolerance:
            continue
        distances.append(math.hypot(x - nearest[1], y - nearest[2]))
    return distances


def same_person(
    left: TrackPath,
    right: TrackPath,
    merge_radius: float = DEFAULT_MERGE_RADIUS_METERS,
    min_overlap: timedelta = DEFAULT_MIN_OVERLAP,
    tolerance: timedelta = DEFAULT_PAIRING_TOLERANCE,
) -> bool:
    """Whether two tracks were the same person, judged by where they were while both were visible.

    Three conditions, each ruling out a way of being wrong:

    Different cameras. One camera reporting two tracks is reporting two
    people, however close together -- collapsing those would undercount a
    queue or a group, which is what occupancy is most used to measure.

    A real overlap. Two people passing each other are briefly in the same
    place, so a short coincidence proves nothing and the tracks must have
    been seen together for a sustained period.

    Sustained proximity, by median rather than mean. Position error spikes
    when someone is momentarily occluded or their feet leave the frame,
    and a mean lets a handful of those decide the answer.
    """
    if left.key.camera_node_id == right.key.camera_node_id:
        return False

    left_start, left_end = left.span
    right_start, right_end = right.span
    overlap = min(left_end, right_end) - max(left_start, right_start)
    if overlap < min_overlap:
        return False

    distances = paired_distances(left, right, tolerance)
    if len(distances) < MIN_PAIRED_SAMPLES:
        return False
    return _median(distances) <= merge_radius


def looks_like(
    left: TrackKey,
    right: TrackKey,
    appearances: dict[TrackKey, Sequence[float]],
    threshold: float = DEFAULT_APPEARANCE_THRESHOLD,
) -> bool:
    """Whether two tracks describe people who look the same, by cosine similarity.

    Only consulted where geometry has nothing to say -- two tracks that
    never overlapped in time. Where they did overlap, position is the
    better evidence by a wide margin: it cannot confuse two people in
    similar coats, and it does not care which way anyone was facing.
    """
    a, b = appearances.get(left), appearances.get(right)
    if a is None or b is None or len(a) != len(b):
        return False
    dot = sum(x * y for x, y in zip(a, b))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return norms > 0 and dot / norms >= threshold


def _overlaps(left: TrackPath, right: TrackPath, min_overlap: timedelta) -> bool:
    """Whether two tracks were both visible for long enough to compare positions at all."""
    left_start, left_end = left.span
    right_start, right_end = right.span
    return (min(left_end, right_end) - max(left_start, right_start)) >= min_overlap


def cluster_tracks(
    paths: Sequence[TrackPath],
    merge_radius: float = DEFAULT_MERGE_RADIUS_METERS,
    min_overlap: timedelta = DEFAULT_MIN_OVERLAP,
    tolerance: timedelta = DEFAULT_PAIRING_TOLERANCE,
    appearances: dict[TrackKey, Sequence[float]] | None = None,
    appearance_threshold: float = DEFAULT_APPEARANCE_THRESHOLD,
) -> dict[TrackKey, str]:
    """Group tracks into people, returning the person id each track belongs to.

    Two kinds of evidence, used where each is the stronger one. Position
    settles any pair that was visible to both cameras at once: it is
    unambiguous, and two people cannot occupy one spot. Appearance is
    consulted only for pairs with no shared moment -- someone leaving one
    camera and entering another -- which position provably cannot reach,
    and which is the whole reason appearance is collected.

    Never both, and in that order. Letting appearance override a position
    disagreement would merge two people in similar clothing standing
    apart, which is the failure that would quietly corrupt every count
    built on top of this.

    Union-find over the pairwise test, so linking is transitive: three
    cameras watching one person produce three tracks that all reach the
    same person even where one pair never matched directly.

    The person id is the smallest track key in the group rather than a
    counter, so it is stable -- the same query run twice, or run over a
    wider window, names the same person the same way instead of
    renumbering everyone.
    """
    parent: dict[TrackKey, TrackKey] = {path.key: path.key for path in paths}

    def find(key: TrackKey) -> TrackKey:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: TrackKey, b: TrackKey) -> None:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return
        # Keep the smaller key as the root so the group's name does not
        # depend on which order the pairs happened to be compared in.
        low, high = sorted([root_a, root_b], key=lambda k: (k.camera_node_id, k.track_id))
        parent[high] = low

    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left.key.camera_node_id == right.key.camera_node_id:
                continue
            if _overlaps(left, right, min_overlap):
                # Seen together: position decides, either way. A disagreement
                # here is two people, and appearance does not get a vote.
                if same_person(left, right, merge_radius, min_overlap, tolerance):
                    union(left.key, right.key)
            elif appearances and looks_like(left.key, right.key, appearances, appearance_threshold):
                union(left.key, right.key)

    return {path.key: f"{find(path.key).camera_node_id}:{find(path.key).track_id}" for path in paths}


def person_ids_for_events(
    events: Iterable,
    merge_radius: float = DEFAULT_MERGE_RADIUS_METERS,
    min_overlap: timedelta = DEFAULT_MIN_OVERLAP,
    appearances: dict[tuple[str, str], Sequence[float]] | None = None,
) -> dict[tuple[str, str], str]:
    """Map each (camera_node_id, track_id) in these events to the person it belongs to.

    The single place the endpoints agree on who a person is. Before this,
    each of them keyed identity on (camera, track) independently, which
    meant every historical figure counted one visitor once per camera that
    could see them.

    Events with no world position cannot be compared to anything, so they
    keep their own track as their person -- the pre-zone behaviour, applied
    only to rows that give no better option rather than to all of them.
    """
    by_track: dict[tuple[str, str], list[tuple[datetime, float, float]]] = {}
    unpositioned: set[tuple[str, str]] = set()

    for event in events:
        if event.track_id is None:
            continue
        key = (event.camera_node_id, event.track_id)
        if event.world_x is None or event.world_y is None:
            unpositioned.add(key)
            continue
        by_track.setdefault(key, []).append((event.timestamp, event.world_x, event.world_y))

    paths = [
        TrackPath(TrackKey(camera_node_id, track_id), points)
        for (camera_node_id, track_id), points in by_track.items()
    ]
    by_key = (
        {TrackKey(camera, track): vector for (camera, track), vector in appearances.items()}
        if appearances
        else None
    )
    clustered = cluster_tracks(paths, merge_radius, min_overlap, appearances=by_key)

    resolved = {(key.camera_node_id, key.track_id): person for key, person in clustered.items()}
    for camera_node_id, track_id in unpositioned:
        resolved.setdefault((camera_node_id, track_id), f"{camera_node_id}:{track_id}")
    return resolved
