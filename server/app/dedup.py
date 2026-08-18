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
