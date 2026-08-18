"""
test_dedup.py

Tests for cross-camera deduplication: that one person seen by several
cameras is counted once, and -- just as important -- that two people
standing close together are not collapsed into one.
"""

from datetime import datetime, timedelta, timezone

from app.dedup import (
    DEFAULT_MERGE_RADIUS_METERS,
    Observation,
    deduplicated_headcount,
    latest_frame_per_camera,
    merge_across_cameras,
)

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


def sighting(camera: str, x: float, y: float, offset_ms: int = 0) -> Observation:
    """Build one camera sighting at a world position, optionally offset back in time."""
    return Observation(
        camera_node_id=camera,
        timestamp=NOW - timedelta(milliseconds=offset_ms),
        world_x=x,
        world_y=y,
    )


def test_one_person_seen_by_three_cameras_counts_once():
    """The whole point: three cameras watching one person report one person."""
    observations = [sighting("cam0", 2.0, 1.5), sighting("cam1", 2.2, 1.6), sighting("cam2", 1.9, 1.4)]
    total, per_camera = deduplicated_headcount(observations)
    assert total == 1
    assert per_camera == {"cam0": 1, "cam1": 1, "cam2": 1}


def test_people_far_apart_are_counted_separately():
    """Two cameras each seeing a different person report two people."""
    total, _ = deduplicated_headcount([sighting("cam0", 0.0, 0.0), sighting("cam1", 6.0, 4.0)])
    assert total == 2


def test_two_people_close_together_on_one_camera_are_not_merged():
    """A single camera's own detections are ground truth, so a queue is not collapsed."""
    total, per_camera = deduplicated_headcount([sighting("cam0", 2.0, 1.5), sighting("cam0", 2.1, 1.55)])
    assert total == 2
    assert per_camera == {"cam0": 2}


def test_a_camera_cannot_claim_the_same_person_twice():
    """Each camera contributes at most one sighting per person, so counts cannot collapse below its own."""
    observations = [
        sighting("cam0", 2.0, 1.5),
        sighting("cam0", 2.2, 1.5),
        sighting("cam1", 2.1, 1.5),
    ]
    total, _ = deduplicated_headcount(observations)
    assert total == 2


def test_partial_overlap_counts_the_union():
    """Cameras covering different parts of a zone contribute everyone, merging only the shared person."""
    observations = [
        sighting("cam0", 0.0, 0.0),
        sighting("cam0", 3.0, 2.0),
        sighting("cam1", 3.1, 2.1),
        sighting("cam1", 6.0, 4.0),
    ]
    total, per_camera = deduplicated_headcount(observations)
    assert total == 3
    assert per_camera == {"cam0": 2, "cam1": 2}


def test_only_the_latest_frame_per_camera_is_used():
    """Older positions belong to a person who has moved, so pooling frames would count a trail."""
    observations = [
        sighting("cam0", 2.0, 1.5),
        sighting("cam0", 1.0, 1.0, offset_ms=900),
        sighting("cam0", 0.0, 0.5, offset_ms=1800),
    ]
    total, per_camera = deduplicated_headcount(observations)
    assert total == 1
    assert per_camera == {"cam0": 1}


def test_frames_are_kept_per_camera_not_globally():
    """A camera lagging slightly behind still contributes, rather than being dropped as stale."""
    frames = latest_frame_per_camera([sighting("cam0", 2.0, 1.5), sighting("cam1", 2.1, 1.6, offset_ms=1500)])
    assert set(frames) == {"cam0", "cam1"}
    assert len(frames["cam1"]) == 1


def test_merge_radius_is_respected_at_its_boundary():
    """Just inside the radius merges, just outside does not."""
    inside = [sighting("cam0", 0.0, 0.0), sighting("cam1", DEFAULT_MERGE_RADIUS_METERS * 0.9, 0.0)]
    outside = [sighting("cam0", 0.0, 0.0), sighting("cam1", DEFAULT_MERGE_RADIUS_METERS * 1.1, 0.0)]
    assert merge_across_cameras(latest_frame_per_camera(inside)) == 1
    assert merge_across_cameras(latest_frame_per_camera(outside)) == 2


def test_no_observations_is_an_empty_zone():
    """A zone nobody is standing in reports zero rather than failing."""
    assert deduplicated_headcount([]) == (0, {})


def test_a_single_camera_reports_its_own_count():
    """With one camera there is nothing to merge and the answer is that camera's count."""
    total, per_camera = deduplicated_headcount(
        [sighting("cam0", 0.0, 0.0), sighting("cam0", 3.0, 0.0), sighting("cam0", 6.0, 0.0)]
    )
    assert total == 3
    assert per_camera == {"cam0": 3}
