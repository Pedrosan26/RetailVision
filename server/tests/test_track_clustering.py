"""
test_track_clustering.py

Tests for grouping tracks into people. Every historical count keys off
this, so the failure modes matter in both directions: merging two people
undercounts a room, and failing to merge one person across cameras is the
double-counting the whole exercise exists to remove.
"""

import unittest
from datetime import datetime, timedelta, timezone

from app.dedup import TrackKey, TrackPath, cluster_tracks, looks_like, paired_distances, same_person

START = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)


def walk(camera: str, track: str, start_at: float, seconds: int, x: float, y: float, drift: float = 0.0):
    """A track standing near (x, y), sampled once a second, optionally drifting in x."""
    return TrackPath(
        key=TrackKey(camera, track),
        points=[
            (START + timedelta(seconds=start_at + i), x + drift * i, y)
            for i in range(seconds)
        ],
    )


class PairedDistanceTests(unittest.TestCase):
    def test_points_are_paired_by_nearest_moment(self):
        """Positions are compared at the same instant, not by list position."""
        a = TrackPath(TrackKey("cam-a", "1"), [(START, 0.0, 0.0), (START + timedelta(seconds=1), 5.0, 0.0)])
        b = TrackPath(TrackKey("cam-b", "1"), [(START, 0.0, 0.0), (START + timedelta(seconds=1), 5.0, 0.0)])
        self.assertEqual(paired_distances(a, b), [0.0, 0.0])

    def test_points_too_far_apart_in_time_are_not_paired(self):
        """A sighting with no near-simultaneous counterpart contributes no distance."""
        a = TrackPath(TrackKey("cam-a", "1"), [(START, 0.0, 0.0)])
        b = TrackPath(TrackKey("cam-b", "1"), [(START + timedelta(seconds=30), 0.0, 0.0)])
        self.assertEqual(paired_distances(a, b), [])


class SamePersonTests(unittest.TestCase):
    def test_two_cameras_watching_one_person_match(self):
        """Tracks holding the same world position while both are visible are one person."""
        self.assertTrue(same_person(walk("cam-a", "1", 0, 10, 2.0, 3.0), walk("cam-b", "7", 0, 10, 2.2, 3.1)))

    def test_one_camera_never_merges_with_itself(self):
        """Two tracks on the same camera are two people, however close, or a group would undercount."""
        self.assertFalse(same_person(walk("cam-a", "1", 0, 10, 2.0, 3.0), walk("cam-a", "2", 0, 10, 2.0, 3.0)))

    def test_people_standing_apart_do_not_match(self):
        """Sustained separation beyond the merge radius keeps two people apart."""
        self.assertFalse(same_person(walk("cam-a", "1", 0, 10, 0.0, 0.0), walk("cam-b", "7", 0, 10, 6.0, 0.0)))

    def test_a_brief_crossing_does_not_merge_two_people(self):
        """Two people passing each other are close for a moment; that is not evidence.

        This is the failure that would matter most in a busy room, where
        people pass one another constantly.
        """
        crossing = TrackPath(
            TrackKey("cam-b", "7"),
            [(START + timedelta(seconds=i), 0.0, 0.0) for i in range(1)],
        )
        self.assertFalse(same_person(walk("cam-a", "1", 0, 10, 0.0, 0.0), crossing))

    def test_a_few_bad_samples_do_not_break_a_real_match(self):
        """Occlusion spikes the position for a moment; the median ignores them where a mean would not."""
        noisy = walk("cam-b", "7", 0, 12, 2.0, 3.0)
        noisy.points[3] = (noisy.points[3][0], 40.0, 40.0)
        noisy.points[8] = (noisy.points[8][0], -30.0, 15.0)
        self.assertTrue(same_person(walk("cam-a", "1", 0, 12, 2.0, 3.0), noisy))


class ClusterTests(unittest.TestCase):
    def test_three_cameras_on_one_person_give_one_person(self):
        """Linking is transitive, so a third camera joins the group even via one pairing."""
        paths = [
            walk("cam-a", "1", 0, 10, 2.0, 3.0),
            walk("cam-b", "7", 0, 10, 2.2, 3.1),
            walk("cam-c", "4", 0, 10, 1.9, 2.9),
        ]
        self.assertEqual(len(set(cluster_tracks(paths).values())), 1)

    def test_two_people_stay_two_people(self):
        """Two people each seen by two cameras resolve to two, not one and not four."""
        paths = [
            walk("cam-a", "1", 0, 10, 0.0, 0.0),
            walk("cam-b", "7", 0, 10, 0.1, 0.1),
            walk("cam-a", "2", 0, 10, 8.0, 8.0),
            walk("cam-b", "8", 0, 10, 8.1, 7.9),
        ]
        clusters = cluster_tracks(paths)
        self.assertEqual(len(set(clusters.values())), 2)
        self.assertEqual(clusters[TrackKey("cam-a", "1")], clusters[TrackKey("cam-b", "7")])
        self.assertEqual(clusters[TrackKey("cam-a", "2")], clusters[TrackKey("cam-b", "8")])

    def test_tracks_that_never_overlap_stay_separate(self):
        """Someone seen by one camera then another later has no shared moment to match on.

        Linking that needs appearance, which this deliberately does not use.
        """
        paths = [walk("cam-a", "1", 0, 10, 2.0, 3.0), walk("cam-b", "7", 600, 10, 2.0, 3.0)]
        self.assertEqual(len(set(cluster_tracks(paths).values())), 2)

    def test_person_id_does_not_depend_on_input_order(self):
        """The same tracks name the same person however the query returned them."""
        paths = [
            walk("cam-a", "1", 0, 10, 2.0, 3.0),
            walk("cam-b", "7", 0, 10, 2.2, 3.1),
            walk("cam-c", "4", 0, 10, 1.9, 2.9),
        ]
        forward = set(cluster_tracks(paths).values())
        backward = set(cluster_tracks(list(reversed(paths))).values())
        self.assertEqual(forward, backward)

    def test_a_lone_track_is_its_own_person(self):
        """One camera seeing someone nobody else saw still counts them once."""
        clusters = cluster_tracks([walk("cam-a", "1", 0, 10, 2.0, 3.0)])
        self.assertEqual(len(set(clusters.values())), 1)

    def test_no_tracks_at_all(self):
        """An empty range produces no people rather than an error."""
        self.assertEqual(cluster_tracks([]), {})



class AppearanceClusterTests(unittest.TestCase):
    """Appearance is consulted only where geometry cannot speak, and never overrules it."""

    SAME = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ALSO_SAME = [0.98, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    DIFFERENT = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_tracks_that_never_overlap_merge_when_they_look_alike(self):
        """Someone leaving one camera and entering another is linked by appearance.

        This is the case position provably cannot reach, and the only
        reason appearance is collected at all.
        """
        early, late = walk("cam-a", "1", 0, 10, 2.0, 3.0), walk("cam-b", "7", 600, 10, 9.0, 9.0)
        clusters = cluster_tracks(
            [early, late],
            appearances={early.key: self.SAME, late.key: self.ALSO_SAME},
        )
        self.assertEqual(len(set(clusters.values())), 1)

    def test_tracks_that_never_overlap_stay_apart_when_they_look_different(self):
        """Two people passing through in turn are still two people."""
        early, late = walk("cam-a", "1", 0, 10, 2.0, 3.0), walk("cam-b", "7", 600, 10, 2.0, 3.0)
        clusters = cluster_tracks(
            [early, late],
            appearances={early.key: self.SAME, late.key: self.DIFFERENT},
        )
        self.assertEqual(len(set(clusters.values())), 2)

    def test_appearance_cannot_override_a_position_disagreement(self):
        """Two people in similar clothing, standing apart while both cameras watch, stay two.

        The guard that matters most: letting appearance win here would
        merge distinct people and corrupt every count built on this.
        """
        left, right = walk("cam-a", "1", 0, 10, 0.0, 0.0), walk("cam-b", "7", 0, 10, 8.0, 8.0)
        clusters = cluster_tracks(
            [left, right],
            appearances={left.key: self.SAME, right.key: self.ALSO_SAME},
        )
        self.assertEqual(len(set(clusters.values())), 2)

    def test_appearance_never_merges_tracks_from_one_camera(self):
        """One camera reporting two tracks is reporting two people, however alike they look."""
        a, b = walk("cam-a", "1", 0, 10, 0.0, 0.0), walk("cam-a", "2", 600, 10, 0.0, 0.0)
        clusters = cluster_tracks([a, b], appearances={a.key: self.SAME, b.key: self.ALSO_SAME})
        self.assertEqual(len(set(clusters.values())), 2)

    def test_missing_appearances_fall_back_to_geometry_alone(self):
        """A node not sending appearances still gets position-based merging, and no crash."""
        left, right = walk("cam-a", "1", 0, 10, 2.0, 3.0), walk("cam-b", "7", 0, 10, 2.1, 3.1)
        self.assertEqual(len(set(cluster_tracks([left, right], appearances={}).values())), 1)

if __name__ == "__main__":
    unittest.main()
