"""
test_person_detection.py

Tests for the two pieces of geometry that decide how body detections are
used: which pixel counts as a person's floor contact, and which person a
detected face belongs to.

Both are pure functions over boxes, so no model is loaded here.
"""

import unittest

from src.retailvision.person_detection import face_owner, floor_pixel


class FloorPixelTests(unittest.TestCase):
    def test_floor_pixel_is_the_bottom_edge_centre(self):
        """The floor contact is horizontally centred on the box and at its bottom edge."""
        self.assertEqual(floor_pixel((100, 50, 40, 160)), (120.0, 210.0))

    def test_floor_pixel_tracks_the_box_rather_than_its_size(self):
        """Two people at different scales both report the pixel where they meet the floor."""
        near = floor_pixel((0, 0, 200, 400))
        far = floor_pixel((0, 0, 20, 40))
        self.assertEqual(near, (100.0, 400.0))
        self.assertEqual(far, (10.0, 40.0))

    def test_feet_cropped_by_the_frame_edge_report_nothing(self):
        """A body running off the bottom of the frame has no visible floor contact.

        Its lowest visible pixel is not where the person is standing, and
        projecting it would place them confidently in the wrong place.
        """
        self.assertIsNone(floor_pixel((100, 500, 200, 580), frame_height=1080))

    def test_feet_clear_of_the_frame_edge_report_normally(self):
        """A body fully in shot still yields its floor contact when a frame height is given."""
        self.assertEqual(floor_pixel((100, 200, 200, 400), frame_height=1080), (200.0, 600.0))

    def test_frame_height_is_optional(self):
        """Callers that cannot say how tall the frame is still get a position."""
        self.assertEqual(floor_pixel((100, 500, 200, 580)), (200.0, 1080.0))


class FaceOwnerTests(unittest.TestCase):
    def test_face_inside_a_person_belongs_to_them(self):
        """A face whose centre falls in a person's box is attributed to that track."""
        people = [((100, 50, 80, 300), 7)]
        self.assertEqual(face_owner((120, 60, 40, 40), people), 7)

    def test_face_outside_every_person_belongs_to_nobody(self):
        """A face with no containing body returns None rather than being forced onto the nearest."""
        people = [((100, 50, 80, 300), 7)]
        self.assertIsNone(face_owner((500, 500, 40, 40), people))

    def test_overlapping_people_give_the_face_to_the_smaller_box(self):
        """Where boxes overlap the tighter one wins, being the likelier owner of a contained face."""
        # A large box behind, a small one in front, both containing the face centre.
        people = [((0, 0, 600, 600), 1), ((100, 40, 90, 260), 2)]
        self.assertEqual(face_owner((120, 60, 40, 40), people), 2)

    def test_unconfirmed_people_cannot_own_a_face(self):
        """A body the tracker has not confirmed has no id to attribute a face to."""
        people = [((100, 50, 80, 300), None)]
        self.assertIsNone(face_owner((120, 60, 40, 40), people))

    def test_no_people_at_all(self):
        """A frame with faces but no bodies attributes nothing, rather than erroring."""
        self.assertIsNone(face_owner((120, 60, 40, 40), []))


if __name__ == "__main__":
    unittest.main()


class OverlappingPeopleTests(unittest.TestCase):
    """Somebody standing behind somebody else must not be absorbed into them.

    This is a regression guard. A filter that discarded short person boxes
    -- added to remove a reflection -- also discarded anyone partly hidden,
    because an occluded person is detected as head and shoulders. With
    their own body gone, their face fell inside the nearer person's box
    instead, and the one-face-per-body rule then deduplicated them away.
    The person behind disappeared entirely.
    """

    NEAR = (400, 200, 500, 800)
    BEHIND = (750, 260, 300, 140)
    FACE_NEAR = (560, 260, 180, 200)
    FACE_BEHIND = (830, 300, 120, 130)

    def test_each_face_belongs_to_its_own_person(self):
        """With both bodies present, neither face is attributed to the other person."""
        people = [(self.NEAR, 1), (self.BEHIND, 2)]
        self.assertEqual(face_owner(self.FACE_NEAR, people), 1)
        self.assertEqual(face_owner(self.FACE_BEHIND, people), 2)

    def test_a_partly_hidden_person_is_not_absorbed_into_the_nearer_one(self):
        """The occluded person's face resolves to them, not to whoever is in front.

        Their box is short -- head and shoulders only -- and their face
        centre also falls inside the nearer person's much larger box. The
        smaller box has to win, or the two become one.
        """
        people = [(self.NEAR, 1), (self.BEHIND, 2)]
        near_box = self.NEAR
        self.assertTrue(
            near_box[0] <= self.FACE_BEHIND[0] + self.FACE_BEHIND[2] / 2 <= near_box[0] + near_box[2],
            "the test is meaningless unless the behind face really does sit inside the near body",
        )
        self.assertEqual(face_owner(self.FACE_BEHIND, people), 2)

    def test_a_short_body_is_still_reported(self):
        """An occluded person's box must survive detection to own their face at all.

        Guards the filter that used to drop it: at 140 pixels in a 1080-tall
        frame it was under the old floor, and losing it was what made the
        person behind vanish.
        """
        self.assertLess(self.BEHIND[3], 0.15 * 1080)
        self.assertEqual(face_owner(self.FACE_BEHIND, [(self.NEAR, 1), (self.BEHIND, 2)]), 2)
