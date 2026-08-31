"""
test_appearance.py

Tests for the per-track appearance accumulator and the similarity it is
compared with. The encoder itself is not exercised here -- it is a
downloaded ONNX model, and what needs guarding is the averaging around
it: an unusable view must not corrupt a track's description, and a
retired track must not linger.
"""

import unittest

import numpy as np

from src.retailvision.appearance import TrackAppearance, similarity


def unit(*components: float) -> np.ndarray:
    """A normalised vector, standing in for an encoder output."""
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class SimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        """The same appearance compared with itself is a perfect match."""
        vector = unit(1.0, 2.0, 3.0)
        self.assertAlmostEqual(similarity(vector, vector), 1.0, places=5)

    def test_orthogonal_vectors_score_zero(self):
        """Appearances sharing nothing score zero rather than something small and positive."""
        self.assertAlmostEqual(similarity(unit(1.0, 0.0), unit(0.0, 1.0)), 0.0, places=5)


class TrackAppearanceTests(unittest.TestCase):
    def setUp(self):
        """Start each test with an empty accumulator."""
        self.appearance = TrackAppearance()

    def test_a_track_with_no_views_has_no_appearance(self):
        """Nothing is invented for a track that has not been seen usably yet."""
        self.assertIsNone(self.appearance.embedding(1))
        self.assertEqual(self.appearance.observations(1), 0)

    def test_views_are_averaged_and_renormalised(self):
        """Several views combine into one vector that is still unit length.

        Length matters: comparison is a dot product, so an un-normalised
        mean would score by how many frames a track had rather than by how
        much it looks like the other one.
        """
        self.appearance.observe(1, unit(1.0, 0.0))
        self.appearance.observe(1, unit(0.0, 1.0))
        mean = self.appearance.embedding(1)
        self.assertAlmostEqual(float(np.linalg.norm(mean)), 1.0, places=5)
        self.assertAlmostEqual(float(mean[0]), float(mean[1]), places=5)

    def test_unusable_views_are_ignored_rather_than_counted(self):
        """A crop the encoder could not describe leaves the track's appearance untouched."""
        self.appearance.observe(1, unit(1.0, 0.0))
        self.appearance.observe(1, None)
        self.assertEqual(self.appearance.observations(1), 1)
        np.testing.assert_allclose(self.appearance.embedding(1), unit(1.0, 0.0), atol=1e-6)

    def test_tracks_are_kept_apart(self):
        """Two tracks accumulate independently, however similar their views."""
        self.appearance.observe(1, unit(1.0, 0.0))
        self.appearance.observe(2, unit(0.0, 1.0))
        self.assertAlmostEqual(similarity(self.appearance.embedding(1), self.appearance.embedding(2)), 0.0, places=5)

    def test_retiring_forgets_tracks_no_longer_present(self):
        """Memory follows who is in the room, not how long the process has been up."""
        self.appearance.observe(1, unit(1.0, 0.0))
        self.appearance.observe(2, unit(0.0, 1.0))
        self.appearance.retire({2})
        self.assertIsNone(self.appearance.embedding(1))
        self.assertIsNotNone(self.appearance.embedding(2))


if __name__ == "__main__":
    unittest.main()
