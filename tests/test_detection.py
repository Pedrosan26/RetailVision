"""
test_detection.py

Unit tests for FaceDetector's result parsing, in particular the tracking
path: box conversion, the null track id ByteTrack reports for a face it
has not yet confirmed, and the pairing of ids to boxes.

The model itself is faked. Loading real weights would make these tests
depend on a checkpoint and a GPU to assert something that is entirely
about unpacking a result object.
"""

import unittest

from src.retailvision.detection import FaceDetector, _boxes_from


class FakeTensor:
    """Stands in for the torch tensors on a YOLO result, which are only ever read via tolist()."""

    def __init__(self, values):
        """Hold the values this tensor reports."""
        self._values = values

    def tolist(self):
        """Return the values as a plain list, as torch does."""
        return self._values


class FakeBoxes:
    """The .boxes attribute of a YOLO result: xyxy coordinates and optional track ids."""

    def __init__(self, xyxy, ids=None):
        """Hold boxes, and track ids when the result came from tracking."""
        self.xyxy = FakeTensor(xyxy)
        self.id = None if ids is None else FakeTensor(ids)


class FakeResult:
    """One YOLO result, carrying just the boxes the detector reads."""

    def __init__(self, xyxy, ids=None):
        """Wrap the given boxes and ids."""
        self.boxes = FakeBoxes(xyxy, ids)


class FakeModel:
    """A YOLO stand-in that returns a prepared result and records how it was called."""

    def __init__(self, result):
        """Hold the result to return from every call."""
        self._result = result
        self.track_calls = []

    def track(self, **kwargs):
        """Record the call and return the prepared result, as model.track does."""
        self.track_calls.append(kwargs)
        return [self._result]


def detector_with(result):
    """Build a FaceDetector around a fake model, skipping the real checkpoint load."""
    detector = object.__new__(FaceDetector)
    detector._model = FakeModel(result)
    detector._device = None
    return detector


class BoxConversionTests(unittest.TestCase):
    def test_xyxy_becomes_integer_xywh(self):
        """Corner coordinates convert to (x, y, width, height) integers."""
        self.assertEqual(_boxes_from(FakeResult([[10.0, 20.0, 50.0, 80.0]])), [(10, 20, 40, 60)])

    def test_no_boxes_gives_no_results(self):
        """A frame with no detections converts to an empty list, not an error."""
        self.assertEqual(_boxes_from(FakeResult([])), [])


class TrackTests(unittest.TestCase):
    def test_ids_are_paired_with_their_boxes(self):
        """Each box comes back alongside the track id at the same position."""
        detector = detector_with(FakeResult([[0, 0, 10, 10], [20, 20, 40, 50]], ids=[7.0, 9.0]))
        self.assertEqual(detector.track(frame=None), [((0, 0, 10, 10), 7), ((20, 20, 20, 30), 9)])

    def test_unconfirmed_detections_report_a_null_id(self):
        """When ByteTrack has confirmed no tracks yet, boxes are still returned, with id None.

        The detection is real even when the tracker will not yet vouch for
        it, so it is reported rather than silently dropped here -- deciding
        what to do with it belongs to the caller.
        """
        detector = detector_with(FakeResult([[0, 0, 10, 10]], ids=None))
        self.assertEqual(detector.track(frame=None), [((0, 0, 10, 10), None)])

    def test_tracking_persists_state_between_frames(self):
        """persist=True is passed, without which every frame would start its tracks over."""
        detector = detector_with(FakeResult([], ids=None))
        detector.track(frame=None)
        self.assertTrue(detector._model.track_calls[0]["persist"])

    def test_tracking_uses_the_project_bytetrack_config(self):
        """The tuned config is passed rather than relying on Ultralytics' defaults."""
        detector = detector_with(FakeResult([], ids=None))
        detector.track(frame=None)
        self.assertTrue(detector._model.track_calls[0]["tracker"].endswith("config/bytetrack.yaml"))


if __name__ == "__main__":
    unittest.main()
