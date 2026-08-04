"""
test_remote_log.py

Unit tests for RemoteLogShipper: batching/flush-threshold behavior and
failure handling, with the HTTP call mocked out -- no real network needed.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.retailvision.remote_log import RemoteLogShipper

SAMPLE_DETECTION = {
    "bbox": (10, 20, 100, 100),
    "age_group": "18-40",
    "gender": "Male",
    "emotion": "Happy",
    "confidence": {"age": 0.91, "gender": 0.98, "emotion": 0.77},
}


class TestRemoteLogShipper(unittest.TestCase):
    """Verify RemoteLogShipper buffers, flushes, and ships records correctly."""

    def _make_shipper(self, **overrides) -> RemoteLogShipper:
        """Build a shipper with test-friendly defaults, overridable per test."""
        kwargs = dict(
            server_url="http://localhost:8000",
            camera_node_id="test-node",
            api_key="test-key",
            batch_size=3,
            flush_interval=100.0,
        )
        kwargs.update(overrides)
        return RemoteLogShipper(**kwargs)

    def test_ship_does_not_flush_below_thresholds(self) -> None:
        """Buffering below both the batch size and flush interval doesn't trigger a send."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            shipper = self._make_shipper(batch_size=5, flush_interval=100.0)
            shipper.ship(SAMPLE_DETECTION)
            shipper.ship(SAMPLE_DETECTION)
            self.assertEqual(len(shipper._buffer), 2)
            mock_post.assert_not_called()

    def test_ship_flushes_at_batch_size(self) -> None:
        """Reaching batch_size triggers a flush, sending exactly that many records."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=202)
            shipper = self._make_shipper(batch_size=3, flush_interval=100.0)
            shipper.ship(SAMPLE_DETECTION)
            shipper.ship(SAMPLE_DETECTION)
            shipper.ship(SAMPLE_DETECTION)
            shipper.close()

            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(len(payload["records"]), 3)

    def test_flush_at_time_interval(self) -> None:
        """A flush_interval of 0 flushes on the very next ship() call regardless of batch size."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=202)
            shipper = self._make_shipper(batch_size=100, flush_interval=0.0)
            shipper.ship(SAMPLE_DETECTION)
            shipper.close()

            mock_post.assert_called_once()

    def test_close_flushes_remaining_records(self) -> None:
        """close() flushes whatever is still buffered, even below the batch threshold."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=202)
            shipper = self._make_shipper(batch_size=100, flush_interval=100.0)
            shipper.ship(SAMPLE_DETECTION)
            shipper.close()

            mock_post.assert_called_once()
            payload = mock_post.call_args.kwargs["json"]
            self.assertEqual(len(payload["records"]), 1)

    def test_empty_flush_is_a_noop(self) -> None:
        """Flushing with nothing buffered doesn't send an empty request."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            shipper = self._make_shipper()
            shipper.flush()
            shipper.close()
            mock_post.assert_not_called()

    def test_payload_format(self) -> None:
        """The shipped payload has camera_node_id and schema-conforming records, no bbox/confidence."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=202)
            shipper = self._make_shipper(batch_size=1, camera_node_id="node-42")
            shipper.ship(SAMPLE_DETECTION)
            shipper.close()

            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            self.assertEqual(payload["camera_node_id"], "node-42")
            record = payload["records"][0]
            self.assertEqual(record["age_group"], "18-40")
            self.assertNotIn("bbox", record)
            self.assertNotIn("confidence", record)
            self.assertEqual(call_kwargs["headers"]["X-API-Key"], "test-key")

    def test_ship_forwards_count_and_dwell_seconds(self) -> None:
        """count and dwell_seconds passed to ship() land in the shipped record, not just the local log."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=202)
            shipper = self._make_shipper(batch_size=1)
            shipper.ship(SAMPLE_DETECTION, count=4, dwell_seconds=12.5)
            shipper.close()

            record = mock_post.call_args.kwargs["json"]["records"][0]
            self.assertEqual(record["count"], 4)
            self.assertEqual(record["dwell_seconds"], 12.5)

    def test_send_failure_does_not_raise(self) -> None:
        """A network failure while shipping is caught and warned about, never propagated to the caller."""
        with patch("src.retailvision.remote_log.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("connection refused")
            shipper = self._make_shipper(batch_size=1)
            try:
                shipper.ship(SAMPLE_DETECTION)
                shipper.close()
            except Exception as exc:  # noqa: BLE001 -- explicitly proving nothing escapes
                self.fail(f"ship()/close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
