"""
remote_log.py

Second sink for anonymized detection records, alongside the local NDJSON
file written by output_log.py: ships the same schema-conforming records
to a central server in real time over HTTP, for deployments running
multiple camera nodes. Batches records instead of posting one per
detection, since a live pipeline logs on the order of tens of records
per second per node and per-event POSTs would dominate request volume
for no benefit. Reuses build_log_record() so the record shape sent
remotely is identical to what's written locally -- this sink never
re-derives or reshapes a detection, only forwards it. The actual HTTP
call runs on a background worker so a slow or unreachable server never
stalls frame capture; on any delivery failure the batch is dropped with
a warning rather than retried, since the local NDJSON file already is
the durable record for that node.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .output_log import build_log_record


class RemoteLogShipper:
    """Buffers anonymized records and ships them to a central server in batches, off the capture thread."""

    def __init__(
        self,
        server_url: str,
        camera_node_id: str,
        api_key: str,
        batch_size: int = 20,
        flush_interval: float = 2.0,
        request_timeout: float = 3.0,
    ) -> None:
        """Configure the target server, this node's identity, and the batching/timeout thresholds."""
        self._ingest_url = server_url.rstrip("/") + "/api/v1/ingest"
        self._camera_node_id = camera_node_id
        self._api_key = api_key
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._request_timeout = request_timeout
        self._buffer: list[dict] = []
        self._last_flush = time.monotonic()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def ship(
        self,
        detection: dict,
        timestamp: str | None = None,
        count: int | None = None,
        dwell_seconds: float | None = None,
    ) -> None:
        """Build a schema-conforming record, buffer it, and flush once the batch size or flush interval is reached."""
        record = build_log_record(detection, timestamp=timestamp, count=count, dwell_seconds=dwell_seconds)
        self._buffer.append(record)

        elapsed = time.monotonic() - self._last_flush
        if len(self._buffer) >= self._batch_size or elapsed >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        """Submit all currently buffered records for delivery in one batch and clear the buffer."""
        if not self._buffer:
            return
        batch = self._buffer
        self._buffer = []
        self._last_flush = time.monotonic()
        self._executor.submit(self._send, batch)

    def _send(self, batch: list[dict]) -> None:
        """POST one batch of records to the server; warn and drop the batch on any failure, without retrying."""
        payload = {"camera_node_id": self._camera_node_id, "records": batch}
        headers = {"X-API-Key": self._api_key}
        try:
            response = requests.post(self._ingest_url, json=payload, headers=headers, timeout=self._request_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Warning: failed to ship {len(batch)} record(s) to {self._ingest_url}: {exc}")

    def close(self) -> None:
        """Flush any remaining buffered records and wait for the background worker to finish delivering them."""
        self.flush()
        self._executor.shutdown(wait=True)
