# Multi-node deployment

Running more than one camera on the same deployment doesn't require the
pipeline to change: each camera gets its own machine (a "camera node")
running the existing pipeline end to end, exactly as `pipeline_demo.py`
does today. What's new is an optional second sink that ships each
camera node's anonymized records to a central server in real time, so
results from every node can be viewed in one place.

## Edge inference, not central streaming

Each camera node runs full detection, classification, tracking, and
counting locally -- the central server never receives raw frames or
face crops, only the same anonymized records already written to the
local log. This was a deliberate choice over streaming raw video to a
central machine for inference: it keeps the privacy guarantee already
built into `output_log.py` (raw pixel data never persists, and now also
never leaves the capture device), and needs far less bandwidth than
video streaming -- a handful of small JSON records per second instead
of a live video feed, per node.

## `RemoteLogShipper`

`src/retailvision/remote_log.py` is a second, independent sink alongside
the local NDJSON file -- not a replacement, and not a branch inside
`output_log.py`. It reuses `output_log.py`'s `build_log_record()`
directly, so the record shipped remotely is identical in shape to the
one written locally.

- **Batched, not per-event.** A live pipeline logs on the order of tens
  of records per second per node; POSTing each one individually would
  dominate request volume for no benefit. Records are buffered and
  flushed once either `batch_size` records have accumulated or
  `flush_interval` seconds have passed, whichever comes first.
- **Off the capture thread.** The actual HTTP request runs on a
  single-worker background thread, so a slow or unreachable server
  never stalls frame capture -- `ship()` itself is just a buffer append
  and a threshold check.
- **Drop on failure, don't retry.** If a batch fails to send (timeout,
  connection error, non-2xx response), it's dropped with a printed
  warning rather than queued for retry. The local NDJSON file is
  already the durable record for that node; a persistent retry buffer
  would be new complexity this prototype doesn't need yet.

## Wire format

```
POST /api/v1/ingest
Headers: X-API-Key: <camera node's key>
Body:
{
  "camera_node_id": "laptop-01",
  "records": [
    {"timestamp": "...", "zone_id": null, "count": null, "age_group": "18-40",
     "gender": "Male", "emotion": "Neutral", "dwell_seconds": null, "engagement_score": null},
    ...
  ]
}
```

`camera_node_id` identifies which node a batch came from at the
transport layer -- it's never written into an individual record, so the
frozen 8-field schema (`docs/schema.md`) stays exactly as it is.

## Usage

Shipping is entirely opt-in. Omitting `--server-url` leaves
`pipeline_demo.py`'s behavior unchanged (local logging only):

```
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo \
    --server-url http://localhost:8000 \
    --camera-node-id laptop-01 \
    --api-key <key>
```

`--camera-node-id` and `--api-key` are both required once `--server-url`
is set.
