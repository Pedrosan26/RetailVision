# Inference log schema

The pipeline's privacy layer (`src/retailvision/output_log.py`) is the only
thing that persists inference results to disk. It never writes raw camera
frames or face crops -- only the anonymized fields below, one record per
detection event, appended to `data/inference_log.json`.

## Format

Newline-delimited JSON (one JSON object per line), not a single JSON array.
A JSON array would require reading and rewriting the entire file on every
append; newline-delimited JSON supports true O(1) appends, which matters
for a log written once per detected face per frame in a real-time pipeline.
Despite the `.json` extension (kept to match the path named in the
originating ticket), each line should be parsed independently.

## Fields

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | ISO 8601 UTC timestamp of the detection event. |
| `zone_id` | string \| null | ID of the zone the person was standing in. `null` when the pipeline runs without `--zones`, and also when zones are configured but the camera cannot currently see a mapped marker -- so "unknown" stays distinct from "outside every zone". |
| `count` | integer \| null | Net occupancy (entries minus exits so far) at the moment of the detection event, from `LineCounter`. `null` if no counter was supplied. |
| `age_group` | string | Predicted age bracket, one of the age classifier's classes (see `docs/datasets/utkface.md`). |
| `gender` | string | Predicted gender, one of the gender classifier's classes. |
| `emotion` | string | Predicted emotion, one of the emotion classifier's classes. |
| `dwell_seconds` | number \| null | Seconds the detected person's track has been present since its entry event. `null` if the track hasn't registered an entry yet, or no counter was supplied. |
| `engagement_score` | number \| null | Normalized 0-100 engagement score for the zone/time window. `null` until the zone-emotion correlation module lands. |

This 8-field shape is frozen: fields not yet computable are included now as
`null` placeholders specifically so that later modules can populate real
values without changing the schema downstream components (e.g. the
dashboard) are built against.

## Example records

```json
{"timestamp": "2026-07-31T10:15:32.101+00:00", "zone_id": null, "count": 3, "age_group": "18-40", "gender": "Male", "emotion": "Neutral", "dwell_seconds": 12.4, "engagement_score": null}
{"timestamp": "2026-07-31T10:15:32.340+00:00", "zone_id": null, "count": 3, "age_group": "6-12", "gender": "Female", "emotion": "Happy", "dwell_seconds": null, "engagement_score": null}
```

## Known limitations

- **No person or track identifier.** A record represents one anonymous
  detection event, not a person -- the internal track ID used to compute
  `dwell_seconds` is never persisted to the log. If multiple people are
  detected in the same frame, their records are indistinguishable beyond
  their independent `age_group`/`gender`/`emotion` values and
  near-identical timestamps. This is deliberate: exposing a persistent
  per-person identifier in an anonymized log would undercut the privacy
  goal it exists for.
- **`dwell_seconds` reflects a single virtual line, not a zone boundary.**
  A track that leaves the frame without recrossing the counting line is
  never marked as exited, so its last logged `dwell_seconds` before
  disappearing understates nothing but also never resolves to a final
  value. See `docs/people_counter.md` for the full counting methodology
  and its limitations.

## What is deliberately excluded

- **Bounding boxes.** `process_frame()` returns pixel-space `bbox` per
  detection, but it is dropped before logging -- it's derived directly
  from frame geometry and isn't needed downstream, so there's no reason to
  persist it.
- **Classifier confidence scores.** Also dropped for the same reason: not
  part of the frozen schema, and not needed by any downstream consumer.
- **Any pixel data.** Neither the source frame nor the face crop used for
  classification is ever written to disk, at any pipeline stage.
