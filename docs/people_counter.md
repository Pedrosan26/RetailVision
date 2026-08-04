# People counter

Virtual-line crossing counter: tracks each detected face across frames and
counts crossings of a single configurable line, maintaining net occupancy
and per-track dwell time. Feeds `count` and `dwell_seconds` into the
inference log (see `docs/schema.md`).

## Why centroid tracking, not a full multi-object tracker

The counter only needs to answer one question: did this track cross this
one line? That doesn't require full appearance-based re-identification or
motion prediction -- centroid tracking (nearest-centroid matching frame to
frame, via the Hungarian algorithm for an optimal assignment when multiple
people are present) is enough to keep a track's identity stable long
enough to detect a single line crossing. A tracker capable of following
someone through occlusion, across multiple zone boundaries, or after
leaving and re-entering frame is a materially bigger problem -- that's
scoped as separate future work (see `docs/future_work/`), not something to
half-build here.

## Components

- **`src/retailvision/tracking.py` -- `CentroidTracker`.** Maintains a
  `{track_id: centroid}` map. Each frame, matches new detections' centroids
  against existing tracks by minimum total distance (`scipy.optimize.
  linear_sum_assignment` over a pairwise distance matrix), gated by
  `max_distance` so a detection far from every existing track starts a new
  one instead of hijacking a distant track's ID. A track survives up to
  `max_disappeared` consecutive frames without a match (handles a momentary
  missed detection) before being dropped.
- **`src/retailvision/counter.py` -- `LineCounter`.** Given this frame's
  `{track_id: centroid}` and a timestamp, compares each track's current
  side of the configured line against its previous side. A change from the
  non-entry side to the entry side is an *entry* (occupancy +1, dwell timer
  starts); the reverse is an *exit* (occupancy -1, dwell timer clears). A
  track's first-ever sighting never emits an event -- there's no previous
  side to compare against yet.

## Configuring the line

`pipeline_demo.py` exposes three flags:

- `--line-axis {x,y}` -- whether the line is vertical (perpendicular to x)
  or horizontal (perpendicular to y). Default `x`.
- `--line-position <pixels>` -- where along that axis the line sits.
  Default: the middle of the frame.
- `--line-direction {increasing,decreasing}` -- which crossing direction
  counts as an entry. Default `increasing` (e.g. left-to-right for a
  vertical line).

Point the camera so the line falls across the actual doorway/entrance
you're counting, and pick the direction that matches which side is
"inside."

## `count` and `dwell_seconds` semantics

- **`count`** is net occupancy: entries minus exits so far, floored at 0.
  It answers "how many people are inside right now," which is what pairs
  with a per-person emotion snapshot for zone-engagement scoring -- not a
  monotonically increasing visit total.
- **`dwell_seconds`** is elapsed time since a specific track's entry event,
  recomputed on every classification record logged for that track while it
  remains inside. It is `null` before a track has entered, and reverts to
  `null` after it exits -- there is no persisted per-person track ID, so
  once a track exits there is nothing left to attach a final dwell value
  to retroactively. The log is append-only by design (see
  `docs/schema.md`); a record's `dwell_seconds` is always a snapshot at
  that moment, never edited after the fact.

## Known limitations

- **Single line, not a zone boundary.** If a tracked person leaves the
  frame without recrossing the counting line, no exit is ever registered
  for them, and occupancy can drift upward over a long session. This is
  the direct consequence of scoping to one virtual line instead of a full
  enclosing zone polygon.
- **No re-identification.** If someone leaves the frame and re-enters
  later, they get a new track ID and are counted as a new entry -- the
  centroid tracker has no memory of a track once it's deregistered.

## Accuracy evaluation

`scripts/evaluate_people_counter.py` replays a pre-recorded clip through
the full detector + tracker + counter pipeline and compares the counted
entries against a manually-counted ground truth:

```
PYTHONPATH=. ./venv/bin/python3 scripts/evaluate_people_counter.py \
    --source path/to/clip.mp4 --expected-entries <manually counted number>
```

Watch the clip yourself first and count entries manually -- that number is
the ground truth the script checks against. It reports counted
entries/exits, accuracy (`1 - |counted - expected| / expected`), and
whether that clears the 80% threshold, writing the same report to
`runs/people_counter/eval_report.json`.

### Results

*Pending: run `scripts/evaluate_people_counter.py` against a recorded clip
with a manually-counted ground truth and record the result here.*
