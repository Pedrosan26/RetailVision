# Server

Central FastAPI service that receives anonymized detection records
shipped from camera nodes (`src/retailvision/remote_log.py`, see
`docs/multi_node.md`) and persists them to a TimescaleDB hypertable.
Lives in `server/`, a separate subproject with its own venv and
dependencies -- FastAPI/SQLAlchemy/asyncpg have nothing to do with the
pipeline's torch/ultralytics stack.

## Layout

```
server/
  app/
    main.py                 # FastAPI app, CORS, router registration
    config.py                # Settings loaded from the environment/.env
    db.py                    # async engine + get_db() session dependency
    auth.py                  # camera-node API key verification
    models/detection.py      # SQLAlchemy ORM: DetectionEvent
    schemas/detection.py     # Pydantic: mirrors the frozen 8-field schema, plus read-endpoint response models
    utils.py                 # as_utc(): normalizes SQLite's naive datetimes to match Postgres' aware ones
    routers/
      ingest.py                # POST /api/v1/ingest
      detections.py             # GET /api/v1/detections
      occupancy.py               # GET /api/v1/occupancy/live
      aggregates.py               # GET /api/v1/aggregates
  migrations/                # Alembic (async)
  tests/                     # pytest, against an in-memory SQLite DB
```

## Endpoints

### `POST /api/v1/ingest`

Validates a batch against the frozen schema, requires `X-API-Key` to
match the configured key for the request's `camera_node_id`, and
bulk-inserts on success.

```
POST /api/v1/ingest
Headers: X-API-Key: <camera node's key>
Body:
{
  "camera_node_id": "laptop-01",
  "records": [
    {"timestamp": "...", "zone_id": null, "count": 3, "age_group": "18-40",
     "gender": "Male", "emotion": "Neutral", "dwell_seconds": null, "engagement_score": null}
  ]
}
-> 202 {"accepted": 1}
```

A malformed record (missing/wrong-typed field) returns `422`. A missing
or incorrect `X-API-Key`, or an unrecognized `camera_node_id`, returns
`401` -- all three cases are treated identically rather than leaking
which part of the credential was wrong.

### `GET /api/v1/detections`

Recent detections, newest first. Query params: `limit` (default 100,
capped at 1000), `camera_node_id`, `zone_id`, `since`, `until` (all
optional filters).

### `GET /api/v1/occupancy/live`

The latest `count` per `COALESCE(zone_id, camera_node_id)`, computed
with a `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY timestamp DESC)`
window query (portable across Postgres and SQLite, unlike Postgres-only
`DISTINCT ON`). Groups by real zones automatically once zone
configuration starts populating `zone_id` -- no query change needed at
that point.

### `GET /api/v1/summary`

Headline figures for one time range in one response: total events,
distinct people, average dwell, the emotion mix, the busiest hour
(the hour-aligned bucket with the most distinct people), and
`peak_occupancy` -- the largest node-reported headcount at any single
moment, deduplicated the way the node deduplicated it rather than
summed across cameras. Query params:
`since`/`until` (default: last 24h), `zone_id`. Backs the dashboard's
KPI row; deriving these client-side from /aggregates would mean summing
per-bucket unique-people counts, which overcounts anyone present across
a bucket boundary. `unique_people` carries the same per-camera caveat as
the aggregates: someone visible to two cameras counts twice.

### `GET /api/v1/visits`

Detection events folded into one row per person: per (camera node,
track), first/last sighting, duration, mode zone, the settled age and
gender, and the emotion mix with its dominant label. Query params:
`since`/`until` (default: last 24h), `zone_id`, `camera_node_id`,
`limit` (default 200, newest last-seen first). Rows without a
`track_id` are excluded rather than surfacing as hundreds of one-event
visits. Duration spans first to last record; emission is change-driven
with a 10s heartbeat, so it understates a stay by at most that
heartbeat. Backs the dashboard's Visits page.

### `POST /api/v1/zones/geometry` / `GET /api/v1/zones/geometry`

Zone floor polygons, in world meters. The polygon is computed on the
nodes from the surveyed marker map, which the server never sees, so each
node uploads its ready zones' geometry once at startup (authenticated
like ingest); the dashboard reads it back to draw the floor map that
world positions land on. One row per zone, last writer wins -- every
node loads the same surveyed map file, so a disagreement means a stale
node rather than a conflict worth merging.

### `GET /api/v1/aggregates`

Time-windowed rollups: detection count, distinct people, age/gender/emotion
distribution, and average dwell/engagement per bucket. Query params: `window` (e.g.
`5m`, `1h`, `24h`; default `5m`), `since`/`until` (default: last 24h),
`zone_id`, and the repeatable demographic filters `age_group`, `gender`
and `emotion`.

The demographic filters narrow which events are counted at all, before
bucketing. Repeating one reads as "any of these"
(`?emotion=Happy&emotion=Neutral`); different dimensions intersect
(`?age_group=18-40&gender=Male` is both). Omitting a dimension does not
filter on it, which is what the dashboard sends when its filters are
cleared -- so "show everything" is the absence of a parameter rather
than a list of every known value, and a label the classifier has never
emitted never has to be enumerated anywhere.

`unique_people` counts distinct `(camera_node_id, track_id)` pairs, not
rows. The pair is the identity because a node's track IDs are only unique
within that node and within one run of its process, so the same ID arriving
from two nodes is two people. Rows with no `track_id` -- from a node that
predates the field -- count as one person each, which is exactly what they
meant before it existed.

This is per-camera, so someone standing where two cameras overlap still
counts twice here. Collapsing that is a spatial question, answered from the
shared world frame by `/occupancy/zones`, not by the track ID.

Bucketing happens in Python, not via TimescaleDB's `time_bucket()` SQL
function: one portable filtered query fetches matching rows, then
`routers/aggregates.py::bucket_events()` groups and aggregates them.
At the data volumes implied by a handful of camera nodes this is simple
and fast enough that raw dialect-specific SQL isn't worth the
portability cost, and it keeps the logic testable against the in-memory
SQLite DB used in tests as well as real TimescaleDB in production. An
unparseable `window` returns `400`.

## Authentication

Deliberately minimal for a prototype: one shared API key per camera
node, configured as a single environment variable (`CAMERA_NODE_API_KEYS`,
`node_id:key` pairs separated by commas) and checked in
`app/auth.py::verify_camera_node_api_key`. No user accounts, sessions,
or per-endpoint roles. This is a scope decision, not an oversight --
revisit before exposing this server beyond a trusted local network.

## Database schema

One hypertable, `detection_events`, partitioned on `timestamp`:

- Primary key is `(id, timestamp)`, not just `id` -- TimescaleDB requires
  the partitioning column in every unique constraint. The SQLAlchemy
  model (`app/models/detection.py`) declares only `id` as its
  Python-level primary key for cross-backend ORM simplicity (including
  SQLite in tests); the real composite constraint is defined in the
  Alembic migration's raw DDL, since it's a TimescaleDB-specific
  requirement, not an ORM one.
- Indexed on `(camera_node_id, timestamp DESC)` for per-node queries.
- A **partial** index on `(zone_id, timestamp DESC) WHERE zone_id IS NOT
  NULL` -- near-free to maintain today since `zone_id` is always null,
  ready the moment zone configuration lands with no migration needed.
- No continuous aggregate yet. A rollup grouped by `zone_id` today would
  just materialize one all-null bucket; read-side aggregation queries
  (a later ticket) compute `time_bucket()` at read time first, and get
  promoted to a continuous aggregate once real zone/engagement data
  validates the query shape.

`ingested_at` (server receipt time) and `camera_node_id` are
server-side/transport-level columns, not part of the frozen 8-field
client schema in `docs/schema.md` -- the client's `timestamp` field is
stored as-is alongside them.

## Local development

TimescaleDB via Docker Compose (`docker-compose.yml` at the repo root).
Host port **5433**, not 5432 -- avoids clashing with a locally installed
Postgres that may already be listening on the default port. Credentials
come from a root-level `.env` (see `.env.example`), not hardcoded in the
compose file:

```
cp .env.example .env    # or set POSTGRES_USER/PASSWORD/DB to your own values
docker compose up -d timescaledb
```

Server setup (separate venv from the pipeline's):

```
cd server
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env          # edit DATABASE_URL's port if you didn't remap it, and set CAMERA_NODE_API_KEYS
./venv/bin/alembic upgrade head
./venv/bin/uvicorn app.main:app --reload --port 8000
```

Sanity check: `curl http://localhost:8000/health`, or open
`http://localhost:8000/docs` for the interactive Swagger UI.

## Tests

```
cd server
./venv/bin/python3 -m pytest tests/ -v
```

Runs against an in-memory SQLite database (`tests/conftest.py`), not
real Postgres/TimescaleDB -- this validates the FastAPI app and ORM
logic (validation, auth, persistence via the ORM), not TimescaleDB-
specific behavior. The hypertable/index DDL is exercised by actually
applying the Alembic migration against real TimescaleDB, verified
manually: `alembic upgrade head` followed by a real `POST
/api/v1/ingest` and confirming the row lands correctly.


## Cross-camera deduplication

`GET /api/v1/occupancy/zones` answers how many people are actually in a
zone, as opposed to `GET /api/v1/occupancy/live`, which reports what each
camera individually last said.

The two differ because **summing per-camera counts is not a headcount**.
Cameras covering one area see each other's subjects, so a person visible to
three cameras is reported three times; with several cameras on one room the
sum can be a multiple of the truth.

Merging is possible because every camera watching a zone reports positions
in the same world frame (see `docs/aruco_zones.md`), so "is this the same
person" becomes a distance check. `app/dedup.py` holds the logic. Two
decisions in it are worth knowing, because each trades one error for
another:

- **Only detections from different cameras are merged.** Two people standing
  close together are reported by one camera as two detections, and merging
  those would undercount a queue or a group -- the situations occupancy is
  most used to measure. A camera's own frame is taken as ground truth for
  how many people it sees.
- **Each camera contributes only its most recent frame.** Positions from a
  second ago belong to someone who has since moved, so pooling frames would
  smear one person into a trail and count them repeatedly.

The merge radius defaults to 1.0m and is tunable per request. It must
exceed the position error, which is dominated by how far each subject's real
head height differs from the assumed one -- roughly half a metre for a
camera a metre above the assumed plane, and much worse for a camera closer
to it. See the head-height discussion in `docs/aruco_zones.md`; a merge
radius smaller than that error counts one person twice, and one much larger
merges two people standing near each other.

Records without a position -- nodes running without `--zones` -- cannot
contribute to a headcount and are excluded.
