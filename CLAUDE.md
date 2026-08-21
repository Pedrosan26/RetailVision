# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RetailVision is an AI-powered computer vision pipeline for real-time customer demographic detection, emotion recognition, foot traffic counting, and zone-engagement scoring in retail environments.

## Stack and architecture

- **Language/stack**: Python, using OpenCV for capture/display and PyTorch/Ultralytics YOLOv8 for detection, recognition, and (planned) scoring stages.
- **Architecture**: a single pipeline service — one process ingests camera streams and runs detection, demographic/emotion recognition, foot traffic counting, and zone-engagement scoring, rather than splitting these into separate microservices.

## Setup

```
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The venv must be created with **Python 3.12** (via Homebrew's `python@3.12`), not the system `/usr/bin/python3` (3.9.6, too old) and not another project's venv that may be first on `PATH`.

`opencv-python` is pinned to `4.10.0.84` in `requirements.txt` — the current latest (5.0.0.93) shipped a broken build on macOS missing `cv2.CascadeClassifier` and the bundled Haar cascade XML files entirely. Production code no longer uses either (`FaceDetector` now runs a YOLOv8 model, see below), but the pin hasn't been re-verified against a newer opencv-python for the other cv2 functionality still used throughout (capture, display, video I/O) — don't bump it without testing generally, not just checking Haar cascade support.

## Running

```
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo             # live camera, full age/gender/emotion pipeline
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --source f  # pre-recorded video file instead of live camera
PYTHONPATH=. ./venv/bin/python3 -m src.retailvision.pipeline_demo --benchmark # headless, prints average FPS on exit
./venv/bin/python3 -m src.retailvision.camera_test                            # camera-only sanity check, no detection
PYTHONPATH=. ./venv/bin/python3 -m unittest discover -s tests -v              # run the test suite
```

The live/video modes open a preview window; press `q` to quit.

## Module layout

- `src/retailvision/camera_test.py` — minimal camera-open/read sanity check.
- `src/retailvision/detection.py` — `FaceDetector`, runs a YOLOv8 detection-mode model trained from scratch on WIDER FACE and fine-tuned for retail camera conditions. First stage of the pipeline.
- `src/retailvision/inference.py` — `InferencePipeline`, runs `FaceDetector` then the fine-tuned age/gender and emotion YOLOv8 classifiers on every detected face crop, returning one dict per person. See `docs/inference_pipeline.md` for architecture and design decisions (notably: one shared detector feeds all three classification heads, so no cross-model bounding-box matching is needed). Emotion output collapses Angry/Disgust/Fear/Sad into a single `negative` label (post-hoc remap of the 7-class classifier's predictions, not a retrain) — see `docs/models/emotion_negative_consolidation.md` for the data behind that decision.
- `src/retailvision/tracking.py` — `CentroidTracker`, assigns stable track IDs to detected faces across frames via nearest-centroid matching (Hungarian algorithm). See `docs/people_counter.md`.
- `src/retailvision/counter.py` — `LineCounter`, detects virtual-line crossings from tracked centroids, maintaining net occupancy and per-track dwell time. See `docs/people_counter.md`.
- `src/retailvision/calibration.py`, `marker_pose.py`, `marker_map.py`, `zones.py` — marker-based zone occupancy, intended to replace the virtual-line counter. Printed ArUco markers at a zone's floor corners are pose-estimated per marker (`cv2.solvePnP` + the marker's known printed size, the only measurement in the system), then merged into one shared world frame so several cameras can cover a zone none of them sees whole — two cameras need **one** marker in common, not four. `CameraLocalizer` exists specifically to resolve the two-fold orientation ambiguity of a lone square marker, which silently corrupts the map if picked by reprojection error alone. Wired into `pipeline_demo.py` behind `--zones` (which requires `--calibration`): `ZoneResolver` localizes the camera each frame, back-projects each detection onto a head-height plane for its coordinate, tests zone membership by sampling the viewing ray across the whole plausible band of face heights (so seated and standing people both count), and populates `zone_id` plus a per-zone live headcount in `count`. Without `--zones` the pipeline is byte-for-byte unchanged and `zone_id` stays null. Markers can be `--marker-mounting floor` or `wall`; wall mounting also needs `--marker-height`, the anchor's height above the floor, or the whole world sits at marker level. See `docs/aruco_zones.md` for the design, the accuracy data, and camera/marker placement geometry; `docs/camera_setup.md` for running several cameras on one machine, calibration technique, and why a good reprojection error does not mean a usable calibration.
- `src/retailvision/person_track.py` — turns the per-frame detection stream into a per-person event stream. Age/gender are decided **once per track** by majority vote over its first few frames then fixed (the classifiers disagree with themselves frame to frame, so a vote beats keeping whichever answer landed last); a track is not reported at all until that vote settles, which also discards single-frame false positives. Emotion is smoothed over a short window, and a record is emitted only when that smoothed emotion or the zone changes, or a heartbeat elapses — so a person sitting still produces a handful of records rather than one per frame. `track_id` is random per track and per process run: anonymous, not durable, and not comparable between cameras.
- `src/retailvision/output_log.py` — privacy layer: converts each `InferencePipeline` detection into an anonymized record (demographic/emotion labels only, never pixel data) and appends it as newline-delimited JSON to `data/inference_log.json`. One record per person per change, not per face per frame — see `person_track.py`. Schema documented in `docs/schema.md`.
- `src/retailvision/remote_log.py` — optional second sink: batches the same anonymized records and ships them to a central server in real time, for multi-node deployments. See `docs/multi_node.md`.
- `src/retailvision/pipeline_demo.py` — wires capture (camera or video file) → `InferencePipeline` → tracking/counting (or marker-based zone occupancy with `--zones`) → live preview with drawn bounding boxes, predictions, and either the counting line or per-zone headcounts, or a headless FPS benchmark. Also logs every detection via `output_log.py`, and optionally ships it via `remote_log.py`.

Each machine reads from its own default camera (`cv2.VideoCapture(0)`) or a local video file. Multiple cameras means multiple machines each running the pipeline (each is a "camera node") — see `docs/multi_node.md` for shipping their results to a central server.

## Server

`server/` is a separate FastAPI + TimescaleDB subproject (own venv/requirements.txt — unrelated to the pipeline's torch/ultralytics deps) that receives shipped records, persists them, and serves read/aggregate endpoints (`/detections`, `/occupancy/live`, `/aggregates`) for the dashboard. See `docs/server.md`. Local dev: `docker compose up -d timescaledb` (host port **5433**, not 5432 — avoids clashing with a locally installed Postgres), then inside `server/`: `./venv/bin/alembic upgrade head` then `./venv/bin/uvicorn app.main:app --reload --port 8000`. Tests (`server/tests/`, pytest) run against an in-memory SQLite DB, not real Postgres — validates app/ORM logic; TimescaleDB-specific DDL (hypertable, indexes) is only exercised by actually applying the Alembic migration.

## Dashboard

`retailVision/` is the React + TypeScript + Vite dashboard (own `package.json` — unrelated to the pipeline/server Python subprojects), reading from the server's endpoints documented above. Colours are CSS custom properties on `:root` (`--app-*` for chrome, `--viz-*` for charts) referenced as Tailwind arbitrary values, so one class covers light and dark — there are no `dark:` variants and no default-palette classes left; see `docs/dashboard.md`. Stack: Tailwind for styling, TanStack Query for server state (fetching/caching/polling the read endpoints), Zustand for client/UI state only (selected camera node/zone, time-range filter — never server data), `react-router-dom` for routing. Local dev: `cd retailVision && nvm use && npm install && npm run dev` (expects the server running at `VITE_API_BASE_URL`, default `http://localhost:8000`).

Requires **Node 22** (pinned via `.nvmrc`) — the Vite 8/oxlint toolchain here needs Node `^20.19 || >=22.12`, and an older Node doesn't error clearly, it just silently skips installing their platform-specific native bindings (`npm install` "succeeds" but `npm run dev`/`lint` then crash on a missing binding). If you hit that, it's this, not a corrupted install — switch Node versions rather than reinstalling.

Layout:
```
retailVision/
  src/
    api/        client.ts (fetch wrapper), detections.ts (endpoint calls), types.ts (mirrors server Pydantic response models)
    hooks/      useLiveOccupancy, useRecentDetections, useAggregates, useSummary -- TanStack Query, polling on an interval
    store/      uiStore.ts -- Zustand, UI-only state
    components/
      layout/     AppShell (sidebar + routed content), Sidebar
      occupancy/  ZoneOccupancyGrid -- per-zone headcount, people seen by several cameras counted once;
                  OccupancyGrid, OccupancyCard -- what each camera node last reported, unmerged
      charts/     StackedAreaChart, DistributionBars, HistoricalCharts -- hand-rolled SVG over /aggregates
      detections/ RecentActivityFeed -- compact recent-detections list for the Overview page
      common/     LoadingState, ErrorState, ui.tsx (Card/PageHeader/StatTile/Badge/EmptyState/SegmentedControl)
    pages/      OverviewPage, DetectionsPage, ZonesPage, NotFoundPage
```

Pages:
- `OverviewPage` — the landing route (`/`); live occupancy grid plus a recent-activity feed. Fully wired to the server's `/occupancy/live` and `/detections` endpoints.
- `DetectionsPage` (`/detections`) — the raw event log: filterable by camera node, zone and time range, paged client-side. One row is one detection event, not one person.
- `ZonesPage` (`/zones`) — one zone at a time: deduplicated headcount, which cameras contribute it, and that zone's history from `/aggregates?zone_id=`.
- `NotFoundPage` — catch-all 404 for unmatched routes.

## Conventions

- Every module gets a file-level docstring explaining its purpose, and every function/method gets a one-line docstring explaining what it does — this project intentionally documents intent inline, don't skip it.
