# Dashboard

React + TypeScript dashboard that reads from the server's endpoints
(`docs/server.md`) and shows live occupancy, recent detections, and
(eventually) per-zone views. Lives in `retailVision/`, a separate
subproject with its own `package.json` -- unrelated to the pipeline and
server's Python dependencies.

## Stack

- **Vite** — dev server and build tool.
- **TypeScript** — throughout.
- **Tailwind CSS v4** — via the `@tailwindcss/vite` plugin, no separate
  PostCSS config needed.
- **TanStack Query** — owns server state: fetching, caching, and polling
  the read endpoints (`useLiveOccupancy`, `useRecentDetections`,
  `useAggregates` in `src/hooks/`). Each hook polls on an interval
  rather than opening a WebSocket/SSE connection -- cheap, no new server
  infra, and sufficient freshness for occupancy/dwell numbers at this
  scale. If push updates are ever needed later, the swap is localized to
  this hooks layer.
- **Zustand** — owns client/UI state only (`src/store/uiStore.ts`):
  selected camera node, selected zone, time-range filter. It never holds
  server data -- that would just re-implement what TanStack Query
  already does. These are deliberately two different tools for two
  different kinds of state, not a redundant pairing.
- **react-router-dom** — client-side routing between pages.

## Layout

```
retailVision/
  src/
    api/
      client.ts          # fetch wrapper: base URL, query-string building, ApiError
      detections.ts       # one typed function per server endpoint
      types.ts             # mirrors server/app/schemas/detection.py's response models
    hooks/
      useLiveOccupancy.ts, useRecentDetections.ts, useAggregates.ts   # TanStack Query, polling
    store/
      uiStore.ts          # Zustand: UI-only state
    components/
      layout/     AppShell (sidebar + routed content), Sidebar
      occupancy/  OccupancyGrid, OccupancyCard
      detections/ RecentActivityFeed
      common/     LoadingState, ErrorState
    pages/
      OverviewPage.tsx     # "/" -- live occupancy + recent activity, fully wired
      DetectionsPage.tsx    # "/detections" -- placeholder, full table lands in a later ticket
      ZonesPage.tsx           # "/zones" -- placeholder, needs zone config (EP-5) first
      NotFoundPage.tsx         # catch-all 404
    App.tsx              # QueryClientProvider + route table
```

`api/types.ts` is hand-kept in sync with the server's Pydantic response
models -- there's no shared codegen between the two languages, so a
server schema change means updating both sides by hand.

## Local development

Requires **Node 22** (a `.nvmrc` pins this — run `nvm use` if you have
[nvm](https://github.com/nvm-sh/nvm) installed). The Vite/oxlint
toolchain here needs Node `^20.19 || >=22.12`; an older Node silently
fails to install their platform-specific native bindings rather than
erroring clearly, so don't try to force it onto an older Node without
checking this first.

```
cd retailVision
nvm use                     # if using nvm
npm install
cp .env.example .env        # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev                 # http://localhost:5173
```

Requires the server (`docs/server.md`) running and reachable at
`VITE_API_BASE_URL` -- CORS on the server's dev defaults already allow
`http://localhost:5173`. With no detections ingested yet, the Overview
page's occupancy grid and activity feed just show an empty state, not
an error.

```
npm run build      # tsc -b && vite build -- production build
npm run lint        # oxlint
```
