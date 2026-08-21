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
      DetectionsPage.tsx    # "/detections" -- the Visits page, one row per person's stay
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


## Charts

Chart colours live as CSS custom properties in `src/index.css` under
`.viz-root`, and chart components are written against those roles rather
than raw hex, so light and dark swap in one place. Both modes are chosen
deliberately -- the dark values are the same hues re-stepped for a dark
surface, not an inversion, which would drift out of the lightness band
where colours stay distinguishable to colour-blind readers.

The categorical order is fixed and assigned by slot, never cycled, so a
series keeps its colour when a filter changes how many series are on
screen.

Charts are hand-rolled SVG (`src/components/charts/`) rather than a
charting library: there are three chart forms, all small, and a library
would bring its own theming layer to reconcile with these tokens plus a
dependency to keep current. Revisit if the chart set grows much past this.

`HistoricalCharts` puts the period and the grouping in a side panel as two
controls rather than a row of fixed presets, since the useful question is
often "the last week, by day" and a preset list has to guess which pairings
matter.

They are not fully independent, though: an hourly view of a month is 720
points on a chart 720 pixels wide, and a day grouped by day is a single
point, which is a number rather than a series. Pairings outside roughly
3-200 points are shown disabled with the reason, and changing the period
moves the grouping to a usable one rather than leaving a selection checked
that is not what is being drawn.

The same panel carries age, gender and emotion filters, which are applied
server-side so every chart on the page answers the same question. Their
options come from a second, unfiltered query over the same range rather
than from the filtered data: reading them off the filtered response would
mean selecting one emotion removed every other emotion from the list,
stranding the reader with no way back. Whenever nothing is filtered the
two queries have identical keys, so TanStack Query serves both from one
request. A single button clears every dimension at once, and states which
of the two it is doing -- "Showing everything" when nothing is set.

Hovering a time chart opens a floating box beside the cursor. It floats
rather than writing into a fixed block below the chart because the
reader's eye is already on the bucket they are asking about.

The hover reads the cursor's vertical position as well as its horizontal
one, so pointing at a band asks about *that series* rather than about the
bucket as a whole: the box leads with that one series -- its count, its
share of the bucket, and how it moved since the previous bucket -- the
other bands drop to a low opacity, and the focused band's own upper edge
is stroked in its colour. Sweeping sideways along a band therefore traces
that series' history.

That is the question a stacked chart is otherwise bad at answering. A
band's *thickness* is easy to read, but its *position* is not, because
every band above the first rides on the ones beneath it, so its outline
carries their movement as well as its own. Isolating one band removes
that borrowed motion.

Pointing above the stack, where no band lies, falls back to summarizing
the whole bucket: total plus every series with its share. A single-series
chart never dims, having nothing to disambiguate.

Both modes end with the average dwell and leading age/gender for that
bucket. On the detections chart, whose single series has no colour to
read, the tooltip is also where that bucket's emotional mix appears.

Hovering is momentary, so the legend entries are also buttons that pin one
series. A reader following a single emotion should not have to keep the
cursor inside a band that may be thin or interrupted to hold their place:
a pinned series stays isolated while the cursor goes elsewhere, including
off the chart. Hovering another band still previews it, and the pin takes
back over as soon as the cursor leaves -- so it is a way to look around
without losing what was deliberately chosen. Clicking the pinned entry
again releases it, so the control that turned isolation on is the one that
turns it off, and a "Show all" appears alongside while anything is pinned.

The pin is held by series *name*, not position. The demographic filters
change which series exist, and an index would quietly come to mean a
different emotion; a name that is no longer present simply stops matching,
which is the right outcome for a series that has been filtered away.

The legend is permanent rather than something the tooltip replaces
-- identity must not rest on colour alone whether or not anyone is
hovering -- and dims in step with the bands so the focused series stays
identifiable there too. Making it the isolation control also hands that
interaction to keyboard users, who have no hover to offer.


## Design tokens

Colours live as CSS custom properties on `:root` in `src/index.css`, split
into interface roles (`--app-*`) and chart roles (`--viz-*`), and components
reference them as Tailwind arbitrary values (`bg-[var(--app-surface)]`).
One class therefore covers light and dark, rather than every element
carrying a `dark:` twin, and a colour changes in exactly one place.

The neutrals are a cool gray ramp on white surfaces -- the grammar of
operations dashboards, where the chrome is deliberately unremarkable so
the numbers are the loudest thing on the page. One blue carries every
interactive affordance; status colours are reserved for status and always
travel with a text label. Figures render in tabular numerals so columns
of numbers align. Interface and chart colours come from the same set
deliberately: a chart wearing a different palette to the panel around it
reads as two systems bolted together.

Both modes are chosen, not derived. Dark is the same hues re-stepped for a
dark surface -- inverting a light palette drifts the series colours out of
the lightness band where they stay distinguishable to colour-blind readers.
The categorical order is fixed by slot and never cycled, so a series keeps
its colour when a filter changes how many series are on screen.

`src/components/common/ui.tsx` holds the shared shapes -- `Card`,
`PageHeader`, `StatTile`, `Badge`, `EmptyState`, `SegmentedControl` -- so
panel treatment and spacing are defined once rather than as repeated
Tailwind strings.

## Pages

- **Overview** (`/`) -- live per-zone headcount, camera feeds, recent
  activity. Ordered by how often each is looked at rather than by how the
  system is built.
- **Zones** (`/zones`) -- one zone at a time: current headcount, which
  cameras are contributing it, and that zone's history. The per-camera
  figures sit beside the headcount rather than behind it, because a zone
  reading four looks equally healthy whether three cameras agree or two have
  silently stopped reporting.
Page sections on the Overview and Zones pages are reorderable: hovering a
section shows up/down controls at its top-right, and the chosen order
persists per-browser alongside the alert settings -- how a page is
arranged is a property of the person reading it, not of the deployment.
Deliberately buttons rather than drag-and-drop: reordering is a
set-and-forget act, not a frequent gesture, so it does not justify a drag
library, and buttons work for keyboard users. Sections added by later
versions append in their default position rather than invalidating a
stored order. The KPI strip and page headers stay fixed: a headline that
can wander stops being a headline.

Four insight features sit on top of the pages:

- **Floor map** (Zones page) -- the zone's surveyed polygon with an hour
  of position heat and dots for people there right now. Heat is a grid of
  cells whose opacity follows sighting density: a sequential encoding in
  the single accent hue, since magnitude on a map is one measure, not
  categories. Live dots are ink with a surface ring so they read against
  any heat level. Geometry arrives via the server's `/zones/geometry`,
  uploaded by nodes at startup; the empty state says to restart nodes
  after upgrading. The map is explorable: wheel or buttons zoom (up to
  8x, around the cursor), dragging pans while zoomed, and hovering a
  cell or a person opens the same floating detail box the time charts
  use -- a cell reports its world coordinate, sighting count and share,
  dominant emotion and last-seen; a person reports their labels and how
  recently they were seen. The scale bar stays two real meters long, so
  zooming visibly stretches it. Positions inherit the head-height
  assumption, and the section heading says so. The polygon is the convex
  hull, so markers mounted on interior partitions are not corners --
  fewer sides than markers is the dent fix working, not data loss.
- **Rhythm of the day** (Visits page) -- average distinct people per
  local hour, folded client-side from a week of hourly aggregates.
  Columns, not the horizontal bars used elsewhere: hours are a cyclic
  scale read like a clock face, not ranked categories. Only the peak
  carries a direct label.
- **Crowding alerts** -- a threshold (N people for M minutes) set on the
  Overview page, persisted per-browser in localStorage: who wants to be
  warned is a property of the person watching, not the deployment. The
  banner renders in the shell so it survives navigation; the "held for M
  minutes" clock runs client-side off the polls the pages already make.
- **Week-over-week deltas** on the KPI tiles -- the same 24-hour window
  one week earlier, which controls for day-of-week the way day-over-day
  does not. Deltas render in neutral ink: more people is not inherently
  good or bad, so status colour would editorialize. They appear only once
  last week has data to compare against.

- **Visits** (`/detections`) -- one row per person's stay, folded from their
  track's records by the server's `/visits` endpoint: arrival, duration,
  zone, dominant mood. The page leads with the aggregate story -- a
  stay-duration histogram (buckets kept in scale order, not size order),
  the mood mix with each visit counted once, median and longest stay --
  and only then the list, filterable by camera, zone and time range and
  paged client-side. The raw per-event stream is deliberately not shown:
  since per-person emission landed it is machinery, and it remains
  reachable through the API.
