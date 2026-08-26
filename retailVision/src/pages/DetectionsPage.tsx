// The Visits page: people, not events.
//
// One row here is one person's stay as seen by one camera -- their track's
// records folded into arrival, duration, place and mood -- because since
// per-person emission landed, the raw event stream is machinery rather than
// information. The page leads with what the visits say in aggregate (how
// long people stay, what mood they left in) and only then lists them; the
// raw events remain reachable through the API for debugging, not here.

import { useMemo, useState } from "react";
import type { Visit } from "../api/types";
import { DistributionBars } from "../components/charts/DistributionBars";
import { HourProfile } from "../components/charts/HourProfile";
import { ErrorState } from "../components/common/ErrorState";
import { LoadingState } from "../components/common/LoadingState";
import {
  CONTROL_CLASS,
  Card,
  CardHeader,
  EmptyState,
  Field,
  PageHeader,
  SegmentedControl,
  StatTile,
} from "../components/common/ui";
import { useLiveOccupancy } from "../hooks/useLiveOccupancy";
import { useVisits } from "../hooks/useVisits";
import { useZoneOccupancy } from "../hooks/useZoneOccupancy";

type RangeKey = "1h" | "24h" | "7d";

const RANGES: Array<{ value: RangeKey; label: string; hours: number }> = [
  { value: "1h", label: "1h", hours: 1 },
  { value: "24h", label: "24h", hours: 24 },
  { value: "7d", label: "7d", hours: 24 * 7 },
];

const PAGE_SIZE = 25;
const FETCH_LIMIT = 500;

// Duration buckets are a scale, so they render in this order, not by size.
const DURATION_BUCKETS: Array<{ label: string; upTo: number }> = [
  { label: "under 30s", upTo: 30 },
  { label: "30s – 2m", upTo: 120 },
  { label: "2 – 5m", upTo: 300 },
  { label: "5 – 15m", upTo: 900 },
  { label: "over 15m", upTo: Infinity },
];

/** Formats seconds as a compact duration, e.g. "48s" or "3m 05s". */
function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}

/** Formats a timestamp for a table row, where the date matters as much as the time. */
function formatRow(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** The median of a sorted-or-not list of numbers, or null when empty. */
function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Counts visits into the fixed duration buckets, keeping every bucket present so the scale reads whole. */
function durationHistogram(visits: Visit[]): Record<string, number> {
  const counts: Record<string, number> = Object.fromEntries(DURATION_BUCKETS.map((b) => [b.label, 0]));
  for (const visit of visits) {
    const bucket = DURATION_BUCKETS.find((b) => visit.duration_seconds < b.upTo) ?? DURATION_BUCKETS[DURATION_BUCKETS.length - 1];
    counts[bucket.label] += 1;
  }
  return counts;
}

/** Renders the Visits page: per-person stays, what they say in aggregate, then the list. */
export function DetectionsPage() {
  const [range, setRange] = useState<RangeKey>("24h");
  const [cameraNodeId, setCameraNodeId] = useState("");
  const [zoneId, setZoneId] = useState("");
  const [page, setPage] = useState(0);

  const since = useMemo(() => {
    const hours = RANGES.find((r) => r.value === range)?.hours ?? 24;
    return new Date(Date.now() - hours * 3600_000).toISOString();
  }, [range]);

  const { data, isPending, isError } = useVisits({
    since,
    limit: FETCH_LIMIT,
    camera_node_id: cameraNodeId || undefined,
    zone_id: zoneId || undefined,
  });

  // Filter options come from what is actually reporting, so the dropdowns
  // cannot offer a camera or zone that would return nothing.
  const { data: occupancy } = useLiveOccupancy();
  const { data: zones } = useZoneOccupancy();
  const cameraOptions = [...new Set((occupancy ?? []).map((row) => row.camera_node_id))].sort();
  const zoneOptions = [...new Set((zones ?? []).map((zone) => zone.zone_id))].sort();

  /** Resets to the first page whenever a filter changes, so the view is never a page past the end. */
  function withReset<T>(setter: (value: T) => void) {
    return (value: T) => {
      setter(value);
      setPage(0);
    };
  }

  const visits = data ?? [];
  const durations = visits.map((v) => v.duration_seconds);
  const medianStay = median(durations);
  const longest = durations.length ? Math.max(...durations) : null;
  const moods = visits.reduce<Record<string, number>>((totals, visit) => {
    totals[visit.dominant_emotion] = (totals[visit.dominant_emotion] ?? 0) + 1;
    return totals;
  }, {});

  const pageCount = Math.max(1, Math.ceil(visits.length / PAGE_SIZE));
  const visible = visits.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const filters = (
    <div className="flex flex-wrap items-end gap-4">
      <Field label="Range">
        <SegmentedControl
          ariaLabel="Time range"
          options={RANGES.map(({ value, label }) => ({ value, label }))}
          value={range}
          onChange={withReset(setRange)}
        />
      </Field>

      <Field label="Camera">
        <select
          className={CONTROL_CLASS}
          value={cameraNodeId}
          onChange={(e) => withReset(setCameraNodeId)(e.target.value)}
        >
          <option value="">All cameras</option>
          {cameraOptions.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Zone">
        <select className={CONTROL_CLASS} value={zoneId} onChange={(e) => withReset(setZoneId)(e.target.value)}>
          <option value="">All zones</option>
          {zoneOptions.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </Field>
    </div>
  );

  let body: React.ReactNode;
  if (isPending) {
    body = <LoadingState label="Loading visits…" />;
  } else if (isError) {
    body = <ErrorState message="Couldn't load visits -- is the server running?" />;
  } else if (visits.length === 0) {
    body = (
      <EmptyState
        title="No visits in this range."
        hint="A visit needs records carrying a track -- nodes ship those while running with --server-url."
      />
    );
  } else {
    body = (
      <div className="flex flex-col gap-6">
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-[var(--app-line)] bg-[var(--app-line)] lg:grid-cols-4">
          <div className="bg-[var(--app-surface)] p-4">
            <StatTile
              label="Visits"
              value={visits.length === FETCH_LIMIT ? `${FETCH_LIMIT}+` : visits.length}
              detail="per camera; overlap counts twice"
            />
          </div>
          <div className="bg-[var(--app-surface)] p-4">
            <StatTile label="Median stay" value={medianStay != null ? formatDuration(medianStay) : "–"} />
          </div>
          <div className="bg-[var(--app-surface)] p-4">
            <StatTile label="Longest stay" value={longest != null ? formatDuration(longest) : "–"} />
          </div>
          <div className="bg-[var(--app-surface)] p-4">
            <StatTile
              label="Left in a good mood"
              value={
                visits.length
                  ? `${Math.round((((moods["happy"] ?? 0) + (moods["surprise"] ?? 0)) / visits.length) * 100)}%`
                  : "–"
              }
              detail="visits whose dominant mood was happy or surprise"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader title="How long people stay" description="Visits by duration. Short spikes are pass-throughs; long tails are workstations." />
            <div className="p-4">
              <DistributionBars caption="Visits by duration" distribution={durationHistogram(visits)} order="given" />
            </div>
          </Card>
          <Card>
            <CardHeader title="Dominant mood per visit" description="Each visit counted once, by the mood it mostly showed." />
            <div className="p-4">
              <DistributionBars caption="Visits by dominant mood" distribution={moods} />
            </div>
          </Card>
        </div>

        <Card>
          <CardHeader
            title="Rhythm of the day"
            description="Average distinct people per hour, over the last 7 days. The peak hour is highlighted."
          />
          <div className="p-4">
            <HourProfile zoneId={zoneId || undefined} />
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Visit log"
            description="One row per person's stay, newest first. IDs are anonymous and per-camera."
          />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[52rem] text-sm">
              <caption className="sr-only">Visits, one row per person's stay, newest first.</caption>
              <thead>
                <tr className="border-b border-[var(--app-line)] text-left text-[0.7rem] uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
                  <th scope="col" className="px-4 py-2 font-medium">Arrived</th>
                  <th scope="col" className="px-4 py-2 text-right font-medium">Stayed</th>
                  <th scope="col" className="px-4 py-2 font-medium">Camera</th>
                  <th scope="col" className="px-4 py-2 font-medium">Zone</th>
                  <th scope="col" className="px-4 py-2 font-medium">Age</th>
                  <th scope="col" className="px-4 py-2 font-medium">Gender</th>
                  <th scope="col" className="px-4 py-2 font-medium">Mostly</th>
                  <th scope="col" className="px-4 py-2 text-right font-medium">Events</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((visit) => (
                  <tr
                    key={`${visit.camera_node_id}-${visit.track_id}`}
                    className="border-b border-[var(--app-line)] last:border-0"
                  >
                    <td className="whitespace-nowrap px-4 py-2 tabular-nums text-[var(--app-ink-secondary)]">
                      {formatRow(visit.first_seen)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2 text-right tabular-nums text-[var(--app-ink)]">
                      {formatDuration(visit.duration_seconds)}
                    </td>
                    <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{visit.camera_node_id}</td>
                    <td className="px-4 py-2 text-[var(--app-ink-secondary)]">
                      {visit.zone_id ?? <span className="text-[var(--app-ink-muted)]">—</span>}
                    </td>
                    <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{visit.age_group}</td>
                    <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{visit.gender}</td>
                    <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{visit.dominant_emotion}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-[var(--app-ink-muted)]">{visit.events}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between px-4 py-3 text-xs text-[var(--app-ink-muted)]">
            <span>
              Page {page + 1} of {pageCount}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
                className={`${CONTROL_CLASS} px-3 py-1 text-xs disabled:opacity-40`}
              >
                Previous
              </button>
              <button
                type="button"
                disabled={page >= pageCount - 1}
                onClick={() => setPage((p) => p + 1)}
                className={`${CONTROL_CLASS} px-3 py-1 text-xs disabled:opacity-40`}
              >
                Next
              </button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Visits"
        description="One row per person's stay, folded from their track's records -- not one row per detection event."
        actions={filters}
      />
      {body}
    </div>
  );
}
