// The detection event log: every record a camera node has shipped, filterable
// and paged.
//
// This is the raw layer beneath the charts -- one row per detection event, not
// per person, so someone standing in view for a minute appears many times.
// That distinction is stated on the page rather than assumed, because a table
// of rows invites being read as a list of people.

import { useMemo, useState } from "react";
import { useRecentDetections } from "../hooks/useRecentDetections";
import { useZoneOccupancy } from "../hooks/useZoneOccupancy";
import { ErrorState } from "../components/common/ErrorState";
import { LoadingState } from "../components/common/LoadingState";
import {
  CONTROL_CLASS,
  Card,
  EmptyState,
  Field,
  PageHeader,
  SegmentedControl,
} from "../components/common/ui";
import { useLiveOccupancy } from "../hooks/useLiveOccupancy";

type RangeKey = "1h" | "24h" | "7d" | "all";

const RANGES: Array<{ value: RangeKey; label: string; hours: number | null }> = [
  { value: "1h", label: "1h", hours: 1 },
  { value: "24h", label: "24h", hours: 24 },
  { value: "7d", label: "7d", hours: 24 * 7 },
  { value: "all", label: "All", hours: null },
];

const PAGE_SIZE = 25;
// Fetched in one go and paged client-side: the endpoint has no cursor, and a
// few hundred rows is well within what a single request and the browser can
// hold. Revisit if this ever needs to page through a full day of events.
const FETCH_LIMIT = 500;

/** Formats a timestamp for a table row, where the date matters as much as the time. */
function formatRow(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Renders the filterable, paged detection event log. */
export function DetectionsPage() {
  const [range, setRange] = useState<RangeKey>("24h");
  const [cameraNodeId, setCameraNodeId] = useState("");
  const [zoneId, setZoneId] = useState("");
  const [page, setPage] = useState(0);

  const since = useMemo(() => {
    const hours = RANGES.find((r) => r.value === range)?.hours;
    return hours == null ? undefined : new Date(Date.now() - hours * 3600_000).toISOString();
  }, [range]);

  const { data, isPending, isError } = useRecentDetections({
    limit: FETCH_LIMIT,
    since,
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

  const rows = data ?? [];
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const visible = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Detections"
        description="One row per detection event, not per person -- someone in view for a minute appears many times."
      />

      <Card>
        <div className="flex flex-wrap items-end gap-4 border-b border-[var(--app-line)] px-4 py-3">
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
            <select
              className={CONTROL_CLASS}
              value={zoneId}
              onChange={(e) => withReset(setZoneId)(e.target.value)}
            >
              <option value="">All zones</option>
              {zoneOptions.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </Field>

          <div className="ml-auto text-xs text-[var(--app-ink-muted)]">
            {rows.length === FETCH_LIMIT
              ? `showing the most recent ${FETCH_LIMIT}`
              : `${rows.length} event${rows.length === 1 ? "" : "s"}`}
          </div>
        </div>

        {isPending ? (
          <LoadingState label="Loading detections…" />
        ) : isError ? (
          <div className="p-4">
            <ErrorState message="Couldn't load detections -- is the server running?" />
          </div>
        ) : rows.length === 0 ? (
          <div className="p-4">
            <EmptyState
              title="No detections in this range."
              hint="Camera nodes ship records only while the pipeline is running with --server-url."
            />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[46rem] text-sm">
                <thead>
                  <tr className="border-b border-[var(--app-line)] text-left text-[0.7rem] uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
                    <th className="px-4 py-2 font-medium">Time</th>
                    <th className="px-4 py-2 font-medium">Camera</th>
                    <th className="px-4 py-2 font-medium">Zone</th>
                    <th className="px-4 py-2 font-medium">Position</th>
                    <th className="px-4 py-2 font-medium">Age</th>
                    <th className="px-4 py-2 font-medium">Gender</th>
                    <th className="px-4 py-2 font-medium">Emotion</th>
                    <th className="px-4 py-2 text-right font-medium">Dwell</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((row) => (
                    <tr key={row.id} className="border-b border-[var(--app-line)] last:border-0">
                      <td className="whitespace-nowrap px-4 py-2 tabular-nums text-[var(--app-ink-secondary)]">
                        {formatRow(row.timestamp)}
                      </td>
                      <td className="px-4 py-2 text-[var(--app-ink)]">{row.camera_node_id}</td>
                      <td className="px-4 py-2 text-[var(--app-ink-secondary)]">
                        {row.zone_id ?? <span className="text-[var(--app-ink-muted)]">—</span>}
                      </td>
                      <td className="whitespace-nowrap px-4 py-2 tabular-nums text-[var(--app-ink-muted)]">
                        {row.world_x != null && row.world_y != null
                          ? `${row.world_x.toFixed(1)}, ${row.world_y.toFixed(1)}m`
                          : "—"}
                      </td>
                      <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{row.age_group}</td>
                      <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{row.gender}</td>
                      <td className="px-4 py-2 text-[var(--app-ink-secondary)]">{row.emotion}</td>
                      <td className="px-4 py-2 text-right tabular-nums text-[var(--app-ink-secondary)]">
                        {row.dwell_seconds == null ? "—" : `${row.dwell_seconds.toFixed(1)}s`}
                      </td>
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
          </>
        )}
      </Card>
    </div>
  );
}
