// One live occupancy tile: a zone or camera node's current count.

import type { Occupancy } from "../../api/types";

/** Formats an ISO timestamp as a short relative "Xs/m/h ago" string. */
function relativeTime(isoTimestamp: string): string {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(isoTimestamp).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

/** Renders one occupancy card: location key, current count, and last-updated time. */
export function OccupancyCard({ occupancy }: { occupancy: Occupancy }) {
  const isZone = occupancy.zone_id !== null;
  return (
    <div className="rounded-lg border border-[var(--app-line)] bg-[var(--app-surface)] p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-[var(--app-ink-muted)]">{occupancy.key}</span>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            isZone
              ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400"
              : "bg-[var(--app-page)] text-[var(--app-ink-muted)]"
          }`}
        >
          {isZone ? "zone" : "camera node"}
        </span>
      </div>
      <div className="mt-3 font-mono text-3xl font-semibold tabular-nums text-[var(--app-ink)]">
        {occupancy.count ?? "—"}
      </div>
      <div className="mt-1 text-xs text-[var(--app-ink-muted)]">Updated {relativeTime(occupancy.timestamp)}</div>
    </div>
  );
}
