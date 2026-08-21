// The headline strip: five figures that answer "how is the space doing"
// before the reader scrolls. Live occupancy is the only instantaneous
// number; the rest summarize the last 24 hours, and each carries its
// honesty caveat in the detail line rather than presenting a rounder
// story than the data supports.
//
// Hairline separators come from a gap-px grid over the line colour, which
// survives any wrap point -- divide-x utilities assume a single row.

import { useSummary } from "../../hooks/useSummary";
import { useZoneOccupancy } from "../../hooks/useZoneOccupancy";
import { StatTile } from "../common/ui";

/** Formats seconds as a compact duration, e.g. "48s" or "3m 05s". */
function formatDwell(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
}

/** Formats a bucket start as the local hour it opens, e.g. "14:00". */
function formatHour(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/** Renders the five-figure KPI strip across the top of the Overview page. */
export function KpiRow() {
  const { data: zones } = useZoneOccupancy();
  const { data: summary } = useSummary();

  const occupancyNow = zones?.reduce((sum, zone) => sum + zone.total, 0);

  const emotions = summary?.emotion_distribution ?? {};
  const emotionTotal = Object.values(emotions).reduce((sum, count) => sum + count, 0);
  const positive = (emotions["happy"] ?? 0) + (emotions["surprise"] ?? 0);
  const positiveShare = emotionTotal > 0 ? Math.round((positive / emotionTotal) * 100) : null;

  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-[var(--app-line)] bg-[var(--app-line)] lg:grid-cols-5">
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="In zones now"
          value={occupancyNow ?? "–"}
          unit={occupancyNow === 1 ? "person" : "people"}
          detail="deduplicated across cameras"
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="People seen · 24h"
          value={summary?.unique_people ?? "–"}
          detail="per camera; overlap counts twice"
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="Avg dwell · 24h"
          value={summary?.avg_dwell_seconds != null ? formatDwell(summary.avg_dwell_seconds) : "–"}
          detail="time present per visit"
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="Positive mood · 24h"
          value={positiveShare != null ? `${positiveShare}%` : "–"}
          detail={emotionTotal > 0 ? `happy + surprise of ${emotionTotal} reads` : "no emotion reads yet"}
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="Busiest hour · 24h"
          value={summary?.busiest_hour_start ? formatHour(summary.busiest_hour_start) : "–"}
          detail={
            summary?.busiest_hour_start
              ? `${summary.busiest_hour_people} ${summary.busiest_hour_people === 1 ? "person" : "people"} seen`
              : "no traffic recorded"
          }
        />
      </div>
    </div>
  );
}
