// The headline strip: five figures that answer "how is the space doing"
// before the reader scrolls. Live occupancy is the only instantaneous
// number; the rest summarize the last 24 hours, and each carries its
// honesty caveat in the detail line rather than presenting a rounder
// story than the data supports.
//
// Where history allows, a figure also carries how it moved against the
// same 24-hour window one week earlier -- the comparison that controls
// for day-of-week, which day-over-day does not. Deltas render in neutral
// ink: more people is not inherently good or bad, so status colour would
// editorialize.
//
// Hairline separators come from a gap-px grid over the line colour, which
// survives any wrap point -- divide-x utilities assume a single row.

import { useMemo } from "react";
import type { Summary } from "../../api/types";
import { useSummary } from "../../hooks/useSummary";
import { useZoneOccupancy } from "../../hooks/useZoneOccupancy";
import { StatTile } from "../common/ui";

const DAY_MS = 24 * 3600_000;
const WEEK_MS = 7 * DAY_MS;

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

/** The happy+surprise share of a summary's emotion reads, as a 0-100 percentage, or null without data. */
function positiveShare(summary: Summary | undefined): number | null {
  if (!summary) return null;
  const total = Object.values(summary.emotion_distribution).reduce((sum, count) => sum + count, 0);
  if (total === 0) return null;
  const positive = (summary.emotion_distribution["happy"] ?? 0) + (summary.emotion_distribution["surprise"] ?? 0);
  return Math.round((positive / total) * 100);
}

/** Renders "vs last week" for a metric, or null when last week has nothing to compare against. */
function versusLastWeek(current: number | null, previous: number | null, unit = "%"): string | null {
  if (current == null || previous == null || previous === 0) return null;
  const change = unit === "%" ? Math.round(((current - previous) / previous) * 100) : current - previous;
  if (change === 0) return "level with last week";
  return `${change > 0 ? "▲" : "▼"} ${Math.abs(change)}${unit} vs last week`;
}

/** Renders the five-figure KPI strip across the top of the Overview page. */
export function KpiRow() {
  const { data: zones } = useZoneOccupancy();
  const { data: summary } = useSummary();
  // The same 24-hour window, one week earlier. Memoized so the query key is
  // stable across renders instead of refetching on every clock tick.
  const lastWeekRange = useMemo(() => {
    const now = Date.now();
    return {
      since: new Date(now - WEEK_MS - DAY_MS).toISOString(),
      until: new Date(now - WEEK_MS).toISOString(),
    };
  }, []);
  const { data: lastWeek } = useSummary(lastWeekRange);
  const hasHistory = (lastWeek?.total_detections ?? 0) > 0;

  const occupancyNow = zones?.reduce((sum, zone) => sum + zone.total, 0);
  const share = positiveShare(summary);

  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-[var(--app-line)] bg-[var(--app-line)] lg:grid-cols-5">
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="In zones now"
          value={occupancyNow ?? "–"}
          unit={occupancyNow === 1 ? "person" : "people"}
          detail={
            summary && summary.peak_occupancy > 0
              ? `peak in 24h: ${summary.peak_occupancy}`
              : "deduplicated across cameras"
          }
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="People seen · 24h"
          value={summary?.unique_people ?? "–"}
          detail={
            (hasHistory && versusLastWeek(summary?.unique_people ?? null, lastWeek?.unique_people ?? null)) ||
            "per camera; overlap counts twice"
          }
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="Avg dwell · 24h"
          value={summary?.avg_dwell_seconds != null ? formatDwell(summary.avg_dwell_seconds) : "–"}
          detail={
            (hasHistory &&
              versusLastWeek(summary?.avg_dwell_seconds ?? null, lastWeek?.avg_dwell_seconds ?? null)) ||
            "time present per visit"
          }
        />
      </div>
      <div className="bg-[var(--app-surface)] p-4">
        <StatTile
          label="Positive mood · 24h"
          value={share != null ? `${share}%` : "–"}
          detail={
            (hasHistory && versusLastWeek(share, positiveShare(lastWeek), " pts")) ||
            (share != null ? "share of happy + surprise" : "no emotion reads yet")
          }
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
