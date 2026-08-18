// Historical view over the detection record: how busy it has been, what the
// emotional mix looked like, and who was there.
//
// Range and granularity are separate controls rather than four fixed presets,
// because the useful question is often "the last week, by day" and a preset
// list has to guess which pairings matter. They cannot be fully independent
// though: an hourly view of a month is 720 points on a chart 720 pixels wide,
// which is a smear rather than a reading, so those combinations are offered
// but disabled with the reason given.

import { useMemo, useState } from "react";
import { useAggregates } from "../../hooks/useAggregates";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";
import { Card, CardHeader } from "../common/ui";
import { DistributionBars } from "./DistributionBars";
import { StackedAreaChart } from "./StackedAreaChart";

type RangeKey = "24h" | "7d" | "30d";
type Granularity = "hour" | "day";

const RANGES: Array<{ key: RangeKey; label: string; hours: number }> = [
  { key: "24h", label: "Last 24 hours", hours: 24 },
  { key: "7d", label: "Last 7 days", hours: 24 * 7 },
  { key: "30d", label: "Last 30 days", hours: 24 * 30 },
];

const GRANULARITIES: Array<{ key: Granularity; label: string; window: string; hours: number }> = [
  { key: "hour", label: "Per hour", window: "1h", hours: 1 },
  { key: "day", label: "Per day", window: "1d", hours: 24 },
];

// Past this the points are narrower than the marks drawn on them, so the chart
// stops being readable however correct the data is. Below the minimum there is
// no shape to read either -- a day grouped by day is one point, which is a
// number, not a series.
const MAX_POINTS = 200;
const MIN_POINTS = 3;

/** Whether a range/granularity pairing produces a chart worth drawing. */
function isPlottable(rangeHours: number, granularityHours: number): boolean {
  const points = pointCount(rangeHours, granularityHours);
  return points >= MIN_POINTS && points <= MAX_POINTS;
}

/** Collects every category name seen across the buckets, so a series is not dropped when absent from one. */
function categoryNames(buckets: Array<Record<string, number>>): string[] {
  const names = new Set<string>();
  for (const bucket of buckets) {
    for (const name of Object.keys(bucket)) names.add(name);
  }
  return [...names].sort();
}

/** Sums a distribution across every bucket in the range. */
function totalDistribution(buckets: Array<Record<string, number>>): Record<string, number> {
  const totals: Record<string, number> = {};
  for (const bucket of buckets) {
    for (const [name, count] of Object.entries(bucket)) {
      totals[name] = (totals[name] ?? 0) + count;
    }
  }
  return totals;
}

/** How many buckets a range/granularity pairing would produce. */
function pointCount(rangeHours: number, granularityHours: number): number {
  return Math.ceil(rangeHours / granularityHours);
}

/** Renders the historical charts alongside a range panel, optionally scoped to one zone. */
export function HistoricalCharts({ zoneId }: { zoneId?: string } = {}) {
  const [rangeKey, setRangeKey] = useState<RangeKey>("7d");
  const [granularity, setGranularity] = useState<Granularity>("day");

  const range = RANGES.find((r) => r.key === rangeKey) ?? RANGES[1];
  const grain = GRANULARITIES.find((g) => g.key === granularity) ?? GRANULARITIES[1];

  const since = useMemo(
    () => new Date(Date.now() - range.hours * 3600_000).toISOString(),
    [range],
  );
  const { data, isPending, isError } = useAggregates({
    window: grain.window,
    since,
    zone_id: zoneId,
  });

  const points = pointCount(range.hours, grain.hours);

  const panel = (
    <Card className="lg:sticky lg:top-7">
      <CardHeader title="Range" />
      <div className="flex flex-col gap-5 p-4">
        <fieldset className="flex flex-col gap-1.5">
          <legend className="mb-1.5 text-[0.7rem] font-medium uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
            Period
          </legend>
          {RANGES.map((option) => (
            <label
              key={option.key}
              className="flex cursor-pointer items-center gap-2 text-sm text-[var(--app-ink-secondary)]"
            >
              <input
                type="radio"
                name="range-period"
                checked={option.key === rangeKey}
                onChange={() => {
                  setRangeKey(option.key);
                  // Keep the selection valid rather than showing a different
                  // granularity than the one the reader can see checked.
                  if (!isPlottable(option.hours, grain.hours)) {
                    const usable = GRANULARITIES.find((g) => isPlottable(option.hours, g.hours));
                    if (usable) setGranularity(usable.key);
                  }
                }}
                className="accent-[var(--app-accent)]"
              />
              {option.label}
            </label>
          ))}
        </fieldset>

        <fieldset className="flex flex-col gap-1.5 border-t border-[var(--app-line)] pt-4">
          <legend className="mb-1.5 text-[0.7rem] font-medium uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
            Group by
          </legend>
          {GRANULARITIES.map((option) => {
            const resulting = pointCount(range.hours, option.hours);
            const usable = isPlottable(range.hours, option.hours);
            const reason = resulting > MAX_POINTS ? "too many points" : "too few points";
            return (
              <label
                key={option.key}
                className={`flex items-center gap-2 text-sm ${
                  usable
                    ? "cursor-pointer text-[var(--app-ink-secondary)]"
                    : "cursor-not-allowed text-[var(--app-ink-muted)]"
                }`}
                title={usable ? undefined : `${resulting} point(s) over this period -- ${reason}`}
              >
                <input
                  type="radio"
                  name="range-granularity"
                  checked={option.key === granularity}
                  disabled={!usable}
                  onChange={() => setGranularity(option.key)}
                  className="accent-[var(--app-accent)]"
                />
                {option.label}
                {!usable && <span className="text-xs">({reason})</span>}
              </label>
            );
          })}
        </fieldset>

        <p className="border-t border-[var(--app-line)] pt-3 text-xs text-[var(--app-ink-muted)]">
          {points} point{points === 1 ? "" : "s"}, one per {granularity}.
        </p>
        <p className="text-xs text-[var(--app-ink-muted)]">
          Counts every detection event, so one person present for a while contributes many.
        </p>
      </div>
    </Card>
  );

  let body: React.ReactNode;
  if (isPending) {
    body = <LoadingState label="Loading history…" />;
  } else if (isError) {
    body = <ErrorState message="Couldn't load history -- is the server running?" />;
  } else {
    const labels = data.map((bucket) => bucket.bucket_start);
    const emotionNames = categoryNames(data.map((bucket) => bucket.emotion_distribution));

    body = (
      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader title="Detections over time" />
          <div className="p-4">
            <StackedAreaChart
              labels={labels}
              series={[{ name: "detections", values: data.map((bucket) => bucket.detection_count) }]}
              spanHours={range.hours}
            />
          </div>
        </Card>

        <Card>
          <CardHeader title="Emotion over time" />
          <div className="p-4">
            <StackedAreaChart
              labels={labels}
              series={emotionNames.map((name) => ({
                name,
                values: data.map((bucket) => bucket.emotion_distribution[name] ?? 0),
              }))}
              spanHours={range.hours}
            />
          </div>
        </Card>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader title="Age group" />
            <div className="p-4">
              <DistributionBars distribution={totalDistribution(data.map((b) => b.age_group_distribution))} />
            </div>
          </Card>
          <Card>
            <CardHeader title="Gender" />
            <div className="p-4">
              <DistributionBars distribution={totalDistribution(data.map((b) => b.gender_distribution))} />
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_15rem] lg:items-start">
      <div className="min-w-0">{body}</div>
      {panel}
    </div>
  );
}
