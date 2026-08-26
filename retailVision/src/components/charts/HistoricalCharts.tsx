// Historical view over the detection record: how busy it has been, what the
// emotional mix looked like, and who was there.
//
// Range and granularity are separate controls rather than four fixed presets,
// because the useful question is often "the last week, by day" and a preset
// list has to guess which pairings matter. They cannot be fully independent
// though: an hourly view of a month is 720 points on a chart 720 pixels wide,
// which is a smear rather than a reading, so those combinations are offered
// but disabled with the reason given.
//
// The demographic filters narrow which detections are counted at all, server
// side, so every chart on the page answers the same question. Their options
// come from an unfiltered query over the same range rather than from the
// filtered data -- otherwise selecting one emotion would remove every other
// emotion from the list and strand the reader with no way back.

import { useMemo, useState } from "react";
import { type AggregateBucket, EMOTION_ORDER } from "../../api/types";
import { useAggregates } from "../../hooks/useAggregates";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";
import { Card, CardHeader } from "../common/ui";
import { DistributionBars } from "./DistributionBars";
import { StackedAreaChart } from "./StackedAreaChart";

type RangeKey = "24h" | "7d" | "30d";
type Granularity = "hour" | "day";
type Dimension = "age_group" | "gender" | "emotion";

const RANGES: Array<{ key: RangeKey; label: string; hours: number }> = [
  { key: "24h", label: "Last 24 hours", hours: 24 },
  { key: "7d", label: "Last 7 days", hours: 24 * 7 },
  { key: "30d", label: "Last 30 days", hours: 24 * 30 },
];

const GRANULARITIES: Array<{ key: Granularity; label: string; window: string; hours: number }> = [
  { key: "hour", label: "Per hour", window: "1h", hours: 1 },
  { key: "day", label: "Per day", window: "1d", hours: 24 },
];

const DIMENSIONS: Array<{ key: Dimension; label: string; distribution: keyof AggregateBucket }> = [
  { key: "age_group", label: "Age", distribution: "age_group_distribution" },
  { key: "gender", label: "Gender", distribution: "gender_distribution" },
  { key: "emotion", label: "Emotion", distribution: "emotion_distribution" },
];

const NO_FILTERS: Record<Dimension, string[]> = { age_group: [], gender: [], emotion: [] };

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

/** The largest category in a distribution, or null when it is empty. */
function topEntry(distribution: Record<string, number>): [string, number] | null {
  const entries = Object.entries(distribution);
  if (entries.length === 0) return null;
  return entries.reduce((best, entry) => (entry[1] > best[1] ? entry : best));
}

/** Builds the extra tooltip rows shared by every chart: how long people stayed and who they were. */
function commonDetails(bucket: AggregateBucket): Array<{ label: string; value: string }> {
  const rows: Array<{ label: string; value: string }> = [
    { label: "Events behind it", value: String(bucket.detection_count) },
  ];
  if (bucket.avg_dwell_seconds !== null) {
    rows.push({ label: "Avg dwell", value: `${bucket.avg_dwell_seconds.toFixed(1)}s` });
  }
  const age = topEntry(bucket.age_group_distribution);
  if (age) rows.push({ label: "Top age", value: `${age[0]} (${age[1]})` });
  const gender = topEntry(bucket.gender_distribution);
  if (gender) rows.push({ label: "Top gender", value: `${gender[0]} (${gender[1]})` });
  return rows;
}

/** Renders the historical charts alongside a range and filter panel, optionally scoped to one zone. */
export function HistoricalCharts({ zoneId }: { zoneId?: string } = {}) {
  const [rangeKey, setRangeKey] = useState<RangeKey>("7d");
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [filters, setFilters] = useState<Record<Dimension, string[]>>(NO_FILTERS);

  const range = RANGES.find((r) => r.key === rangeKey) ?? RANGES[1];
  const grain = GRANULARITIES.find((g) => g.key === granularity) ?? GRANULARITIES[1];

  const since = useMemo(
    () => new Date(Date.now() - range.hours * 3600_000).toISOString(),
    [range],
  );

  const scope = { window: grain.window, since, zone_id: zoneId };
  const { data, isPending, isError } = useAggregates({ ...scope, ...filters });
  // Same range, no demographic filters: the stable list of what exists to pick
  // from. Identical to the query above whenever nothing is filtered, so the
  // cache serves both from one request in the common case.
  const { data: unfiltered } = useAggregates(scope);

  const points = pointCount(range.hours, grain.hours);
  const activeCount = DIMENSIONS.reduce((sum, d) => sum + filters[d.key].length, 0);

  /** Adds or removes one value from a dimension's filter. */
  function toggle(dimension: Dimension, value: string) {
    setFilters((current) => {
      const chosen = current[dimension];
      return {
        ...current,
        [dimension]: chosen.includes(value) ? chosen.filter((v) => v !== value) : [...chosen, value],
      };
    });
  }

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

        {DIMENSIONS.map((dimension) => {
          const options = categoryNames(
            (unfiltered ?? []).map((bucket) => bucket[dimension.distribution] as Record<string, number>),
          );
          if (options.length === 0) return null;
          return (
            <fieldset key={dimension.key} className="flex flex-col gap-1.5 border-t border-[var(--app-line)] pt-4">
              <legend className="mb-1.5 text-[0.7rem] font-medium uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
                {dimension.label}
              </legend>
              {options.map((option) => (
                <label
                  key={option}
                  className="flex cursor-pointer items-center gap-2 text-sm text-[var(--app-ink-secondary)]"
                >
                  <input
                    type="checkbox"
                    checked={filters[dimension.key].includes(option)}
                    onChange={() => toggle(dimension.key, option)}
                    className="accent-[var(--app-accent)]"
                  />
                  {option}
                </label>
              ))}
            </fieldset>
          );
        })}

        <div className="border-t border-[var(--app-line)] pt-3">
          <button
            type="button"
            onClick={() => setFilters(NO_FILTERS)}
            disabled={activeCount === 0}
            className={`w-full rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
              activeCount === 0
                ? "cursor-not-allowed border-[var(--app-line)] text-[var(--app-ink-muted)]"
                : "border-[var(--app-accent)] bg-[var(--app-accent-wash)] text-[var(--app-accent)]"
            }`}
          >
            {activeCount === 0 ? "Showing everything" : `Clear ${activeCount} filter${activeCount === 1 ? "" : "s"}`}
          </button>
        </div>

        <p className="text-xs text-[var(--app-ink-muted)]">
          {points} point{points === 1 ? "" : "s"}, one per {granularity}.
        </p>
        <p className="text-xs text-[var(--app-ink-muted)]">
          People are counted once per camera. Anyone visible to two cameras still counts twice here.
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

    // The detections chart is a single undifferentiated series, so its tooltip
    // is where the emotional mix for that bucket has to appear.
    const detectionDetails = data.map((bucket) => [
      ...emotionNames
        .filter((name) => (bucket.emotion_distribution[name] ?? 0) > 0)
        .map((name) => ({ label: name, value: String(bucket.emotion_distribution[name]) })),
      ...commonDetails(bucket),
    ]);
    const emotionDetails = data.map(commonDetails);

    body = (
      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader
            title="People over time"
            description="Distinct people, not detection events -- someone who stayed a while counts once."
          />
          <div className="p-4">
            <StackedAreaChart
              labels={labels}
              series={[{ name: "people", values: data.map((bucket) => bucket.unique_people) }]}
              spanHours={range.hours}
              bucketHours={grain.hours}
              unitLabel="people"
              details={detectionDetails}
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
              categoryOrder={EMOTION_ORDER}
              spanHours={range.hours}
              bucketHours={grain.hours}
              details={emotionDetails}
            />
          </div>
        </Card>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader title="Age group" />
            <div className="p-4">
              <DistributionBars
                caption="Detections by age group"
                distribution={totalDistribution(data.map((b) => b.age_group_distribution))}
              />
            </div>
          </Card>
          <Card>
            <CardHeader title="Gender" />
            <div className="p-4">
              <DistributionBars
                caption="Detections by gender"
                distribution={totalDistribution(data.map((b) => b.gender_distribution))}
              />
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
