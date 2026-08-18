// Historical view over the detection record: how busy it has been, what the
// emotional mix looked like, and who was there.
//
// The time range drives the bucket size too, since the two cannot be chosen
// independently -- 5-minute buckets over a week would be thousands of points
// on a chart 720 pixels wide, and hourly buckets over an hour would be one.
// Each preset therefore carries the bucket that keeps the series readable.

import { useMemo, useState } from "react";
import { useAggregates } from "../../hooks/useAggregates";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";
import { Card, CardHeader, SegmentedControl } from "../common/ui";
import { DistributionBars } from "./DistributionBars";
import { StackedAreaChart } from "./StackedAreaChart";

interface Range {
  label: string;
  hours: number;
  window: string;
}

const RANGES: Range[] = [
  { label: "1h", hours: 1, window: "5m" },
  { label: "6h", hours: 6, window: "15m" },
  { label: "24h", hours: 24, window: "1h" },
  { label: "7d", hours: 24 * 7, window: "6h" },
];

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

/** Renders the historical charts with a shared time-range selector, optionally scoped to one zone. */
export function HistoricalCharts({ zoneId }: { zoneId?: string } = {}) {
  const [range, setRange] = useState<Range>(RANGES[2]);

  const since = useMemo(
    () => new Date(Date.now() - range.hours * 3600_000).toISOString(),
    [range],
  );
  const { data, isPending, isError } = useAggregates({
    window: range.window,
    since,
    zone_id: zoneId,
  });

  const selector = (
    <SegmentedControl
      ariaLabel="Time range"
      options={RANGES.map((option) => ({ value: option.label, label: option.label }))}
      value={range.label}
      onChange={(label) => setRange(RANGES.find((option) => option.label === label) ?? RANGES[2])}
    />
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
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-[var(--app-ink-muted)]">
          Counts every detection event, so one person present for a while contributes many.
        </p>
        {selector}
      </div>
      {body}
    </div>
  );
}
