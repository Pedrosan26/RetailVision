// The daily rhythm: average distinct people per hour of the day, folded
// from a week of hourly buckets. "Busiest hour" is one number; the shape
// of the day -- when it ramps, peaks, empties -- is a behaviour, and it is
// the axis a staffing or layout decision actually runs along.
//
// Columns rather than the horizontal bars used elsewhere: hours are a
// cyclic scale read left to right like a clock face, not ranked categories.

import { useMemo } from "react";
import { useAggregates } from "../../hooks/useAggregates";

const WIDTH = 720;
const HEIGHT = 150;
const PADDING = { top: 14, bottom: 22, left: 8, right: 8 };
const LOOKBACK_DAYS = 7;

/** Averages a week of hourly buckets into one value per local hour of day. */
function foldByHour(buckets: Array<{ bucket_start: string; unique_people: number }>): number[] {
  const totals = Array.from({ length: 24 }, () => ({ sum: 0, days: 0 }));
  for (const bucket of buckets) {
    const hour = new Date(bucket.bucket_start).getHours();
    totals[hour].sum += bucket.unique_people;
    totals[hour].days += 1;
  }
  return totals.map(({ sum, days }) => (days === 0 ? 0 : sum / days));
}

/** Renders the average-people-per-hour-of-day profile over the last week. */
export function HourProfile({ zoneId }: { zoneId?: string }) {
  const since = useMemo(() => new Date(Date.now() - LOOKBACK_DAYS * 24 * 3600_000).toISOString(), []);
  const { data } = useAggregates({ window: "1h", since, zone_id: zoneId });

  const perHour = foldByHour(data ?? []);
  const max = Math.max(...perHour, 1);
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
  const step = plotWidth / 24;
  const barWidth = step * 0.7;
  const peakHour = perHour.indexOf(Math.max(...perHour));

  if ((data ?? []).length === 0) {
    return (
      <div className="flex h-[150px] items-center justify-center text-sm text-[var(--app-ink-muted)]">
        No data in the last {LOOKBACK_DAYS} days.
      </div>
    );
  }

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full" role="img" aria-label="Average people by hour of day">
      {perHour.map((value, hour) => {
        const barHeight = (value / max) * plotHeight;
        const x = PADDING.left + hour * step + (step - barWidth) / 2;
        const y = PADDING.top + plotHeight - barHeight;
        const isPeak = hour === peakHour && value > 0;
        return (
          <g key={hour}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={Math.max(barHeight, value > 0 ? 2 : 0)}
              rx={2}
              fill="var(--viz-series-1)"
              opacity={isPeak ? 1 : 0.55}
            >
              <title>{`${String(hour).padStart(2, "0")}:00 -- avg ${value.toFixed(1)} people`}</title>
            </rect>
            {/* Only the peak carries a direct label; a number on every bar is noise. */}
            {isPeak && (
              <text x={x + barWidth / 2} y={y - 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--viz-ink)">
                {value.toFixed(1)}
              </text>
            )}
            {hour % 3 === 0 && (
              <text
                x={PADDING.left + hour * step + step / 2}
                y={HEIGHT - 6}
                textAnchor="middle"
                fontSize={9}
                fill="var(--viz-ink-muted)"
              >
                {String(hour).padStart(2, "0")}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
