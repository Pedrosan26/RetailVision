// Scale and path helpers shared by the chart components.
//
// Deliberately hand-rolled rather than pulling in a charting library: the
// dashboard needs three chart forms, all of them small, and a library would
// bring its own theming layer to reconcile with the palette tokens plus a
// dependency to keep current. If the chart set grows much past this, revisit.

/** The eight categorical slots, assigned in fixed order and never cycled. */
export const SERIES_VARS = [
  "var(--viz-series-1)",
  "var(--viz-series-2)",
  "var(--viz-series-3)",
  "var(--viz-series-4)",
  "var(--viz-series-5)",
] as const;

/** Returns the colour for a series slot, or muted ink once the slots run out. */
export function seriesColor(index: number): string {
  return SERIES_VARS[index] ?? "var(--viz-ink-muted)";
}

export interface Scale {
  (value: number): number;
}

/** Builds a linear scale mapping [domainMin, domainMax] onto [rangeMin, rangeMax]. */
export function linearScale(
  domainMin: number,
  domainMax: number,
  rangeMin: number,
  rangeMax: number,
): Scale {
  const span = domainMax - domainMin;
  // A flat series would divide by zero; pin it to the middle of the range so a
  // constant line renders instead of vanishing.
  if (span === 0) return () => (rangeMin + rangeMax) / 2;
  return (value: number) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

/** Rounds a maximum up to a readable axis top, so ticks land on whole numbers. */
export function niceMax(value: number): number {
  if (value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
}

/** Evenly spaced tick values from 0 to max inclusive. */
export function ticks(max: number, count = 4): number[] {
  return Array.from({ length: count + 1 }, (_, i) => (max / count) * i);
}

/** Builds an SVG path through points, as a polyline. */
export function linePath(points: Array<[number, number]>): string {
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
}

/** Builds a closed SVG path for the band between an upper and lower edge. */
export function areaPath(upper: Array<[number, number]>, lower: Array<[number, number]>): string {
  if (upper.length === 0) return "";
  const back = [...lower].reverse();
  return `${linePath(upper)} L${back[0][0].toFixed(2)},${back[0][1].toFixed(2)} ${linePath(back).slice(1)} Z`;
}

/** Formats a bucket timestamp for an axis tick, at a granularity suited to the range. */
export function formatTick(iso: string, spanHours: number): string {
  const date = new Date(iso);
  if (spanHours > 48) {
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/** Formats a bucket timestamp for a tooltip, where the full moment matters. */
export function formatMoment(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Formats the span a bucket covers, so a tooltip names the window rather than just the instant it opened. */
export function formatBucketRange(iso: string, bucketHours: number): string {
  const start = new Date(iso);
  // A daily bucket is identified by its date; printing both ends just repeats it.
  if (bucketHours >= 24) {
    return start.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  }
  const end = new Date(start.getTime() + bucketHours * 3600_000);
  const time = (date: Date) => date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  return `${start.toLocaleDateString(undefined, { month: "short", day: "numeric" })}, ${time(start)} – ${time(end)}`;
}
