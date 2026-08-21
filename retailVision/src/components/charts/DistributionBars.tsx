// Magnitude comparison across a handful of named categories, as horizontal
// bars. Horizontal because category names are words -- age bands, emotions,
// genders -- and words read straight across rather than rotated under a
// vertical axis.
//
// Every bar is directly labelled with its value, so nothing depends on
// estimating a length against an axis, and a single hue is used throughout:
// the categories here are one measure split up, not separate series, so
// colouring them differently would imply a distinction that is not there.

export interface DistributionBarsProps {
  /** Category name to count. */
  distribution: Record<string, number>;
  /** Shown when every category is absent. */
  emptyLabel?: string;
  /**
   * "value" (default) renders largest first, right for unordered categories.
   * "given" keeps insertion order, for categories that are themselves a scale
   * -- duration buckets, hours -- where sorting by size would shuffle the axis.
   */
  order?: "value" | "given";
}

/** Renders a labelled horizontal bar per category. */
export function DistributionBars({
  distribution,
  emptyLabel = "No data in this range.",
  order = "value",
}: DistributionBarsProps) {
  const entries = Object.entries(distribution);
  if (order === "value") entries.sort(([, a], [, b]) => b - a);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);

  if (entries.length === 0 || total === 0) {
    return <div className="py-6 text-center text-sm text-[var(--app-ink-muted)]">{emptyLabel}</div>;
  }

  const max = Math.max(...entries.map(([, count]) => count));

  return (
    <div className="flex flex-col gap-2">
      {entries.map(([name, count]) => (
        <div key={name} className="grid grid-cols-[7rem_1fr_4.5rem] items-center gap-3">
          <span className="truncate text-xs text-[var(--app-ink-secondary)]" title={name}>
            {name}
          </span>
          <div className="h-4 overflow-hidden rounded-sm" style={{ background: "var(--viz-grid)" }}>
            <div
              className="h-full rounded-sm"
              style={{ width: `${Math.max((count / max) * 100, 1)}%`, background: "var(--viz-series-1)" }}
            />
          </div>
          <span className="text-right text-xs tabular-nums text-[var(--app-ink)]">
            {count}
            <span className="ml-1 text-[var(--app-ink-muted)]">{Math.round((count / total) * 100)}%</span>
          </span>
        </div>
      ))}
    </div>
  );
}
