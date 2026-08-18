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
  /** Category name to count. Rendered largest first. */
  distribution: Record<string, number>;
  /** Shown when every category is absent. */
  emptyLabel?: string;
}

/** Renders a labelled horizontal bar per category, largest first. */
export function DistributionBars({ distribution, emptyLabel = "No data in this range." }: DistributionBarsProps) {
  const entries = Object.entries(distribution).sort(([, a], [, b]) => b - a);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);

  if (entries.length === 0 || total === 0) {
    return <div className="py-6 text-center text-sm text-slate-400">{emptyLabel}</div>;
  }

  const max = Math.max(...entries.map(([, count]) => count));

  return (
    <div className="viz-root flex flex-col gap-2">
      {entries.map(([name, count]) => (
        <div key={name} className="grid grid-cols-[7rem_1fr_4.5rem] items-center gap-3">
          <span className="truncate text-xs text-slate-600 dark:text-slate-300" title={name}>
            {name}
          </span>
          <div className="h-4 overflow-hidden rounded-sm" style={{ background: "var(--viz-grid)" }}>
            <div
              className="h-full rounded-sm"
              style={{ width: `${Math.max((count / max) * 100, 1)}%`, background: "var(--viz-series-1)" }}
            />
          </div>
          <span className="text-right text-xs tabular-nums text-slate-700 dark:text-slate-200">
            {count}
            <span className="ml-1 text-slate-400">{Math.round((count / total) * 100)}%</span>
          </span>
        </div>
      ))}
    </div>
  );
}
