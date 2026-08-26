// Magnitude comparison across a handful of named categories, as horizontal
// bars. Horizontal because category names are words -- age bands, emotions,
// genders -- and words read straight across rather than rotated under a
// vertical axis.
//
// Every bar is directly labelled with its value, so nothing depends on
// estimating a length against an axis, and a single hue is used throughout:
// the categories here are one measure split up, not separate series, so
// colouring them differently would imply a distinction that is not there.
//
// Built as a real <table> rather than styled divs. The content is tabular --
// a category, its count, its share -- so native semantics give assistive
// technology row/column navigation and a caption for free, where a grid of
// divs would announce an undifferentiated stream of words and numbers. The
// bar itself is aria-hidden: it encodes the same number as the cell beside
// it, and announcing it twice helps nobody.

export interface DistributionBarsProps {
  /** Category name to count. */
  distribution: Record<string, number>;
  /**
   * Names the table for assistive technology. Visually hidden, since every
   * call site already shows the same words in a card heading above it.
   */
  caption: string;
  /** Shown when every category is absent. */
  emptyLabel?: string;
  /**
   * "value" (default) renders largest first, right for unordered categories.
   * "given" keeps insertion order, for categories that are themselves a scale
   * -- duration buckets, hours -- where sorting by size would shuffle the axis.
   */
  order?: "value" | "given";
}

/** Renders a labelled horizontal bar per category, as a semantic table. */
export function DistributionBars({
  distribution,
  caption,
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
    <table className="w-full table-fixed border-collapse">
      <caption className="sr-only">{caption}</caption>
      <thead className="sr-only">
        <tr>
          <th scope="col">Category</th>
          <th scope="col">Relative size</th>
          <th scope="col">Count and share</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([name, count]) => (
          <tr key={name}>
            <th
              scope="row"
              className="w-28 truncate py-1 pr-3 text-left text-xs font-normal text-[var(--app-ink-secondary)]"
              title={name}
            >
              {name}
            </th>
            <td className="py-1" aria-hidden="true">
              <div className="h-4 overflow-hidden rounded-sm" style={{ background: "var(--viz-grid)" }}>
                <div
                  className="h-full rounded-sm"
                  style={{ width: `${Math.max((count / max) * 100, 1)}%`, background: "var(--viz-series-1)" }}
                />
              </div>
            </td>
            <td className="w-[4.5rem] py-1 pl-3 text-right text-xs tabular-nums text-[var(--app-ink)]">
              {count}
              <span className="ml-1 text-[var(--app-ink-muted)]">{Math.round((count / total) * 100)}%</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
