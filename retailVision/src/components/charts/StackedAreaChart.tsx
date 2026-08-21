// Composition over time: how many detections fell into each category in each
// time bucket, stacked so the outline is also the total.
//
// Hovering reads the cursor's vertical position as well as its horizontal one,
// so pointing at a band asks about that series rather than about the bucket as
// a whole. The tooltip then leads with that one series -- its count, its share,
// and how it moved since the previous bucket -- and the other bands dim so the
// focused one's shape is legible across the whole span. Sweeping sideways along
// a band therefore traces that series' history, which is the question a stacked
// chart is otherwise bad at answering: an upper band's thickness is easy to
// read, its position is not, because it rides on everything beneath it.
//
// Pointing above the stack, where no band lies, falls back to summarizing the
// whole bucket.
//
// Hovering is momentary, so the legend entries are also buttons that pin one
// series: a reader following a single emotion should not have to keep the
// cursor inside a band that may be thin, or interrupted, to hold their place.
// A pinned series stays isolated while the cursor goes elsewhere -- including
// off the chart entirely -- and hovering another band still previews it, with
// the pin taking back over as soon as the cursor leaves. That makes it a way
// to look around without losing what was deliberately chosen.
//
// The legend is permanent rather than something the tooltip replaces.
// Identity never rests on colour alone, which needs the legend present whether
// or not anyone is hovering; making it the isolation control also gives that
// interaction to keyboard users, who have no hover to offer.

import { useRef, useState } from "react";
import {
  type Scale,
  areaPath,
  formatBucketRange,
  formatTick,
  linearScale,
  linePath,
  niceMax,
  seriesColor,
  ticks,
} from "./chartScales";

const HEIGHT = 220;
const PADDING = { top: 12, right: 12, bottom: 28, left: 40 };
// The floating box is a fixed width so it can be flipped to the other side of
// the cursor near the right edge without the content reflowing as it moves.
const TOOLTIP_WIDTH = 232;
const DIMMED_OPACITY = 0.22;
const BAND_OPACITY = 0.85;

interface Hover {
  index: number;
  x: number;
  y: number;
  /** Which band the cursor is inside, or null when it is above the stack. */
  band: number | null;
}

export interface StackedAreaChartProps {
  /** Bucket start timestamps, oldest first. */
  labels: string[];
  /** One entry per series: its name and a value per bucket. */
  series: Array<{ name: string; values: number[] }>;
  /** Hours spanned, which decides whether ticks read as times or dates. */
  spanHours: number;
  /** Hours one bucket covers, so the tooltip can name the window it summarizes. */
  bucketHours: number;
  /** Word for one unit, used in the tooltip. */
  unitLabel?: string;
  /** Extra rows to show in the tooltip for each bucket, indexed alongside `labels`. */
  details?: Array<Array<{ label: string; value: string }>>;
}

/** Renders stacked areas over time, with hover focused on whichever band the cursor is inside. */
export function StackedAreaChart({
  labels,
  series,
  spanHours,
  bucketHours,
  unitLabel = "detections",
  details,
}: StackedAreaChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  // Pinned by name rather than by position: the filters change which series
  // exist, and an index would quietly come to mean a different one. A name
  // that is no longer present simply stops matching, which is the right
  // outcome for a series that has been filtered away.
  const [pinnedName, setPinnedName] = useState<string | null>(null);

  if (labels.length === 0 || series.length === 0) {
    return (
      <div className="flex h-[220px] items-center justify-center text-sm text-[var(--app-ink-muted)]">
        No data in this range.
      </div>
    );
  }

  const width = 720;
  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const totals = labels.map((_, i) => series.reduce((sum, s) => sum + (s.values[i] ?? 0), 0));
  const yMax = niceMax(Math.max(...totals, 1));
  const x: Scale = linearScale(0, Math.max(labels.length - 1, 1), PADDING.left, PADDING.left + plotWidth);
  const y: Scale = linearScale(0, yMax, PADDING.top + plotHeight, PADDING.top);

  // Cumulative baselines, so each series sits on the one below it.
  let running = labels.map(() => 0);
  const bands = series.map((s, seriesIndex) => {
    const lower = running.map((value, i) => [x(i), y(value)] as [number, number]);
    running = running.map((value, i) => value + (s.values[i] ?? 0));
    const upper = running.map((value, i) => [x(i), y(value)] as [number, number]);
    return { name: s.name, colour: seriesColor(seriesIndex), upper, lower };
  });

  const tickValues = ticks(yMax);
  const tickEvery = Math.max(1, Math.ceil(labels.length / 6));
  // A single-series chart has nothing to disambiguate, so it keeps the plain
  // whole-bucket reading rather than dimming its only band.
  const focusable = series.length > 1;

  /** Reads the cursor's position as a bucket index plus the band it falls inside. */
  function track(event: React.MouseEvent<SVGRectElement>) {
    const svg = event.currentTarget.ownerSVGElement;
    const container = containerRef.current;
    if (!svg || !container) return;
    const box = svg.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;

    // The SVG scales to its container, so screen pixels have to come back to
    // viewBox units before they can be compared with the plotted geometry.
    const svgX = ((event.clientX - box.left) * width) / box.width;
    const svgY = ((event.clientY - box.top) * HEIGHT) / box.height;

    const span = Math.max(labels.length - 1, 1);
    const position = labels.length <= 1 ? 0 : ((svgX - PADDING.left) / plotWidth) * span;
    const index = Math.min(Math.max(Math.round(position), 0), labels.length - 1);

    let band: number | null = null;
    if (focusable) {
      const found = bands.findIndex(
        (b) => svgY >= b.upper[index][1] && svgY <= b.lower[index][1],
      );
      band = found === -1 ? null : found;
    }

    const containerBox = container.getBoundingClientRect();
    setHover({
      index,
      band,
      x: event.clientX - containerBox.left,
      y: event.clientY - containerBox.top,
    });
  }

  const containerWidth = containerRef.current?.clientWidth ?? 0;
  const flip = hover !== null && hover.x + TOOLTIP_WIDTH + 24 > containerWidth;

  const pinnedIndex = focusable && pinnedName !== null
    ? (series.findIndex((s) => s.name === pinnedName) === -1
        ? null
        : series.findIndex((s) => s.name === pinnedName))
    : null;
  // Pointing at a band wins while the cursor is over one, and the pin takes
  // back over the moment it leaves -- so hovering explores without losing the
  // series the reader deliberately chose.
  const focused = hover?.band ?? pinnedIndex;

  /** Describes how a series moved from the previous bucket to this one. */
  function delta(seriesIndex: number, index: number): string {
    if (index === 0) return "first bucket in range";
    const current = series[seriesIndex].values[index] ?? 0;
    const previous = series[seriesIndex].values[index - 1] ?? 0;
    const change = current - previous;
    if (change === 0) return "unchanged from previous";
    return `${change > 0 ? "+" : "−"}${Math.abs(change)} from previous`;
  }

  return (
    <div>
      <div ref={containerRef} className="relative">
        <svg
          viewBox={`0 0 ${width} ${HEIGHT}`}
          className="w-full"
          role="img"
          aria-label={`${series.map((s) => s.name).join(", ")} over time`}
          onMouseLeave={() => setHover(null)}
        >
          {tickValues.map((value) => (
            <g key={value}>
              <line
                x1={PADDING.left}
                x2={PADDING.left + plotWidth}
                y1={y(value)}
                y2={y(value)}
                stroke="var(--viz-grid)"
                strokeWidth={1}
              />
              <text
                x={PADDING.left - 8}
                y={y(value)}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={10}
                fill="var(--viz-ink-muted)"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {Math.round(value)}
              </text>
            </g>
          ))}

          {bands.map((band, seriesIndex) => (
            <g key={band.name}>
              {/* A hairline of surface between stacked fills keeps the boundary
                  legible when two adjacent hues are close in lightness. */}
              <path
                d={areaPath(band.upper, band.lower)}
                fill={band.colour}
                opacity={focused === null || focused === seriesIndex ? BAND_OPACITY : DIMMED_OPACITY}
              />
              <path d={linePath(band.upper)} fill="none" stroke="var(--viz-surface)" strokeWidth={2} />
              {/* Tracing the focused band's own edge makes its shape readable
                  even where it is thin. */}
              {focused === seriesIndex && (
                <path d={linePath(band.upper)} fill="none" stroke={band.colour} strokeWidth={2} />
              )}
            </g>
          ))}

          {labels.map((label, i) =>
            i % tickEvery === 0 ? (
              <text
                key={label}
                x={x(i)}
                y={HEIGHT - 8}
                textAnchor="middle"
                fontSize={10}
                fill="var(--viz-ink-muted)"
              >
                {formatTick(label, spanHours)}
              </text>
            ) : null,
          )}

          <line
            x1={PADDING.left}
            x2={PADDING.left + plotWidth}
            y1={y(0)}
            y2={y(0)}
            stroke="var(--viz-axis)"
            strokeWidth={1}
          />

          {hover !== null && (
            <>
              <line
                x1={x(hover.index)}
                x2={x(hover.index)}
                y1={PADDING.top}
                y2={PADDING.top + plotHeight}
                stroke="var(--viz-ink-muted)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              {bands.map((band, seriesIndex) =>
                focused === null || focused === seriesIndex ? (
                  <circle
                    key={`marker-${band.name}`}
                    cx={band.upper[hover.index][0]}
                    cy={band.upper[hover.index][1]}
                    r={focused === seriesIndex ? 4.5 : 3.5}
                    fill={band.colour}
                    stroke="var(--viz-surface)"
                    strokeWidth={2}
                  />
                ) : null,
              )}
            </>
          )}

          {/* One hit area over the whole plot: the band is resolved from the
              cursor's height, which per-bucket columns could not report. */}
          <rect
            x={PADDING.left}
            y={PADDING.top}
            width={plotWidth}
            height={plotHeight}
            fill="transparent"
            onMouseMove={track}
          />
        </svg>

        {hover !== null && (
          <div
            // Never a hover target itself, or the box would chase the cursor it
            // just stole focus from.
            className="pointer-events-none absolute z-10 rounded-lg border border-[var(--app-line-strong)] bg-[var(--app-raised)] p-3 shadow-lg"
            style={{
              width: TOOLTIP_WIDTH,
              left: flip ? hover.x - TOOLTIP_WIDTH - 12 : hover.x + 12,
              top: Math.max(hover.y - 16, 0),
            }}
          >
            <div className="text-xs font-medium text-[var(--app-ink)]">
              {formatBucketRange(labels[hover.index], bucketHours)}
            </div>

            {focused !== null ? (
              <>
                <div className="mt-1.5 flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: bands[focused].colour }}
                  />
                  <span className="truncate text-xs font-medium text-[var(--app-ink)]">
                    {bands[focused].name}
                  </span>
                </div>
                <div className="mt-0.5 flex items-baseline gap-1.5">
                  <span className="text-lg font-semibold tabular-nums text-[var(--app-ink)]">
                    {series[focused].values[hover.index] ?? 0}
                  </span>
                  {totals[hover.index] > 0 && (
                    <span className="text-xs text-[var(--app-ink-secondary)]">
                      {Math.round(((series[focused].values[hover.index] ?? 0) / totals[hover.index]) * 100)}% of{" "}
                      {totals[hover.index]}
                    </span>
                  )}
                </div>
                <div className="text-xs text-[var(--app-ink-muted)]">{delta(focused, hover.index)}</div>

                <div className="mt-2 flex flex-col gap-1 border-t border-[var(--app-line)] pt-2">
                  {bands.map((band, seriesIndex) =>
                    seriesIndex === focused ? null : (
                      <div key={band.name} className="flex items-baseline gap-1.5 text-xs">
                        <span
                          aria-hidden
                          className="inline-block h-2 w-2 shrink-0 translate-y-px rounded-full opacity-60"
                          style={{ background: band.colour }}
                        />
                        <span className="truncate text-[var(--app-ink-muted)]">{band.name}</span>
                        <span className="ml-auto shrink-0 tabular-nums text-[var(--app-ink-secondary)]">
                          {series[seriesIndex].values[hover.index] ?? 0}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </>
            ) : (
              <>
                <div className="mt-1.5 flex items-baseline gap-1.5 border-b border-[var(--app-line)] pb-2">
                  <span className="text-lg font-semibold tabular-nums text-[var(--app-ink)]">
                    {totals[hover.index]}
                  </span>
                  <span className="text-xs text-[var(--app-ink-muted)]">{unitLabel}</span>
                </div>
                <div className="mt-2 flex flex-col gap-1">
                  {bands.map((band, seriesIndex) => {
                    const value = series[seriesIndex].values[hover.index] ?? 0;
                    const total = totals[hover.index];
                    return (
                      <div key={band.name} className="flex items-baseline gap-1.5 text-xs">
                        <span
                          aria-hidden
                          className="inline-block h-2.5 w-2.5 shrink-0 translate-y-px rounded-full"
                          style={{ background: band.colour }}
                        />
                        <span className="truncate text-[var(--app-ink-secondary)]">{band.name}</span>
                        <span className="ml-auto shrink-0 tabular-nums font-medium text-[var(--app-ink)]">
                          {value}
                        </span>
                        {total > 0 && (
                          <span className="w-9 shrink-0 text-right tabular-nums text-[var(--app-ink-muted)]">
                            {Math.round((value / total) * 100)}%
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            )}

            {details?.[hover.index]?.length ? (
              <div className="mt-2 flex flex-col gap-1 border-t border-[var(--app-line)] pt-2">
                {details[hover.index].map((row) => (
                  <div key={row.label} className="flex items-baseline justify-between gap-2 text-xs">
                    <span className="text-[var(--app-ink-muted)]">{row.label}</span>
                    <span className="tabular-nums text-[var(--app-ink-secondary)]">{row.value}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        )}
      </div>

      {series.length > 1 && (
        <div className="mt-2 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs">
          {bands.map((band, seriesIndex) => {
            const total = series[seriesIndex].values.reduce((sum, value) => sum + value, 0);
            const isPinned = pinnedIndex === seriesIndex;
            return (
              <button
                key={band.name}
                type="button"
                // Clicking the pinned series again releases it, so the control
                // that turned isolation on is the one that turns it off.
                onClick={() => setPinnedName(isPinned ? null : band.name)}
                aria-pressed={isPinned}
                title={isPinned ? `Stop isolating ${band.name}` : `Isolate ${band.name}`}
                className={`inline-flex cursor-pointer items-center gap-1.5 rounded-md border px-2 py-1 transition-opacity ${
                  focused === null || focused === seriesIndex ? "opacity-100" : "opacity-45"
                } ${
                  isPinned
                    ? "border-[var(--app-line-strong)] bg-[var(--app-accent-wash)]"
                    : "border-transparent hover:border-[var(--app-line)]"
                } text-[var(--app-ink-secondary)]`}
              >
                <span
                  aria-hidden
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: band.colour }}
                />
                {band.name}
                <span className="tabular-nums font-medium text-[var(--app-ink)]">{total}</span>
              </button>
            );
          })}
          {pinnedIndex !== null && (
            <button
              type="button"
              onClick={() => setPinnedName(null)}
              className="ml-1 rounded-md px-2 py-1 text-[var(--app-accent)] underline underline-offset-2"
            >
              Show all
            </button>
          )}
        </div>
      )}
    </div>
  );
}
