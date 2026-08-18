// Composition over time: how many detections fell into each category in each
// time bucket, stacked so the outline is also the total.
//
// One series renders as a plain area with no legend -- the title already names
// it. Two or more get a legend carrying the current values, which is also what
// discharges the palette's contrast relief rule: identity never rests on colour
// alone.

import { useState } from "react";
import {
  type Scale,
  areaPath,
  formatMoment,
  formatTick,
  linearScale,
  linePath,
  niceMax,
  seriesColor,
  ticks,
} from "./chartScales";

const HEIGHT = 220;
const PADDING = { top: 12, right: 12, bottom: 28, left: 40 };

export interface StackedAreaChartProps {
  /** Bucket start timestamps, oldest first. */
  labels: string[];
  /** One entry per series: its name and a value per bucket. */
  series: Array<{ name: string; values: number[] }>;
  /** Hours spanned, which decides whether ticks read as times or dates. */
  spanHours: number;
  /** Word for one unit, used in the tooltip. */
  unitLabel?: string;
}

/** Renders stacked areas over time with a hover crosshair and per-bucket tooltip. */
export function StackedAreaChart({ labels, series, spanHours, unitLabel = "detections" }: StackedAreaChartProps) {
  const [hover, setHover] = useState<number | null>(null);

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

  return (
    <div>
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

        {bands.map((band) => (
          <g key={band.name}>
            {/* A hairline of surface between stacked fills keeps the boundary
                legible when two adjacent hues are close in lightness. */}
            <path d={areaPath(band.upper, band.lower)} fill={band.colour} opacity={0.85} />
            <path d={linePath(band.upper)} fill="none" stroke="var(--viz-surface)" strokeWidth={2} />
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
          <line
            x1={x(hover)}
            x2={x(hover)}
            y1={PADDING.top}
            y2={PADDING.top + plotHeight}
            stroke="var(--viz-ink-muted)"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}

        {/* Invisible hit bands, wider than any mark, so hovering is forgiving. */}
        {labels.map((label, i) => (
          <rect
            key={`hit-${label}`}
            x={x(i) - plotWidth / Math.max(labels.length, 1) / 2}
            y={PADDING.top}
            width={plotWidth / Math.max(labels.length, 1)}
            height={plotHeight}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
          />
        ))}
      </svg>

      <div className="mt-2 min-h-[3.5rem] text-xs">
        {hover !== null ? (
          <div>
            <div className="font-medium text-[var(--app-ink)]">{formatMoment(labels[hover])}</div>
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
              {bands.map((band, seriesIndex) => (
                <span key={band.name} className="inline-flex items-center gap-1.5 text-[var(--app-ink-secondary)]">
                  <span
                    aria-hidden
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ background: band.colour }}
                  />
                  {band.name}
                  <span className="tabular-nums font-medium text-[var(--app-ink)]">
                    {series[seriesIndex].values[hover] ?? 0}
                  </span>
                </span>
              ))}
              <span className="text-[var(--app-ink-muted)]">
                total <span className="tabular-nums font-medium">{totals[hover]}</span> {unitLabel}
              </span>
            </div>
          </div>
        ) : series.length > 1 ? (
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {bands.map((band, seriesIndex) => {
              const total = series[seriesIndex].values.reduce((sum, value) => sum + value, 0);
              return (
                <span key={band.name} className="inline-flex items-center gap-1.5 text-[var(--app-ink-secondary)]">
                  <span
                    aria-hidden
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ background: band.colour }}
                  />
                  {band.name}
                  <span className="tabular-nums font-medium text-[var(--app-ink)]">{total}</span>
                </span>
              );
            })}
          </div>
        ) : (
          <div className="text-[var(--app-ink-muted)]">Hover for values.</div>
        )}
      </div>
    </div>
  );
}
