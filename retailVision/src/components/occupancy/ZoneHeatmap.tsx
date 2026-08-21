// Top-down floor map of one zone: its surveyed polygon, where people have
// accumulated over the last hour, and who is standing there right now.
//
// The heat layer is a grid of cells whose opacity follows sighting density
// -- a sequential encoding in the single accent hue, since magnitude on a
// map is one measure, not categories. Live people are dots on top, drawn
// in ink with a surface ring so they read against any heat level beneath.
//
// The map is explorable: wheel (or the buttons) zooms, dragging pans while
// zoomed, and hovering a cell or a person opens a floating detail box next
// to the cursor -- the same treatment the time charts use, and for the
// same reason: the reader's eye is already on the spot they are asking
// about.
//
// Positions inherit the pipeline's head-height assumption, so the map is a
// picture of where the system believes people are -- the honest framing for
// judging whether the zone geometry and the positions agree with the room.

import { useEffect, useMemo, useRef, useState } from "react";
import { useRecentDetections } from "../../hooks/useRecentDetections";
import { useZoneGeometry } from "../../hooks/useZoneGeometry";
import { EmptyState } from "../common/ui";

const CELL_METERS = 0.25;
const HEAT_WINDOW_HOURS = 1;
const LIVE_WINDOW_MS = 10_000;
const FETCH_LIMIT = 500;
const PADDING_METERS = 0.6;
const SVG_WIDTH = 720;
const MAX_ZOOM = 8;
const WHEEL_STEP = 1.15;
const TOOLTIP_WIDTH = 200;

interface CellStats {
  count: number;
  lastTs: number;
  emotions: Record<string, number>;
}

interface View {
  x: number;
  y: number;
  w: number;
  h: number;
}

type HoverDetail =
  | { kind: "cell"; worldX: number; worldY: number; stats: CellStats }
  | { kind: "dot"; agoSeconds: number; emotion: string; age: string; gender: string };

type Hover = HoverDetail & { x: number; y: number };

/** Formats how long ago a timestamp was, compactly. */
function formatAgo(ms: number): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s ago`;
}

/** The most frequent emotion in a distribution. */
function topEmotion(emotions: Record<string, number>): string {
  return Object.entries(emotions).reduce((best, entry) => (entry[1] > best[1] ? entry : best))[0];
}

/** Renders one zone's floor polygon with an hour of position heat, live person dots, zoom/pan, and hover detail. */
export function ZoneHeatmap({ zoneId }: { zoneId: string }) {
  const { data: geometry } = useZoneGeometry();
  const since = useMemo(() => new Date(Date.now() - HEAT_WINDOW_HOURS * 3600_000).toISOString(), []);
  const { data: detections } = useRecentDetections({ zone_id: zoneId, since, limit: FETCH_LIMIT });

  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<{ pointerX: number; pointerY: number; view: View } | null>(null);
  // null view = fit-to-zone; a View = zoomed/panned viewBox in base svg units.
  const [view, setView] = useState<View | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  const zone = geometry?.find((g) => g.zone_id === zoneId);

  // World box around the polygon, padded so edge cells and dots are not cut.
  const frame = useMemo(() => {
    if (!zone) return null;
    const xs = zone.polygon.map(([x]) => x);
    const ys = zone.polygon.map(([, y]) => y);
    const minX = Math.min(...xs) - PADDING_METERS;
    const maxX = Math.max(...xs) + PADDING_METERS;
    const minY = Math.min(...ys) - PADDING_METERS;
    const maxY = Math.max(...ys) + PADDING_METERS;
    const scale = SVG_WIDTH / (maxX - minX);
    return { minX, minY, maxY, scale, height: (maxY - minY) * scale };
  }, [zone]);

  // Wheel zoom must preventDefault to stop the page scrolling, which React's
  // synthetic wheel handler cannot (browsers register it passive), so the
  // listener goes on the element directly.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !frame) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = svg.getBoundingClientRect();
      setView((current) => {
        const base = current ?? { x: 0, y: 0, w: SVG_WIDTH, h: frame.height };
        const factor = event.deltaY > 0 ? WHEEL_STEP : 1 / WHEEL_STEP;
        const width = Math.min(SVG_WIDTH, Math.max(SVG_WIDTH / MAX_ZOOM, base.w * factor));
        if (width === base.w) return current;
        // Keep the world point under the cursor fixed while the scale changes.
        const pointX = base.x + ((event.clientX - rect.left) / rect.width) * base.w;
        const pointY = base.y + ((event.clientY - rect.top) / rect.height) * base.h;
        const ratio = width / base.w;
        const next = {
          x: pointX - (pointX - base.x) * ratio,
          y: pointY - (pointY - base.y) * ratio,
          w: width,
          h: (width / SVG_WIDTH) * frame.height,
        };
        return width >= SVG_WIDTH ? null : next;
      });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [frame]);

  if (!zone || !frame) {
    return (
      <EmptyState
        title="No floor shape for this zone yet."
        hint="A camera node uploads its zone geometry once at startup -- restart the nodes after upgrading them."
      />
    );
  }

  const { minX, minY, maxY, scale, height } = frame;
  const toSvg = ([x, y]: [number, number]): [number, number] => [(x - minX) * scale, (maxY - y) * scale];

  const positioned = (detections ?? []).filter((d) => d.world_x != null && d.world_y != null);
  const now = Date.now();
  const live = positioned.filter((d) => now - new Date(d.timestamp).getTime() < LIVE_WINDOW_MS);
  const totalSightings = positioned.length;

  const cells = new Map<string, CellStats>();
  for (const detection of positioned) {
    const column = Math.floor((detection.world_x! - minX) / CELL_METERS);
    const row = Math.floor((detection.world_y! - minY) / CELL_METERS);
    const key = `${column}:${row}`;
    const stats = cells.get(key) ?? { count: 0, lastTs: 0, emotions: {} };
    stats.count += 1;
    stats.lastTs = Math.max(stats.lastTs, new Date(detection.timestamp).getTime());
    stats.emotions[detection.emotion] = (stats.emotions[detection.emotion] ?? 0) + 1;
    cells.set(key, stats);
  }
  const maxCount = Math.max(...[...cells.values()].map((c) => c.count), 1);

  const ring = zone.polygon.map(toSvg).map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const cellSide = CELL_METERS * scale;
  const viewBox = view ?? { x: 0, y: 0, w: SVG_WIDTH, h: height };
  const zoomLevel = SVG_WIDTH / viewBox.w;

  /** Records what is hovered and where the cursor sits inside the container. */
  function track(event: React.MouseEvent, entry: HoverDetail) {
    if (dragRef.current) return;
    const box = containerRef.current?.getBoundingClientRect();
    if (!box) return;
    setHover({ ...entry, x: event.clientX - box.left, y: event.clientY - box.top });
  }

  /** Applies a zoom factor around the current view's centre, for the buttons. */
  function zoomBy(factor: number) {
    const base = view ?? { x: 0, y: 0, w: SVG_WIDTH, h: height };
    const width = Math.min(SVG_WIDTH, Math.max(SVG_WIDTH / MAX_ZOOM, base.w * factor));
    if (width >= SVG_WIDTH) {
      setView(null);
      return;
    }
    const ratio = width / base.w;
    const centerX = base.x + base.w / 2;
    const centerY = base.y + base.h / 2;
    setView({
      x: centerX - (base.w * ratio) / 2,
      y: centerY - (base.h * ratio) / 2,
      w: width,
      h: (width / SVG_WIDTH) * height,
    });
  }

  function onPointerDown(event: React.PointerEvent<SVGSVGElement>) {
    if (!view) return; // nothing to pan at fit
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerX: event.clientX, pointerY: event.clientY, view };
    setHover(null);
  }

  function onPointerMove(event: React.PointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const perPixel = drag.view.w / rect.width;
    setView({
      ...drag.view,
      x: drag.view.x - (event.clientX - drag.pointerX) * perPixel,
      y: drag.view.y - (event.clientY - drag.pointerY) * perPixel,
    });
  }

  function onPointerUp() {
    dragRef.current = null;
  }

  const flip = hover !== null && hover.x + TOOLTIP_WIDTH + 24 > (containerRef.current?.clientWidth ?? 0);

  return (
    <div>
      {/* Capped rather than full-column: a floor map wants to be glanceable
          next to everything else, and the zoom exists for close reading. */}
      <div ref={containerRef} className="relative mx-auto w-full max-w-[30rem]">
        <svg
          ref={svgRef}
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
          className="block h-auto max-h-[30rem] w-full touch-none"
          style={{ cursor: view ? "grab" : "default" }}
          role="img"
          aria-label={`Floor map of ${zoneId} with position heat and live people`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onMouseLeave={() => setHover(null)}
        >
          <polygon points={ring} fill="var(--app-accent-wash)" fillOpacity={0.5} stroke="var(--viz-axis)" strokeWidth={1.5} />

          {[...cells.entries()].map(([key, stats]) => {
            const [column, row] = key.split(":").map(Number);
            const worldX = minX + (column + 0.5) * CELL_METERS;
            const worldY = minY + (row + 0.5) * CELL_METERS;
            const [x, y] = toSvg([minX + column * CELL_METERS, minY + (row + 1) * CELL_METERS]);
            return (
              <rect
                key={key}
                x={x}
                y={y}
                width={cellSide}
                height={cellSide}
                fill="var(--app-accent)"
                // Sequential ramp: opacity tracks density, floor keeps a lone
                // sighting visible, ceiling keeps dots legible on the peak.
                opacity={0.12 + 0.55 * (stats.count / maxCount)}
                onMouseEnter={(e) => track(e, { kind: "cell", worldX, worldY, stats })}
                onMouseMove={(e) => track(e, { kind: "cell", worldX, worldY, stats })}
              />
            );
          })}

          {live.map((detection) => {
            const [x, y] = toSvg([detection.world_x!, detection.world_y!]);
            const agoSeconds = Math.round((now - new Date(detection.timestamp).getTime()) / 1000);
            const detail = {
              kind: "dot" as const,
              agoSeconds,
              emotion: detection.emotion,
              age: detection.age_group,
              gender: detection.gender,
            };
            return (
              <circle
                key={detection.id}
                cx={x}
                cy={y}
                r={6 / Math.sqrt(zoomLevel)}
                fill="var(--app-ink)"
                stroke="var(--app-surface)"
                strokeWidth={2.5 / Math.sqrt(zoomLevel)}
                onMouseEnter={(e) => track(e, detail)}
                onMouseMove={(e) => track(e, detail)}
              />
            );
          })}

          {/* Scale bar: always 2 real meters long, so zooming visibly stretches
              it -- that is the point of a scale bar. Offsets and type scale
              with the view so the bar hugs the corner at any zoom. */}
          {(() => {
            const unit = viewBox.w / SVG_WIDTH;
            const x0 = viewBox.x + 12 * unit;
            const y0 = viewBox.y + viewBox.h - 14 * unit;
            return (
              <g>
                <line x1={x0} y1={y0} x2={x0 + 2 * scale} y2={y0} stroke="var(--viz-ink-muted)" strokeWidth={1.5 * unit} />
                <line x1={x0} y1={y0 - 5 * unit} x2={x0} y2={y0 + 5 * unit} stroke="var(--viz-ink-muted)" strokeWidth={1.5 * unit} />
                <line x1={x0 + 2 * scale} y1={y0 - 5 * unit} x2={x0 + 2 * scale} y2={y0 + 5 * unit} stroke="var(--viz-ink-muted)" strokeWidth={1.5 * unit} />
                <text x={x0 + scale} y={y0 - 8 * unit} textAnchor="middle" fontSize={10 * unit} fill="var(--viz-ink-muted)">
                  2 m
                </text>
              </g>
            );
          })()}
        </svg>

        {/* Zoom controls: an overlay, not chart content. */}
        <div className="absolute right-2 top-2 flex flex-col overflow-hidden rounded-md border border-[var(--app-line-strong)] bg-[var(--app-raised)]">
          <button type="button" aria-label="Zoom in" onClick={() => zoomBy(1 / 1.5)} className="px-2 py-1 text-sm text-[var(--app-ink-secondary)] hover:bg-[var(--app-accent-wash)]">
            +
          </button>
          <button type="button" aria-label="Zoom out" onClick={() => zoomBy(1.5)} className="border-t border-[var(--app-line)] px-2 py-1 text-sm text-[var(--app-ink-secondary)] hover:bg-[var(--app-accent-wash)]">
            −
          </button>
          <button type="button" aria-label="Reset zoom" onClick={() => setView(null)} disabled={!view} className="border-t border-[var(--app-line)] px-2 py-1 text-xs text-[var(--app-ink-muted)] hover:bg-[var(--app-accent-wash)] disabled:opacity-40">
            fit
          </button>
        </div>

        {hover !== null && (
          <div
            className="pointer-events-none absolute z-10 rounded-lg border border-[var(--app-line-strong)] bg-[var(--app-raised)] p-2.5 text-xs shadow-lg"
            style={{
              width: TOOLTIP_WIDTH,
              left: flip ? hover.x - TOOLTIP_WIDTH - 12 : hover.x + 12,
              top: Math.max(hover.y - 12, 0),
            }}
          >
            {hover.kind === "cell" ? (
              <>
                <div className="font-medium tabular-nums text-[var(--app-ink)]">
                  ({hover.worldX.toFixed(1)}, {hover.worldY.toFixed(1)}) m
                </div>
                <div className="mt-1 flex flex-col gap-0.5 text-[var(--app-ink-secondary)]">
                  <span>
                    {hover.stats.count} sighting{hover.stats.count === 1 ? "" : "s"}
                    <span className="text-[var(--app-ink-muted)]">
                      {" "}
                      · {Math.round((hover.stats.count / Math.max(totalSightings, 1)) * 100)}% of the hour
                    </span>
                  </span>
                  <span>mostly {topEmotion(hover.stats.emotions)}</span>
                  <span className="text-[var(--app-ink-muted)]">last seen {formatAgo(now - hover.stats.lastTs)}</span>
                </div>
              </>
            ) : (
              <>
                <div className="font-medium text-[var(--app-ink)]">Person here now</div>
                <div className="mt-1 flex flex-col gap-0.5 text-[var(--app-ink-secondary)]">
                  <span>
                    {hover.age} · {hover.gender} · {hover.emotion}
                  </span>
                  <span className="text-[var(--app-ink-muted)]">seen {hover.agoSeconds}s ago</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--app-ink-muted)]">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm bg-[var(--app-accent)] opacity-60" />
          sightings, last {HEAT_WINDOW_HOURS}h
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full border-2 border-[var(--app-surface)] bg-[var(--app-ink)]" />
          person right now
        </span>
        <span>{positioned.length} positioned sightings{positioned.length === FETCH_LIMIT ? " (capped)" : ""}</span>
        <span className="ml-auto">
          scroll to zoom{view ? " · drag to pan" : ""} · {zoomLevel.toFixed(1)}×
        </span>
      </div>
    </div>
  );
}
