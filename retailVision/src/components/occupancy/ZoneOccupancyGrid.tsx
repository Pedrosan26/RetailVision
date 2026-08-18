// Per-zone headcount, with the contributing cameras shown underneath.
//
// The headline number is deduplicated: several cameras watching one zone all
// see each other's subjects, so adding their counts would report the same
// person several times. The per-camera figures are kept visible rather than
// hidden, because they are how you notice that one camera has stopped
// contributing or is seeing far more than its neighbours -- a total alone
// looks equally healthy either way.

import { useZoneOccupancy } from "../../hooks/useZoneOccupancy";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";

/** Shows one card per zone: the deduplicated headcount, and each camera's own count. */
export function ZoneOccupancyGrid() {
  const { data, isPending, isError } = useZoneOccupancy();

  if (isPending) return <LoadingState label="Loading zone occupancy…" />;
  if (isError) return <ErrorState message="Couldn't load zone occupancy -- is the server running?" />;
  if (data.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-[var(--app-line-strong)] py-8 text-center text-sm text-[var(--app-ink-muted)]">
        No zones reporting yet. Camera nodes need <code className="font-mono text-xs">--zones</code> to report positions.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {data.map((zone) => {
        const cameras = Object.entries(zone.per_camera).sort(([a], [b]) => a.localeCompare(b));
        const summed = cameras.reduce((running, [, count]) => running + count, 0);
        const merged = summed - zone.total;

        return (
          <div
            key={zone.zone_id}
            className="rounded-lg border border-[var(--app-line)] bg-[var(--app-surface)] p-4"
          >
            <div className="text-xs font-medium uppercase tracking-wide text-[var(--app-ink-muted)]">
              {zone.zone_id}
            </div>

            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-4xl font-semibold tabular-nums text-[var(--app-ink)]">
                {zone.total}
              </span>
              <span className="text-sm text-[var(--app-ink-muted)]">
                {zone.total === 1 ? "person" : "people"}
              </span>
            </div>

            <div className="mt-3 space-y-1 border-t border-[var(--app-line)] pt-3">
              {cameras.map(([cameraNodeId, count]) => (
                <div key={cameraNodeId} className="flex items-baseline justify-between text-sm">
                  <span className="truncate text-[var(--app-ink-secondary)]">{cameraNodeId}</span>
                  <span className="tabular-nums text-[var(--app-ink)]">{count}</span>
                </div>
              ))}
            </div>

            <div className="mt-3 text-xs text-[var(--app-ink-muted)]">
              {zone.cameras_reporting} camera{zone.cameras_reporting === 1 ? "" : "s"} reporting
              {merged > 0 && <> · {merged} duplicate sighting{merged === 1 ? "" : "s"} merged</>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
