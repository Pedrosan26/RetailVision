// Per-zone headcount, with the contributing cameras shown underneath.
//
// The headline number is deduplicated: several cameras watching one zone all
// see each other's subjects, so adding their counts would report the same
// person several times. The per-camera figures are kept visible rather than
// hidden, because they are how you notice that one camera has stopped
// contributing or is seeing far more than its neighbours -- a total alone
// looks equally healthy either way.

import { useMemo } from "react";
import { useZoneGeometry } from "../../hooks/useZoneGeometry";
import { useZoneOccupancy } from "../../hooks/useZoneOccupancy";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";

/** Shows one card per zone: the deduplicated headcount, and each camera's own count. */
export function ZoneOccupancyGrid() {
  const { data: occupancy, isPending, isError } = useZoneOccupancy();
  const { data: geometry, isPending: geometryPending } = useZoneGeometry();

  // A zone exists because it was surveyed, not because someone is standing in
  // it. Occupancy covers only the last few seconds, so keying the cards off it
  // made every zone vanish whenever the room went quiet -- and the message it
  // left behind blamed a missing --zones flag for what was just an empty room.
  const cards = useMemo(() => {
    const ids = new Set<string>();
    for (const shape of geometry ?? []) ids.add(shape.zone_id);
    for (const zone of occupancy ?? []) ids.add(zone.zone_id);
    return [...ids].sort().map((zoneId) => ({
      zoneId,
      reported: occupancy?.find((z) => z.zone_id === zoneId) ?? null,
    }));
  }, [geometry, occupancy]);

  if (isPending || geometryPending) return <LoadingState label="Loading zone occupancy…" />;
  if (isError) return <ErrorState message="Couldn't load zone occupancy -- is the server running?" />;
  if (cards.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-[var(--app-line-strong)] py-8 text-center text-sm text-[var(--app-ink-muted)]">
        No zones surveyed yet. Camera nodes need <code className="font-mono text-xs">--zones</code> and a surveyed{" "}
        <code className="font-mono text-xs">--marker-map</code> to report positions.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {cards.map(({ zoneId, reported }) => {
        const cameras = reported ? Object.entries(reported.per_camera).sort(([a], [b]) => a.localeCompare(b)) : [];
        const total = reported?.total ?? 0;
        const camerasReporting = reported?.cameras_reporting ?? 0;
        const summed = cameras.reduce((running, [, count]) => running + count, 0);
        const merged = summed - total;

        return (
          <div
            key={zoneId}
            className="rounded-lg border border-[var(--app-line)] bg-[var(--app-surface)] p-4"
          >
            <div className="text-xs font-medium uppercase tracking-wide text-[var(--app-ink-muted)]">
              {zoneId}
            </div>

            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-4xl font-semibold tabular-nums text-[var(--app-ink)]">
                {total}
              </span>
              <span className="text-sm text-[var(--app-ink-muted)]">
                {total === 1 ? "person" : "people"}
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
              {reported === null ? (
                "nobody in view right now"
              ) : (
                <>
                  {camerasReporting} camera{camerasReporting === 1 ? "" : "s"} reporting
                  {merged > 0 && <> · {merged} duplicate sighting{merged === 1 ? "" : "s"} merged</>}
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
