// Per-zone view: how many people are in a zone now, which cameras are
// contributing that answer, and how it has behaved over time.
//
// The per-camera figures sit beside the headcount rather than behind it. A
// zone reading four people looks equally healthy whether three cameras agree
// or two have silently stopped reporting, and the only way to tell is to see
// what each one said.

import { useEffect, useMemo, useState } from "react";
import { HistoricalCharts } from "../components/charts/HistoricalCharts";
import { ErrorState } from "../components/common/ErrorState";
import { LoadingState } from "../components/common/LoadingState";
import {
  Badge,
  Card,
  CardHeader,
  EmptyState,
  LiveDot,
  PageHeader,
  SectionHeading,
  StatTile,
} from "../components/common/ui";
import { ArrangeableSections } from "../components/common/ArrangeableSections";
import { ZoneHeatmap } from "../components/occupancy/ZoneHeatmap";
import { useZoneGeometry } from "../hooks/useZoneGeometry";
import { useZoneOccupancy } from "../hooks/useZoneOccupancy";

// A zone whose newest record is older than this is not reporting any more.
// Nodes ship in batches, so a few seconds of silence is normal.
const STALE_AFTER_MS = 20_000;

/** Renders per-zone occupancy, contributing cameras, and that zone's history. */
export function ZonesPage() {
  const { data: occupancy, isPending, isError } = useZoneOccupancy();
  const { data: geometry, isPending: geometryPending } = useZoneGeometry();
  const [selected, setSelected] = useState<string | null>(null);

  // Which zones exist comes from the surveyed geometry the server keeps, not
  // from live occupancy: that endpoint reports only the last few seconds, so
  // it is empty whenever nobody happens to be walking through -- the normal
  // state most of the time. Driving the page off it meant an empty room
  // erased the zone selector, the floor map and the whole history with it.
  // Live occupancy still supplies the counts; it just no longer decides
  // whether there is anything to show.
  const zoneIds = useMemo(() => {
    const ids = new Set<string>();
    for (const shape of geometry ?? []) ids.add(shape.zone_id);
    for (const zone of occupancy ?? []) ids.add(zone.zone_id);
    return [...ids].sort();
  }, [geometry, occupancy]);

  // Select the first zone once one appears, but never override a choice the
  // reader has already made.
  useEffect(() => {
    if (selected === null && zoneIds.length > 0) setSelected(zoneIds[0]);
  }, [zoneIds, selected]);

  if (isPending || geometryPending) return <LoadingState label="Loading zones…" />;
  if (isError) return <ErrorState message="Couldn't load zones -- is the server running?" />;

  if (zoneIds.length === 0) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader title="Zones" description="Occupancy and history for each marked floor area." />
        <EmptyState
          title="No zones reporting yet."
          hint="A camera node contributes to a zone when it runs with --zones and a surveyed --marker-map, and can see a mapped marker."
        />
      </div>
    );
  }

  const zoneId = selected !== null && zoneIds.includes(selected) ? selected : zoneIds[0];
  // Null when the zone exists but nobody has been seen in it recently, which
  // is a quiet room rather than a missing zone -- the difference the page
  // previously could not express.
  const reported = occupancy?.find((z) => z.zone_id === zoneId) ?? null;
  const cameras = reported ? Object.entries(reported.per_camera).sort(([a], [b]) => a.localeCompare(b)) : [];
  const total = reported?.total ?? 0;
  const camerasReporting = reported?.cameras_reporting ?? 0;
  const summed = cameras.reduce((running, [, count]) => running + count, 0);
  const merged = summed - total;
  const live = reported !== null && Date.now() - new Date(reported.timestamp).getTime() < STALE_AFTER_MS;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Zones"
        description="Occupancy and history for each marked floor area. Someone visible to several cameras is counted once."
        actions={
          zoneIds.length > 1 ? (
            <div className="flex flex-wrap gap-1.5">
              {zoneIds.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setSelected(option)}
                  aria-pressed={option === zoneId}
                  className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                    option === zoneId
                      ? "border-[var(--app-accent)] bg-[var(--app-accent-wash)] text-[var(--app-accent)]"
                      : "border-[var(--app-line-strong)] text-[var(--app-ink-secondary)]"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          ) : undefined
        }
      />

      <ArrangeableSections
        page="zones"
        sections={[
          {
            id: "status",
            children: (
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[18rem_1fr]">
        <Card className="p-4">
          <div className="flex items-start justify-between">
            <StatTile
              label="In this zone"
              value={total}
              unit={total === 1 ? "person" : "people"}
              detail={
                reported === null
                  ? "nobody in view in the last few seconds"
                  : merged > 0
                    ? `${merged} duplicate sighting${merged === 1 ? "" : "s"} merged`
                    : "no overlap between cameras right now"
              }
            />
            <LiveDot live={live} />
          </div>

          <div className="mt-5 border-t border-[var(--app-line)] pt-3">
            <SectionHeading>Cameras</SectionHeading>
            <div className="flex flex-col gap-1.5">
              {cameras.map(([cameraNodeId, count]) => (
                <div key={cameraNodeId} className="flex items-baseline justify-between gap-2 text-sm">
                  <span className="min-w-0 truncate text-[var(--app-ink-secondary)]">{cameraNodeId}</span>
                  <span className="tabular-nums text-[var(--app-ink)]">{count}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Badge tone={camerasReporting > 1 ? "good" : "warning"}>
                {camerasReporting} reporting
              </Badge>
              {reported !== null && !live && <Badge tone="critical">stale</Badge>}
            </div>
            {camerasReporting === 1 && (
              <p className="mt-2 text-xs text-[var(--app-ink-muted)]">
                Only one camera is contributing, so anyone it cannot see is not counted.
              </p>
            )}
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Coverage"
            description="How each camera's count compares. Large gaps usually mean one is pointed away from where people are."
          />
          <div className="p-4">
            {cameras.length === 0 ? (
              <EmptyState
                title="Nobody in this zone right now."
                hint="Counts appear as soon as a camera sees someone. History and the floor map below still cover earlier activity."
              />
            ) : (
              <div className="flex flex-col gap-2">
                {cameras.map(([cameraNodeId, count]) => {
                  const max = Math.max(...cameras.map(([, c]) => c), 1);
                  return (
                    <div key={cameraNodeId} className="grid grid-cols-[9rem_1fr_2.5rem] items-center gap-3">
                      <span className="truncate text-xs text-[var(--app-ink-secondary)]">{cameraNodeId}</span>
                      <div className="h-4 overflow-hidden rounded-sm bg-[var(--app-line)]">
                        <div
                          className="h-full rounded-sm"
                          style={{
                            width: `${Math.max((count / max) * 100, 2)}%`,
                            background: "var(--app-accent)",
                          }}
                        />
                      </div>
                      <span className="text-right text-xs tabular-nums text-[var(--app-ink)]">{count}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </Card>
      </div>
            ),
          },
          {
            id: "floor map",
            children: (
              <>
                <SectionHeading hint="Where people accumulate, and who is there right now. Positions carry the head-height assumption.">
                  Floor map
                </SectionHeading>
                <Card>
                  <div className="p-4">
                    <ZoneHeatmap zoneId={zoneId} />
                  </div>
                </Card>
              </>
            ),
          },
          {
            id: "history",
            children: (
              <>
                <SectionHeading hint={`Filtered to ${zoneId}.`}>History</SectionHeading>
                <HistoricalCharts zoneId={zoneId} />
              </>
            ),
          },
        ]}
      />
    </div>
  );
}
