// Per-zone view: how many people are in a zone now, which cameras are
// contributing that answer, and how it has behaved over time.
//
// The per-camera figures sit beside the headcount rather than behind it. A
// zone reading four people looks equally healthy whether three cameras agree
// or two have silently stopped reporting, and the only way to tell is to see
// what each one said.

import { useEffect, useState } from "react";
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
import { useZoneOccupancy } from "../hooks/useZoneOccupancy";

// A zone whose newest record is older than this is not reporting any more.
// Nodes ship in batches, so a few seconds of silence is normal.
const STALE_AFTER_MS = 20_000;

/** Renders per-zone occupancy, contributing cameras, and that zone's history. */
export function ZonesPage() {
  const { data, isPending, isError } = useZoneOccupancy();
  const [selected, setSelected] = useState<string | null>(null);

  // Select the first zone once one appears, but never override a choice the
  // reader has already made.
  useEffect(() => {
    if (selected === null && data && data.length > 0) setSelected(data[0].zone_id);
  }, [data, selected]);

  if (isPending) return <LoadingState label="Loading zones…" />;
  if (isError) return <ErrorState message="Couldn't load zones -- is the server running?" />;

  if (data.length === 0) {
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

  const zone = data.find((z) => z.zone_id === selected) ?? data[0];
  const cameras = Object.entries(zone.per_camera).sort(([a], [b]) => a.localeCompare(b));
  const summed = cameras.reduce((running, [, count]) => running + count, 0);
  const merged = summed - zone.total;
  const age = Date.now() - new Date(zone.timestamp).getTime();
  const live = age < STALE_AFTER_MS;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Zones"
        description="Occupancy and history for each marked floor area. Someone visible to several cameras is counted once."
        actions={
          data.length > 1 ? (
            <div className="flex flex-wrap gap-1.5">
              {data.map((option) => (
                <button
                  key={option.zone_id}
                  type="button"
                  onClick={() => setSelected(option.zone_id)}
                  aria-pressed={option.zone_id === zone.zone_id}
                  className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                    option.zone_id === zone.zone_id
                      ? "border-[var(--app-accent)] bg-[var(--app-accent-wash)] text-[var(--app-accent)]"
                      : "border-[var(--app-line-strong)] text-[var(--app-ink-secondary)]"
                  }`}
                >
                  {option.zone_id}
                </button>
              ))}
            </div>
          ) : undefined
        }
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[18rem_1fr]">
        <Card className="p-4">
          <div className="flex items-start justify-between">
            <StatTile
              label="In this zone"
              value={zone.total}
              unit={zone.total === 1 ? "person" : "people"}
              detail={
                merged > 0
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
                  <span className="truncate text-[var(--app-ink-secondary)]">{cameraNodeId}</span>
                  <span className="tabular-nums text-[var(--app-ink)]">{count}</span>
                </div>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Badge tone={zone.cameras_reporting > 1 ? "good" : "warning"}>
                {zone.cameras_reporting} reporting
              </Badge>
              {!live && <Badge tone="critical">stale</Badge>}
            </div>
            {zone.cameras_reporting === 1 && (
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
              <EmptyState title="No cameras reporting for this zone." />
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

      <section>
        <SectionHeading hint={`Filtered to ${zone.zone_id}.`}>History</SectionHeading>
        <HistoricalCharts zoneId={zone.zone_id} />
      </section>
    </div>
  );
}
