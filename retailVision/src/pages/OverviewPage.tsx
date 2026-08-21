// Landing page: what is happening right now.
//
// Ordered by how often it is looked at rather than by how the system is
// built -- the headline figures first so the page answers "how is the
// space doing" before any scrolling, then the per-zone headcount, the
// cameras producing it, and the raw event stream. History lives on the
// Zones page, where it can be scoped to one area; repeating it here would
// make the live view slower to read.

import { CameraFeedGrid } from "../components/cameras/CameraFeedGrid";
import { AlertSettings } from "../components/common/AlertBanner";
import { ArrangeableSections } from "../components/common/ArrangeableSections";
import { RecentActivityFeed } from "../components/detections/RecentActivityFeed";
import { Card, CardHeader, LiveDot, PageHeader, SectionHeading } from "../components/common/ui";
import { KpiRow } from "../components/occupancy/KpiRow";
import { ZoneOccupancyGrid } from "../components/occupancy/ZoneOccupancyGrid";
import { useLiveOccupancy } from "../hooks/useLiveOccupancy";

// A camera whose newest record is older than this has stopped reporting.
const STALE_AFTER_MS = 20_000;

/** Compact per-camera health chips: which nodes are reporting and how recently. */
function CameraStatusRow() {
  const { data } = useLiveOccupancy();
  if (!data || data.length === 0) return null;

  const newestByNode = new Map<string, number>();
  for (const row of data) {
    const at = new Date(row.timestamp).getTime();
    newestByNode.set(row.camera_node_id, Math.max(newestByNode.get(row.camera_node_id) ?? 0, at));
  }

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {[...newestByNode.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([node, at]) => (
        <span key={node} className="inline-flex items-center gap-1.5 text-xs text-[var(--app-ink-secondary)]">
          {node}
          <LiveDot live={Date.now() - at < STALE_AFTER_MS} />
        </span>
      ))}
    </div>
  );
}

/** Renders the Overview page: headline figures, live occupancy, camera feeds, and recent activity. */
export function OverviewPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Overview"
        description="Live occupancy across every marked zone, and what each camera is currently seeing."
      />

      <KpiRow />

      <ArrangeableSections
        page="overview"
        sections={[
          {
            id: "zones",
            children: (
              <>
                <SectionHeading hint="Someone visible to several cameras is counted once.">
                  People per zone
                </SectionHeading>
                <ZoneOccupancyGrid />
              </>
            ),
          },
          {
            id: "cameras",
            children: (
              <>
                <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2 pr-16">
                  <SectionHeading hint="Frames leave a camera node only while it runs with --stream-frames.">
                    Camera feeds
                  </SectionHeading>
                  <CameraStatusRow />
                </div>
                <CameraFeedGrid />
              </>
            ),
          },
          {
            id: "activity",
            children: (
              <>
                <SectionHeading>Recent activity</SectionHeading>
                <Card>
                  <CardHeader title="Latest detections" description="Newest first, across all cameras." />
                  <div className="px-4">
                    <RecentActivityFeed />
                  </div>
                </Card>
              </>
            ),
          },
          {
            id: "alerts",
            children: (
              <>
                <SectionHeading hint="Fires a banner on every page once the condition has held. Saved in this browser.">
                  Alerts
                </SectionHeading>
                <Card className="p-4">
                  <AlertSettings />
                </Card>
              </>
            ),
          },
        ]}
      />
    </div>
  );
}
