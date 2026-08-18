// Landing page: what is happening right now.
//
// Ordered by how often it is looked at rather than by how the system is
// built -- the headcount first, then the cameras producing it, then the raw
// event stream. History lives on the Zones page, where it can be scoped to
// one area; repeating it here would make the live view slower to read.

import { CameraFeedGrid } from "../components/cameras/CameraFeedGrid";
import { RecentActivityFeed } from "../components/detections/RecentActivityFeed";
import { Card, CardHeader, PageHeader, SectionHeading } from "../components/common/ui";
import { ZoneOccupancyGrid } from "../components/occupancy/ZoneOccupancyGrid";

/** Renders the Overview page: live per-zone occupancy, camera feeds, and recent activity. */
export function OverviewPage() {
  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Overview"
        description="Live occupancy across every marked zone, and what each camera is currently seeing."
      />

      <section>
        <SectionHeading hint="Someone visible to several cameras is counted once.">
          People per zone
        </SectionHeading>
        <ZoneOccupancyGrid />
      </section>

      <section>
        <SectionHeading hint="Frames leave a camera node only while it runs with --stream-frames.">
          Camera feeds
        </SectionHeading>
        <CameraFeedGrid />
      </section>

      <section>
        <SectionHeading>Recent activity</SectionHeading>
        <Card>
          <CardHeader title="Latest detections" description="Newest first, across all cameras." />
          <div className="px-4">
            <RecentActivityFeed />
          </div>
        </Card>
      </section>
    </div>
  );
}
