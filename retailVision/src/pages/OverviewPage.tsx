// Landing page: live occupancy across all camera nodes/zones, plus a
// recent-activity feed.

import { CameraFeedGrid } from "../components/cameras/CameraFeedGrid";
import { RecentActivityFeed } from "../components/detections/RecentActivityFeed";
import { OccupancyGrid } from "../components/occupancy/OccupancyGrid";

/** Renders the Overview page: live camera feeds, occupancy grid, and recent activity feed. */
export function OverviewPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Overview</h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Live occupancy and recent detections across all camera nodes.
        </p>
      </div>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Live camera feeds
        </h2>
        <CameraFeedGrid />
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Live occupancy
        </h2>
        <OccupancyGrid />
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Recent activity
        </h2>
        <div className="rounded-lg border border-slate-200 bg-white px-5 dark:border-slate-800 dark:bg-slate-950">
          <RecentActivityFeed />
        </div>
      </section>
    </div>
  );
}
