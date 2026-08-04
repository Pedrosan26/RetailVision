// Compact list of the most recent detections, for the Overview page.
// The full filterable table is a separate page (RV-034).

import { useRecentDetections } from "../../hooks/useRecentDetections";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";

const RECENT_LIMIT = 8;

/** Renders the last few detections as a compact activity feed. */
export function RecentActivityFeed() {
  const { data, isPending, isError } = useRecentDetections({ limit: RECENT_LIMIT });

  if (isPending) return <LoadingState label="Loading recent activity…" />;
  if (isError) return <ErrorState message="Couldn't load recent detections -- is the server running?" />;
  if (data.length === 0) {
    return <div className="py-8 text-center text-sm text-slate-400">No detections reported yet.</div>;
  }

  return (
    <ul className="divide-y divide-slate-100 dark:divide-slate-800">
      {data.map((detection) => (
        <li key={detection.id} className="flex items-center justify-between py-3 text-sm">
          <div className="flex items-center gap-3">
            <span className="font-medium text-slate-700 dark:text-slate-300">{detection.camera_node_id}</span>
            <span className="text-slate-400">{detection.age_group}</span>
            <span className="text-slate-400">{detection.gender}</span>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
              {detection.emotion}
            </span>
          </div>
          <time className="font-mono text-xs tabular-nums text-slate-400" dateTime={detection.timestamp}>
            {new Date(detection.timestamp).toLocaleTimeString()}
          </time>
        </li>
      ))}
    </ul>
  );
}
