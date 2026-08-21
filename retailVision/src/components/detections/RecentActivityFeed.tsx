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
    return <div className="py-8 text-center text-sm text-[var(--app-ink-muted)]">No detections reported yet.</div>;
  }

  return (
    <ul className="divide-y divide-[var(--app-line)]">
      {data.map((detection) => (
        <li key={detection.id} className="flex items-center justify-between py-3 text-sm">
          <div className="flex items-center gap-3">
            <span className="font-medium text-[var(--app-ink)]">{detection.camera_node_id}</span>
            <span className="text-[var(--app-ink-muted)]">{detection.age_group}</span>
            <span className="text-[var(--app-ink-muted)]">{detection.gender}</span>
            <span className="rounded-full border border-[var(--app-line-strong)] px-2 py-0.5 text-xs font-medium text-[var(--app-ink-secondary)]">
              {detection.emotion}
            </span>
          </div>
          <time className="font-mono text-xs tabular-nums text-[var(--app-ink-muted)]" dateTime={detection.timestamp}>
            {new Date(detection.timestamp).toLocaleTimeString()}
          </time>
        </li>
      ))}
    </ul>
  );
}
