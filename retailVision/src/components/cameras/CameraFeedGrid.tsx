// Grid of live camera feeds, one per camera node currently reporting
// occupancy -- reuses the occupancy poll rather than a separate query,
// since the set of known camera nodes is the same either way.

import { useLiveOccupancy } from "../../hooks/useLiveOccupancy";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";
import { CameraFeed } from "./CameraFeed";

/** Renders one live-preview tile per camera node currently reporting occupancy. */
export function CameraFeedGrid() {
  const { data, isPending, isError } = useLiveOccupancy();

  if (isPending) return <LoadingState label="Loading cameras…" />;
  if (isError) return <ErrorState message="Couldn't load cameras -- is the server running?" />;
  if (data.length === 0) {
    return <div className="py-8 text-center text-sm text-[var(--app-ink-muted)]">No camera nodes reporting yet.</div>;
  }

  const cameraNodeIds = [...new Set(data.map((occupancy) => occupancy.camera_node_id))];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {cameraNodeIds.map((cameraNodeId) => (
        <div key={cameraNodeId}>
          <div className="mb-1 text-xs font-medium text-[var(--app-ink-muted)]">{cameraNodeId}</div>
          <CameraFeed cameraNodeId={cameraNodeId} />
        </div>
      ))}
    </div>
  );
}
