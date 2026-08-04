// Grid of live occupancy cards, one per zone/camera node, with the
// loading/error/empty states a live-polled panel needs.

import { useLiveOccupancy } from "../../hooks/useLiveOccupancy";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";
import { OccupancyCard } from "./OccupancyCard";

/** Renders the live occupancy grid, polling the server on an interval. */
export function OccupancyGrid() {
  const { data, isPending, isError } = useLiveOccupancy();

  if (isPending) return <LoadingState label="Loading occupancy…" />;
  if (isError) return <ErrorState message="Couldn't load occupancy -- is the server running?" />;
  if (data.length === 0) {
    return <div className="py-8 text-center text-sm text-slate-400">No detections reported yet.</div>;
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {data.map((occupancy) => (
        <OccupancyCard key={occupancy.key} occupancy={occupancy} />
      ))}
    </div>
  );
}
