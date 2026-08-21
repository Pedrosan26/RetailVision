// Fetches zone floor polygons. Polled slowly: geometry changes only when a
// room is re-surveyed and a node restarts, so a minute of staleness is fine.

import { useQuery } from "@tanstack/react-query";
import { fetchZoneGeometry } from "../api/detections";

const POLL_INTERVAL_MS = 60000;

/** Returns every zone's floor polygon, refetched on a slow interval. */
export function useZoneGeometry() {
  return useQuery({
    queryKey: ["zone-geometry"],
    queryFn: fetchZoneGeometry,
    refetchInterval: POLL_INTERVAL_MS,
  });
}
