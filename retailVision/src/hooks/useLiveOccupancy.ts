// Polls the live occupancy endpoint so the Overview page reflects
// entries/exits without a manual refresh.

import { useQuery } from "@tanstack/react-query";
import { fetchOccupancy } from "../api/detections";

const POLL_INTERVAL_MS = 5000;

/** Returns the latest occupancy per zone/camera node, refetched on an interval. */
export function useLiveOccupancy() {
  return useQuery({
    queryKey: ["occupancy", "live"],
    queryFn: fetchOccupancy,
    refetchInterval: POLL_INTERVAL_MS,
  });
}
