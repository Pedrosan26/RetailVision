// Polls recent detections, for the Overview page's activity feed and the
// Detections page's table. Both want the same data with different filters.

import { useQuery } from "@tanstack/react-query";
import { fetchDetections } from "../api/detections";
import type { DetectionFilters } from "../api/types";

const POLL_INTERVAL_MS = 5000;

/** Returns recent detections matching the given filters, refetched on an interval. */
export function useRecentDetections(filters: DetectionFilters = {}) {
  return useQuery({
    queryKey: ["detections", filters],
    queryFn: () => fetchDetections(filters),
    refetchInterval: POLL_INTERVAL_MS,
  });
}
