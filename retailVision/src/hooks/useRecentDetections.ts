// Polls recent detections for the Overview page's activity feed and
// (later, RV-034) the full detections table.

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
