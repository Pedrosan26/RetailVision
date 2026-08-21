// Polls per-person visits for the Visits page. Same cadence as the other
// range-scoped queries: these are folded summaries, not a live ticker.

import { useQuery } from "@tanstack/react-query";
import { fetchVisits } from "../api/detections";
import type { VisitFilters } from "../api/types";

const POLL_INTERVAL_MS = 15000;

/** Returns visits matching the given filters, refetched on an interval. */
export function useVisits(filters: VisitFilters = {}) {
  return useQuery({
    queryKey: ["visits", filters],
    queryFn: () => fetchVisits(filters),
    refetchInterval: POLL_INTERVAL_MS,
  });
}
