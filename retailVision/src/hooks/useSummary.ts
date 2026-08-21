// Polls the headline figures for the KPI row. A slower interval than the
// occupancy poll: these are 24-hour totals, so a fresh number every few
// seconds would imply a precision the range does not have.

import { useQuery } from "@tanstack/react-query";
import { fetchSummary } from "../api/detections";

const POLL_INTERVAL_MS = 30000;

/** Returns the summary figures for a time range, refetched on an interval. */
export function useSummary(params: { since?: string; zone_id?: string } = {}) {
  return useQuery({
    queryKey: ["summary", params],
    queryFn: () => fetchSummary(params),
    refetchInterval: POLL_INTERVAL_MS,
  });
}
