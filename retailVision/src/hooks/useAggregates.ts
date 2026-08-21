// Polls time-windowed aggregate rollups for the Overview page's charts.

import { useQuery } from "@tanstack/react-query";
import { fetchAggregates } from "../api/detections";
import type { AggregateFilters } from "../api/types";

const POLL_INTERVAL_MS = 15000;

/** Returns aggregate buckets matching the given filters, refetched on an interval. */
export function useAggregates(filters: AggregateFilters = {}) {
  return useQuery({
    queryKey: ["aggregates", filters],
    queryFn: () => fetchAggregates(filters),
    refetchInterval: POLL_INTERVAL_MS,
  });
}
