// Polls the per-zone headcount endpoint. Distinct from useLiveOccupancy,
// which reports what each camera individually said: several cameras watching
// one zone all see each other's subjects, so their counts cannot simply be
// added. The server merges detections by world position to answer how many
// people are actually present.

import { useQuery } from "@tanstack/react-query";
import { fetchZoneOccupancy } from "../api/detections";

const POLL_INTERVAL_MS = 5000;

/** Returns each zone's deduplicated headcount plus the per-camera figures behind it. */
export function useZoneOccupancy() {
  return useQuery({
    queryKey: ["occupancy", "zones"],
    queryFn: fetchZoneOccupancy,
    refetchInterval: POLL_INTERVAL_MS,
  });
}
