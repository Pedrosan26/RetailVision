// Which camera nodes currently have a publisher connected, so the cameras
// page can offer them. Polled rather than pushed: the list changes only when
// a node starts or stops, which is rare next to the frame rate itself.

import { useQuery } from "@tanstack/react-query";
import { fetchLiveCameras } from "../api/detections";

const POLL_INTERVAL_MS = 5000;

/** Returns the camera node ids currently streaming frames. */
export function useLiveCameras() {
  return useQuery({
    queryKey: ["live-cameras"],
    queryFn: fetchLiveCameras,
    refetchInterval: POLL_INTERVAL_MS,
  });
}
