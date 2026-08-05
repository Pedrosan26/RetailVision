// Live view for one camera node: refetches the server's latest-frame
// endpoint on an interval and swaps the <img> src -- a periodically
// refreshed still image, not a real video stream, matching this
// dashboard's polling-over-WebSocket approach everywhere else.

import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const REFRESH_INTERVAL_MS = 1000;

/** Shows a camera node's most recently streamed frame, refreshed on an interval. */
export function CameraFeed({ cameraNodeId }: { cameraNodeId: string }) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  return (
    <img
      src={`${API_BASE}/api/v1/frames/${cameraNodeId}?t=${tick}`}
      alt={`Live view from ${cameraNodeId}`}
      className="aspect-video w-full rounded-md border border-slate-200 bg-slate-100 object-cover dark:border-slate-800 dark:bg-slate-900"
    />
  );
}
