// Live view for one camera node: renders the server's MJPEG stream
// endpoint directly in an <img> tag (browsers support multipart JPEG
// streams natively), rather than polling a still image on an interval --
// avoids stacking a client poll delay on top of the stream's own rate.
// Shows an explicit offline state and retries the connection on a timer
// if the stream errors (server down, camera node not connected yet).

import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const RETRY_DELAY_MS = 3000;

/** Shows a camera node's live MJPEG stream, with an offline placeholder and auto-retry on failure. */
export function CameraFeed({ cameraNodeId }: { cameraNodeId: string }) {
  const [retryKey, setRetryKey] = useState(0);
  const [isOffline, setIsOffline] = useState(false);

  const handleError = () => {
    setIsOffline(true);
    setTimeout(() => setRetryKey((key) => key + 1), RETRY_DELAY_MS);
  };

  if (isOffline) {
    return (
      <div className="flex aspect-video w-full items-center justify-center rounded-md border border-slate-200 bg-slate-100 text-xs text-slate-400 dark:border-slate-800 dark:bg-slate-900">
        {cameraNodeId} offline -- retrying…
      </div>
    );
  }

  return (
    <img
      key={retryKey}
      src={`${API_BASE}/api/v1/frames/${cameraNodeId}/stream`}
      alt={`Live view from ${cameraNodeId}`}
      className="aspect-video w-full rounded-md border border-slate-200 bg-slate-100 object-cover dark:border-slate-800 dark:bg-slate-900"
      onLoad={() => setIsOffline(false)}
      onError={handleError}
    />
  );
}
