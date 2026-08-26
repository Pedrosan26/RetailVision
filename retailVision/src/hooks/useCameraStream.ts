// Subscribes to one camera node's live frames over a WebSocket.
//
// The server pushes JPEG frames as binary messages; each becomes an object
// URL for an <img> to display. The previous URL is revoked as soon as the
// next arrives -- at fifteen frames a second, an unreleased blob per frame
// leaks memory by the megabyte within a minute.
//
// Reconnects on its own, because the interesting case is a camera node that
// is restarted or briefly unreachable, and a viewer left showing a frozen
// frame with no way back is worse than a brief "reconnecting".

import { useEffect, useState } from "react";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const RECONNECT_MS = 2000;

export type StreamStatus = "connecting" | "live" | "offline";

/** Converts the configured HTTP base URL to its WebSocket equivalent. */
function socketBase(): string {
  if (BASE_URL.startsWith("https://")) return `wss://${BASE_URL.slice("https://".length)}`;
  if (BASE_URL.startsWith("http://")) return `ws://${BASE_URL.slice("http://".length)}`;
  return BASE_URL;
}

/** Returns the latest frame from a camera node as an object URL, plus the connection status. */
export function useCameraStream(cameraNodeId: string | null) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<StreamStatus>("connecting");

  useEffect(() => {
    if (!cameraNodeId) {
      setFrameUrl(null);
      return;
    }

    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryTimer: number | undefined;
    let objectUrl: string | null = null;

    const release = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    };

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      socket = new WebSocket(`${socketBase()}/api/v1/ws/watch/${encodeURIComponent(cameraNodeId)}`);
      socket.binaryType = "blob";

      socket.onopen = () => !cancelled && setStatus("live");
      socket.onmessage = (event) => {
        if (cancelled || !(event.data instanceof Blob)) return;
        const next = URL.createObjectURL(event.data);
        release();
        objectUrl = next;
        setFrameUrl(next);
        setStatus("live");
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (cancelled) return;
        setStatus("offline");
        retryTimer = window.setTimeout(connect, RECONNECT_MS);
      };
    };

    connect();

    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
      socket?.close();
      release();
    };
  }, [cameraNodeId]);

  return { frameUrl, status };
}
