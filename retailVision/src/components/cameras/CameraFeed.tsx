// One camera node's live view, sized to be actually looked at.
//
// object-contain rather than object-cover: the frame carries drawn bounding
// boxes and zone overlays, and cropping to fill a fixed aspect ratio would
// cut off the very annotations the view exists to show.

import { useCameraStream } from "../../hooks/useCameraStream";

export interface CameraFeedProps {
  /** Camera node to watch, or null to show nothing. */
  cameraNodeId: string | null;
}

/** Renders the live frame from one camera node, with its connection state. */
export function CameraFeed({ cameraNodeId }: CameraFeedProps) {
  const { frameUrl, status } = useCameraStream(cameraNodeId);

  return (
    <figure className="m-0">
      <div className="relative overflow-hidden rounded-lg border border-[var(--app-line)] bg-black">
        {frameUrl ? (
          <img
            src={frameUrl}
            alt={`Live view from ${cameraNodeId}, with detected faces outlined`}
            className="mx-auto block max-h-[75vh] w-full object-contain"
          />
        ) : (
          <div className="flex aspect-video w-full items-center justify-center px-6 text-center text-sm text-[var(--app-ink-muted)]">
            {cameraNodeId === null
              ? "Select a camera to watch."
              : "Waiting for the first frame. The node must run with --stream-frames."}
          </div>
        )}

        <div className="absolute left-3 top-3 flex items-center gap-2 rounded-md bg-black/60 px-2.5 py-1 text-xs font-medium text-white">
          <span
            aria-hidden
            className={`inline-block h-2 w-2 rounded-full ${
              status === "live" ? "bg-[var(--viz-series-3)]" : "bg-[var(--viz-series-2)]"
            }`}
          />
          {cameraNodeId ?? "no camera"}
          <span className="text-white/70">{status}</span>
        </div>
      </div>
      <figcaption className="mt-2 text-xs text-[var(--app-ink-muted)]">
        Frames are held in the server's memory for viewing only. They are never written to disk or the
        database, and are not part of the anonymized detection record.
      </figcaption>
    </figure>
  );
}
