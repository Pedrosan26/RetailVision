// Live camera views, one at a time and large.
//
// A grid of thumbnails was the previous shape and it made the feeds too
// small to read -- the point of showing the annotated frame is to see which
// faces were detected and where the zone boundary falls, neither of which
// survives being shrunk into a card. One camera fills the space, and a row
// of buttons switches between them.

import { useEffect, useState } from "react";
import { CameraFeed } from "../components/cameras/CameraFeed";
import { ErrorState } from "../components/common/ErrorState";
import { LoadingState } from "../components/common/LoadingState";
import { EmptyState, PageHeader } from "../components/common/ui";
import { useLiveCameras } from "../hooks/useLiveCameras";

/** Renders one camera node's live view at full width, with a selector for the others. */
export function CamerasPage() {
  const { data: cameras, isPending, isError } = useLiveCameras();
  const [selected, setSelected] = useState<string | null>(null);

  // Follow the available list: pick the first camera once one appears, and
  // fall back to another if the one being watched stops streaming.
  useEffect(() => {
    if (!cameras || cameras.length === 0) return;
    setSelected((current) => (current !== null && cameras.includes(current) ? current : cameras[0]));
  }, [cameras]);

  if (isPending) return <LoadingState label="Looking for camera streams…" />;
  if (isError) return <ErrorState message="Couldn't reach the server to list cameras." />;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Cameras"
        description="What each node is seeing right now, with its detections drawn on."
        actions={
          cameras.length > 1 ? (
            <div className="flex flex-wrap gap-1.5">
              {cameras.map((cameraNodeId) => (
                <button
                  key={cameraNodeId}
                  type="button"
                  onClick={() => setSelected(cameraNodeId)}
                  aria-pressed={cameraNodeId === selected}
                  className={`rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                    cameraNodeId === selected
                      ? "border-[var(--app-accent)] bg-[var(--app-accent-wash)] text-[var(--app-accent)]"
                      : "border-[var(--app-line-strong)] text-[var(--app-ink-secondary)]"
                  }`}
                >
                  {cameraNodeId}
                </button>
              ))}
            </div>
          ) : undefined
        }
      />

      {cameras.length === 0 ? (
        <EmptyState
          title="No camera is streaming right now."
          hint="Streaming is opt-in: start a node with --stream-frames alongside --server-url. Everything else on the dashboard works without it."
        />
      ) : (
        <CameraFeed cameraNodeId={selected} />
      )}
    </div>
  );
}
