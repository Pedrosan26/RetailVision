// Shared loading placeholder for any panel waiting on a query.

/** Renders a centered loading message inside its parent container. */
export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return <div className="py-10 text-center text-sm text-[var(--app-ink-muted)]">{label}</div>;
}
