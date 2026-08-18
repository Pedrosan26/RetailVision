// Catch-all route for unmatched paths.

import { Link } from "react-router-dom";

/** Renders a 404 message with a link back to the Overview page. */
export function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
      <h1 className="text-2xl font-semibold text-[var(--app-ink)]">Page not found</h1>
      <p className="text-sm text-[var(--app-ink-muted)]">
        <Link to="/" className="text-[var(--app-accent)] underline">
          Back to Overview
        </Link>
      </p>
    </div>
  );
}
