// Catch-all route for unmatched paths.

import { Link } from "react-router-dom";

/** Renders a 404 message with a link back to the Overview page. */
export function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-24 text-center">
      <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Page not found</h1>
      <p className="text-sm text-slate-500 dark:text-slate-400">
        <Link to="/" className="text-slate-700 underline hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100">
          Back to Overview
        </Link>
      </p>
    </div>
  );
}
