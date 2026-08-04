// Shared error placeholder for any panel whose query failed -- most
// commonly the server not running yet.

/** Renders a centered error message with the underlying failure reason. */
export function ErrorState({ message = "Couldn't reach the server." }: { message?: string }) {
  return (
    <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
      {message}
    </div>
  );
}
