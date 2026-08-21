// Shared error placeholder for any panel whose query failed -- most
// commonly the server not running yet.

/** Renders an error message with the underlying failure reason. */
export function ErrorState({ message = "Couldn't reach the server." }: { message?: string }) {
  return (
    <div className="rounded-md border border-[var(--app-critical)] bg-[var(--app-critical-wash)] px-4 py-3 text-sm text-[var(--app-critical)]">
      {message}
    </div>
  );
}
