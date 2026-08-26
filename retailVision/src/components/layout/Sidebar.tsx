// Left-hand navigation.
//
// Each item carries a one-line description of what the page answers, because
// "Zones" and "Detections" both sound like they might hold the same thing
// until you have opened both.

import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Overview", hint: "Live view", end: true },
  { to: "/zones", label: "Zones", hint: "Per-area history" },
  { to: "/cameras", label: "Cameras", hint: "What each node sees" },
  { to: "/detections", label: "Visits", hint: "One row per person" },
];

/** Renders the sidebar nav, highlighting the currently active route. */
export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-[var(--app-line)] bg-[var(--app-surface)] px-3 py-5">
      <div className="mb-6 px-2">
        <div className="text-sm font-semibold tracking-tight text-[var(--app-ink)]">RetailVision</div>
        <div className="mt-0.5 text-[0.7rem] text-[var(--app-ink-muted)]">Zone occupancy</div>
      </div>

      <nav className="flex flex-col gap-0.5">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `rounded-md px-3 py-2 transition-colors ${
                isActive
                  ? "bg-[var(--app-accent-wash)] text-[var(--app-accent)]"
                  : "text-[var(--app-ink-secondary)] hover:bg-[var(--app-page)]"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span className="block text-sm font-medium">{item.label}</span>
                <span
                  className={`block text-[0.7rem] ${
                    isActive ? "text-[var(--app-accent)] opacity-80" : "text-[var(--app-ink-muted)]"
                  }`}
                >
                  {item.hint}
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
