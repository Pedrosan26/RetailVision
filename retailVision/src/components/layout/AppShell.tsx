// Top-level page layout: sidebar nav + routed page content. Every page
// renders inside this shell via react-router's <Outlet />.

import { Outlet } from "react-router-dom";
import { AlertBanner } from "../common/AlertBanner";
import { Sidebar } from "./Sidebar";

/** Wraps every route with the shared sidebar layout and the crowding alert banner. */
export function AppShell() {
  return (
    <div className="flex min-h-screen bg-[var(--app-page)]">
      <Sidebar />
      {/* Capped so lines of text and wide tables stay readable on a large
          display, rather than stretching the full width of the window. */}
      <main className="flex-1 overflow-x-auto px-8 py-7">
        <div className="mx-auto max-w-6xl">
          {/* In the shell, not a page: an alert must survive navigation. */}
          <AlertBanner />
          <Outlet />
        </div>
      </main>
    </div>
  );
}
