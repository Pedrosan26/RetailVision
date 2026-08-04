// Top-level page layout: sidebar nav + routed page content. Every page
// renders inside this shell via react-router's <Outlet />.

import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";

/** Wraps every route with the shared sidebar layout. */
export function AppShell() {
  return (
    <div className="flex min-h-screen bg-slate-50 dark:bg-slate-900">
      <Sidebar />
      <main className="flex-1 overflow-x-auto px-8 py-6">
        <Outlet />
      </main>
    </div>
  );
}
