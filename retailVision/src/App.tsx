// Top-level app: TanStack Query provider + route table.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DetectionsPage } from "./pages/DetectionsPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { OverviewPage } from "./pages/OverviewPage";
import { ZonesPage } from "./pages/ZonesPage";

const queryClient = new QueryClient();

/** Renders the dashboard: query provider, router, and the shell's routed pages. */
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<OverviewPage />} />
            <Route path="detections" element={<DetectionsPage />} />
            <Route path="zones" element={<ZonesPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
