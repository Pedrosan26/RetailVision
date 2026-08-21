// Zustand store for client-side UI state only (selected camera node,
// time-range filter, alert thresholds) -- server data (detections/
// occupancy/aggregates) lives in TanStack Query's cache instead, not here.
// Keeping these separate avoids re-implementing fetching/caching/polling
// on top of Zustand.
//
// Alert thresholds persist to localStorage: a limit someone set on Monday
// should still hold on Tuesday, and there is no server-side settings store
// to keep it in. They are per-browser by design -- who wants to be warned
// is a property of the person watching, not of the deployment.

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type TimeWindow = "5m" | "1h" | "24h";

interface UiState {
  selectedCameraNodeId: string | null;
  selectedZoneId: string | null;
  timeWindow: TimeWindow;
  /** People in any one zone at or above this triggers an alert; null disables alerts. */
  alertLimit: number | null;
  /** Minutes the limit must hold before the alert fires, filtering momentary spikes. */
  alertMinutes: number;
  /** Per-page section ordering, keyed by page id, as chosen with the move controls. */
  sectionOrder: Record<string, string[]>;
  setSelectedCameraNodeId: (id: string | null) => void;
  setSelectedZoneId: (id: string | null) => void;
  setTimeWindow: (window: TimeWindow) => void;
  setAlertLimit: (limit: number | null) => void;
  setAlertMinutes: (minutes: number) => void;
  setSectionOrder: (page: string, order: string[]) => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCameraNodeId: null,
      selectedZoneId: null,
      timeWindow: "5m",
      alertLimit: null,
      alertMinutes: 5,
      sectionOrder: {},
      setSelectedCameraNodeId: (id) => set({ selectedCameraNodeId: id }),
      setSelectedZoneId: (id) => set({ selectedZoneId: id }),
      setTimeWindow: (window) => set({ timeWindow: window }),
      setAlertLimit: (limit) => set({ alertLimit: limit }),
      setAlertMinutes: (minutes) => set({ alertMinutes: minutes }),
      setSectionOrder: (page, order) =>
        set((state) => ({ sectionOrder: { ...state.sectionOrder, [page]: order } })),
    }),
    {
      name: "retailvision-ui",
      // Alert config and layout persist: a limit set on Monday and a page
      // arranged to taste should both survive to Tuesday. Selections and
      // ranges start fresh each session, or a filter set weeks ago silently
      // narrows today.
      partialize: (state) => ({
        alertLimit: state.alertLimit,
        alertMinutes: state.alertMinutes,
        sectionOrder: state.sectionOrder,
      }),
    },
  ),
);
