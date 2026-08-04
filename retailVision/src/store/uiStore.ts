// Zustand store for client-side UI state only (selected camera node,
// time-range filter) -- server data (detections/occupancy/aggregates)
// lives in TanStack Query's cache instead, not here. Keeping these
// separate avoids re-implementing fetching/caching/polling on top of
// Zustand.

import { create } from "zustand";

export type TimeWindow = "5m" | "1h" | "24h";

interface UiState {
  selectedCameraNodeId: string | null;
  selectedZoneId: string | null;
  timeWindow: TimeWindow;
  setSelectedCameraNodeId: (id: string | null) => void;
  setSelectedZoneId: (id: string | null) => void;
  setTimeWindow: (window: TimeWindow) => void;
}

export const useUiStore = create<UiState>((set) => ({
  selectedCameraNodeId: null,
  selectedZoneId: null,
  timeWindow: "5m",
  setSelectedCameraNodeId: (id) => set({ selectedCameraNodeId: id }),
  setSelectedZoneId: (id) => set({ selectedZoneId: id }),
  setTimeWindow: (window) => set({ timeWindow: window }),
}));
