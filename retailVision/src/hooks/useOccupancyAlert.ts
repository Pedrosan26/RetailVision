// Watches zone occupancy against the configured threshold and reports
// which zones have been over it for long enough to matter.
//
// "Long enough" is tracked client-side from the polls this page already
// makes: the first poll where a zone reaches the limit starts its clock,
// any poll below the limit clears it. That makes the alert a property of
// the open dashboard rather than of the server -- consistent with the
// threshold itself living in this browser's settings.

import { useRef } from "react";
import { useUiStore } from "../store/uiStore";
import { useZoneOccupancy } from "./useZoneOccupancy";

export interface OccupancyAlert {
  zone_id: string;
  total: number;
  /** How long the zone has been at or over the limit, in whole minutes. */
  minutes: number;
}

/** Returns the zones currently over the configured limit for the configured duration. */
export function useOccupancyAlert(): OccupancyAlert[] {
  const { alertLimit, alertMinutes } = useUiStore();
  const { data } = useZoneOccupancy();
  const overSince = useRef(new Map<string, number>());

  if (alertLimit == null || !data) return [];

  const now = Date.now();
  const alerts: OccupancyAlert[] = [];
  const seen = new Set<string>();

  for (const zone of data) {
    seen.add(zone.zone_id);
    if (zone.total >= alertLimit) {
      if (!overSince.current.has(zone.zone_id)) overSince.current.set(zone.zone_id, now);
      const heldMs = now - (overSince.current.get(zone.zone_id) ?? now);
      if (heldMs >= alertMinutes * 60_000) {
        alerts.push({ zone_id: zone.zone_id, total: zone.total, minutes: Math.floor(heldMs / 60_000) });
      }
    } else {
      overSince.current.delete(zone.zone_id);
    }
  }
  // A zone that stopped reporting entirely also stops counting toward an alert.
  for (const zoneId of [...overSince.current.keys()]) {
    if (!seen.has(zoneId)) overSince.current.delete(zoneId);
  }

  return alerts;
}
