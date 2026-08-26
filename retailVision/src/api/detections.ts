// Typed wrappers around the server's read endpoints. One function per
// endpoint, called from the hooks/ layer -- keeps TanStack Query hooks
// free of URL/query-string details.

import { apiGet } from "./client";
import type {
  AggregateBucket,
  AggregateFilters,
  Detection,
  DetectionFilters,
  Occupancy,
  Summary,
  Visit,
  VisitFilters,
  ZoneGeometry,
  ZoneOccupancy,
} from "./types";

/** Fetches recent detections, newest first, optionally filtered. */
export function fetchDetections(filters: DetectionFilters = {}): Promise<Detection[]> {
  return apiGet<Detection[]>("/api/v1/detections", { ...filters });
}

/** Fetches each camera node's latest reported count, one row per camera and zone. */
export function fetchOccupancy(): Promise<Occupancy[]> {
  return apiGet<Occupancy[]>("/api/v1/occupancy/live");
}

/** Fetches each zone's headcount, with a person seen by several cameras counted once. */
export function fetchZoneOccupancy(): Promise<ZoneOccupancy[]> {
  return apiGet<ZoneOccupancy[]>("/api/v1/occupancy/zones");
}

/** Fetches time-windowed aggregate rollups. */
export function fetchAggregates(filters: AggregateFilters = {}): Promise<AggregateBucket[]> {
  return apiGet<AggregateBucket[]>("/api/v1/aggregates", { ...filters });
}

/** Fetches the headline figures for one time range (default: server-side last 24h). */
export function fetchSummary(
  params: { since?: string; until?: string; zone_id?: string } = {},
): Promise<Summary> {
  return apiGet<Summary>("/api/v1/summary", { ...params });
}

/** Fetches per-person visits, newest first, optionally filtered. */
export function fetchVisits(filters: VisitFilters = {}): Promise<Visit[]> {
  return apiGet<Visit[]>("/api/v1/visits", { ...filters });
}

/** Fetches every zone's floor polygon, as uploaded by the camera nodes. */
export function fetchZoneGeometry(): Promise<ZoneGeometry[]> {
  return apiGet<ZoneGeometry[]>("/api/v1/zones/geometry");
}

/** Fetches the camera node ids currently streaming live frames. */
export async function fetchLiveCameras(): Promise<string[]> {
  const body = await apiGet<{ cameras: string[] }>("/api/v1/frames/cameras");
  return body.cameras;
}
