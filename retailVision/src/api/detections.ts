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
