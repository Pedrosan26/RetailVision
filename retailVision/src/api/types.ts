// Response shapes returned by the FastAPI server's read endpoints.
// Mirrors server/app/schemas/detection.py -- keep these in sync by hand,
// there's no shared codegen between the two languages.

export interface Detection {
  id: number;
  camera_node_id: string;
  timestamp: string;
  track_id: string | null;
  zone_id: string | null;
  world_x: number | null;
  world_y: number | null;
  count: number | null;
  age_group: string;
  gender: string;
  emotion: string;
  dwell_seconds: number | null;
  engagement_score: number | null;
}

export interface Occupancy {
  key: string;
  camera_node_id: string;
  zone_id: string | null;
  count: number | null;
  timestamp: string;
}

/** A zone's headcount with people seen by several cameras counted once. */
export interface ZoneOccupancy {
  zone_id: string;
  total: number;
  per_camera: Record<string, number>;
  cameras_reporting: number;
  timestamp: string;
}

/** Headline figures over one time range. unique_people is per camera: overlap counts twice. */
export interface Summary {
  since: string;
  until: string;
  total_detections: number;
  unique_people: number;
  avg_dwell_seconds: number | null;
  emotion_distribution: Record<string, number>;
  busiest_hour_start: string | null;
  busiest_hour_people: number;
}

export interface AggregateBucket {
  bucket_start: string;
  detection_count: number;
  /** Distinct people behind those events; a person present across a bucket contributes many events but one person. */
  unique_people: number;
  age_group_distribution: Record<string, number>;
  gender_distribution: Record<string, number>;
  emotion_distribution: Record<string, number>;
  avg_dwell_seconds: number | null;
  avg_engagement_score: number | null;
}

export interface DetectionFilters {
  limit?: number;
  camera_node_id?: string;
  zone_id?: string;
  since?: string;
  until?: string;
}

export interface AggregateFilters {
  window?: string;
  since?: string;
  until?: string;
  zone_id?: string;
  // Each is "any of these" within its dimension, and "and" across dimensions.
  // An omitted or empty list means that dimension is not filtered on.
  age_group?: string[];
  gender?: string[];
  emotion?: string[];
}
