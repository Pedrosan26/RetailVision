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
  /** Largest node-reported headcount at any single moment in the range. */
  peak_occupancy: number;
}

/** One person's visit as seen by one camera: their track's records folded into a single row. */
export interface Visit {
  camera_node_id: string;
  track_id: string;
  first_seen: string;
  last_seen: string;
  duration_seconds: number;
  zone_id: string | null;
  age_group: string;
  gender: string;
  dominant_emotion: string;
  emotion_distribution: Record<string, number>;
  events: number;
}

/** A zone's floor polygon in world meters, uploaded by a camera node at startup. */
export interface ZoneGeometry {
  zone_id: string;
  camera_node_id: string;
  polygon: Array<[number, number]>;
  updated_at: string;
}

export interface VisitFilters {
  since?: string;
  until?: string;
  zone_id?: string;
  camera_node_id?: string;
  limit?: number;
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

/**
 * Every emotion label the pipeline can emit, in a fixed order.
 *
 * The classifier collapses angry/disgust/fear/sad into "negative", so this
 * set is closed and known ahead of time. Charts colour by position in this
 * list rather than by position in whatever subset a given time range
 * happens to contain -- otherwise a quiet hour missing "surprise" would
 * repaint every other series, and a reader who learned "neutral is green"
 * would see it orange for no reason connected to the data.
 */
export const EMOTION_ORDER = ["happy", "neutral", "surprise", "negative"] as const;
