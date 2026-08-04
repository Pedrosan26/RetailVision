// Thin fetch wrapper shared by every endpoint module in api/. Centralizes
// the base URL and error handling so callers just get back parsed JSON
// or a thrown ApiError.

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Builds a query string from a params object, dropping undefined/null values. */
function buildQuery(params?: Record<string, string | number | undefined>): string {
  if (!params) return "";
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null);
  if (entries.length === 0) return "";
  const search = new URLSearchParams(entries.map(([k, v]) => [k, String(v)]));
  return `?${search.toString()}`;
}

/** GETs a JSON resource from the server API, throwing ApiError on a non-2xx response. */
export async function apiGet<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}${buildQuery(params)}`);
  if (!response.ok) {
    throw new ApiError(`${response.status} ${response.statusText}`, response.status);
  }
  return response.json() as Promise<T>;
}
