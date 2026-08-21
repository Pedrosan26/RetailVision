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

export type QueryValue = string | number | undefined | null | Array<string | number>;

/** Builds a query string from a params object, dropping empty values and repeating array values as one param each. */
function buildQuery(params?: Record<string, QueryValue>): string {
  if (!params) return "";
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    // An array repeats the key once per value, which is how the server reads
    // "any of these". An empty array appends nothing, so a filter that selects
    // everything is sent as no filter at all rather than as an impossible one.
    if (Array.isArray(value)) {
      for (const item of value) search.append(key, String(item));
    } else {
      search.append(key, String(value));
    }
  }
  const query = search.toString();
  return query ? `?${query}` : "";
}

/** GETs a JSON resource from the server API, throwing ApiError on a non-2xx response. */
export async function apiGet<T>(path: string, params?: Record<string, QueryValue>): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}${buildQuery(params)}`);
  if (!response.ok) {
    throw new ApiError(`${response.status} ${response.statusText}`, response.status);
  }
  return response.json() as Promise<T>;
}
