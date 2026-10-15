import useSWR, { type SWRConfiguration } from "swr";

export const API = "/api/v1";

export class ApiError extends Error {
  constructor(public status: number, message: string, public details?: unknown) {
    super(message);
  }
}

async function handle(res: Response) {
  if (res.status === 401 && typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
    window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
  }
  if (!res.ok) {
    let msg = res.statusText;
    let details: unknown;
    try {
      const body = await res.json();
      msg = body?.error?.message ?? msg;
      details = body?.error?.details;
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(res.status, msg, details);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

export async function apiGet<T>(path: string): Promise<T> {
  return handle(await fetch(`${API}${path}`, { credentials: "same-origin" }));
}

export async function apiSend<T>(path: string, method: "POST" | "PUT" | "PATCH" | "DELETE", body?: unknown): Promise<T> {
  return handle(
    await fetch(`${API}${path}`, {
      method,
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-Requested-With": "splittability" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export function useApi<T>(path: string | null, config?: SWRConfiguration) {
  return useSWR<T>(path, (p: string) => apiGet<T>(p), { revalidateOnFocus: true, ...config });
}
