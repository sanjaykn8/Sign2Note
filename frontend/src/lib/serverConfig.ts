/**
 * Where the Sign2Notes gateway (backend/server.js) lives, and how to reach
 * it reliably.
 *
 * Resolution order for the server URL:
 *   1. a URL the user saved under "Connection settings" (localStorage)
 *   2. VITE_API_URL (build-time)
 *   3. the host this page was loaded from, port 3001
 *
 * (3) is what makes other devices work out of the box: a laptop that opens
 * http://<main-laptop-ip>:8080 talks to http://<main-laptop-ip>:3001. The
 * old default, "http://localhost:3001", pointed every other device at
 * ITSELF, so only the main laptop could ever reach the backend.
 */

const URL_KEY = "sign2notes.serverUrl";
const TOKEN_KEY = "sign2notes.apiToken";

function safeGet(key: string): string | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage.getItem(key) : null;
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string | null) {
  try {
    if (typeof localStorage === "undefined") return;
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* storage unavailable (private mode) -- setting lasts for this page only */
  }
}

let memoryUrl: string | null = null;
let memoryToken: string | null = null;

export function defaultServerUrl(): string {
  const env = import.meta.env?.VITE_API_URL as string | undefined;
  if (env) return env.replace(/\/+$/, "");
  if (typeof window !== "undefined" && window.location?.hostname) {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    return `${protocol}//${window.location.hostname}:3001`;
  }
  return "http://localhost:3001";
}

export function getServerUrl(): string {
  return (memoryUrl ?? safeGet(URL_KEY) ?? defaultServerUrl()).replace(/\/+$/, "");
}

export function getApiToken(): string {
  return memoryToken ?? safeGet(TOKEN_KEY) ?? "";
}

/** Validates and stores the server URL (empty string = back to default). */
export function setServerUrl(url: string): void {
  const trimmed = url.trim();
  if (trimmed) {
    const parsed = new URL(trimmed); // throws on garbage
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Server URL must start with http:// or https://");
    }
  }
  memoryUrl = trimmed ? trimmed.replace(/\/+$/, "") : null;
  safeSet(URL_KEY, memoryUrl);
}

export function setApiToken(token: string): void {
  memoryToken = token.trim() || null;
  safeSet(TOKEN_KEY, memoryToken);
}

export class ApiError extends Error {
  status: number;
  body: Record<string, unknown>;
  constructor(message: string, status: number, body: Record<string, unknown> = {}) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export interface ApiFetchOptions {
  /** Abort after this many ms (default 30 s). */
  timeoutMs?: number;
  /** Extra attempts after a NETWORK failure / 502-504. Only use for
   * idempotent requests. Bounded -- never retries forever. */
  retries?: number;
  retryDelayMs?: number;
  signal?: AbortSignal;
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * fetch() against the configured server with: auth header, timeout,
 * bounded retry on transient failures, and errors turned into readable
 * messages (`ApiError.status` 0 = could not reach the server at all).
 */
export async function apiFetch(path: string, init: RequestInit = {}, opts: ApiFetchOptions = {}): Promise<Response> {
  const { timeoutMs = 30000, retries = 0, retryDelayMs = 800 } = opts;
  const url = `${getServerUrl()}${path}`;
  const headers = new Headers(init.headers);
  const token = getApiToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let lastError: unknown = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    opts.signal?.addEventListener("abort", onAbort);
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...init, headers, signal: controller.signal });
      if ([502, 503, 504].includes(res.status) && attempt < retries) {
        lastError = new ApiError(`Server error ${res.status}`, res.status);
      } else {
        return res;
      }
    } catch (err) {
      if (opts.signal?.aborted) throw new ApiError("Request cancelled.", 0);
      const timedOut = controller.signal.aborted;
      lastError = new ApiError(
        timedOut
          ? `The server at ${getServerUrl()} did not respond within ${Math.round(timeoutMs / 1000)} s.`
          : `Couldn't reach the server at ${getServerUrl()}. Check that it is running and that this device is on the same network (Connection settings).`,
        0
      );
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onAbort);
    }
    if (attempt < retries) await sleep(retryDelayMs * (attempt + 1));
  }
  throw lastError;
}

/** apiFetch + JSON parse + non-2xx -> ApiError with the server's message. */
export async function apiJson<T>(path: string, init: RequestInit = {}, opts: ApiFetchOptions = {}): Promise<T> {
  const res = await apiFetch(path, init, opts);
  const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok) {
    const msg =
      typeof body.error === "string"
        ? body.error
        : Array.isArray(body.detail)
          ? `Invalid request: ${(body.detail as { msg?: string }[]).map((d) => d.msg).join("; ")}`
          : res.status === 401
            ? "The server requires an access token (Connection settings)."
            : `Server error ${res.status}`;
    throw new ApiError(msg, res.status, body);
  }
  return body as T;
}
