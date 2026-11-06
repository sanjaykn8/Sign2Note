import { useCallback, useEffect, useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { checkHealth, type HealthResult } from "@/lib/api";
import { ApiError, apiFetch, defaultServerUrl, getApiToken, getServerUrl, setApiToken, setServerUrl } from "@/lib/serverConfig";
import { Settings2 } from "lucide-react";

type Status =
  | { kind: "checking" }
  | { kind: "ok"; health: HealthResult }
  | { kind: "degraded"; message: string; health?: HealthResult }
  | { kind: "down"; message: string };

const POLL_MS = 30000;

function summarize(h: HealthResult): Status {
  if (h.ml === "unavailable") return { kind: "degraded", message: "Server reachable, but its ML service is not running.", health: h };
  if (!h.ml.model_ready) {
    return { kind: "degraded", message: h.ml.model_problem || "No usable recognition model on the server.", health: h };
  }
  return { kind: "ok", health: h };
}

/**
 * Server reachability + recognition-model availability, with the
 * Connection settings (server address and optional access token) that
 * other devices need to reach the main laptop. Polls /health every 30 s
 * and on window focus -- never continuously.
 */
export default function ConnectionStatus() {
  const [status, setStatus] = useState<Status>({ kind: "checking" });
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState(getServerUrl());
  const [token, setToken] = useState(getApiToken());
  const [formError, setFormError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const h = await checkHealth();
      setStatus(summarize(h));
      // /health is public; check the token too when the server requires one.
      if (h.auth_required) {
        const r = await apiFetch("/model/meta", {}, { timeoutMs: 8000 });
        if (r.status === 401) setStatus({ kind: "degraded", message: "Server requires an access token (enter it below).", health: h });
      }
    } catch (err) {
      setStatus({ kind: "down", message: err instanceof ApiError || err instanceof Error ? err.message : "Server unreachable." });
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    const onFocus = () => refresh();
    window.addEventListener("focus", onFocus);
    return () => {
      clearInterval(id);
      window.removeEventListener("focus", onFocus);
    };
  }, [refresh]);

  const save = async () => {
    setFormError(null);
    try {
      setServerUrl(url === defaultServerUrl() ? "" : url);
      setApiToken(token);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Invalid server address.");
      return;
    }
    setStatus({ kind: "checking" });
    await refresh();
  };

  const dot =
    status.kind === "ok" ? "bg-emerald-500" : status.kind === "checking" ? "bg-muted-foreground/40" : status.kind === "degraded" ? "bg-amber-500" : "bg-destructive";
  const label =
    status.kind === "ok" ? "Server connected" : status.kind === "checking" ? "Checking server…" : status.kind === "degraded" ? "Server: limited" : "Server offline";
  const ml = status.kind === "ok" || status.kind === "degraded" ? status.health?.ml : undefined;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`${label}. Open connection settings`}
        >
          <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
          <span className="hidden md:inline">{label}</span>
          <Settings2 className="h-3.5 w-3.5" aria-hidden />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-3 text-sm">
        <div>
          <p className="font-semibold">Connection</p>
          <p className="text-xs text-muted-foreground" role="status">
            {status.kind === "ok" ? "Connected." : status.kind === "checking" ? "Checking…" : status.message}
          </p>
          {ml && ml !== "unavailable" && (
            <ul className="mt-2 space-y-0.5 text-xs text-muted-foreground">
              <li>
                Recognition model: {ml.model_ready ? `ready (${ml.model_version ?? "unknown version"}, feature schema v${ml.feature_schema_version}${ml.is_legacy_schema ? ", legacy hand-only" : ""})` : "not available"}
              </li>
              <li>Language-generation service: {ml.llm_provider} / {ml.llm_model} (template fallback always available)</li>
            </ul>
          )}
        </div>
        <label className="block space-y-1">
          <span className="text-xs font-medium">Server address</span>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder={defaultServerUrl()}
            className="w-full rounded border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          />
          <span className="block text-[11px] text-muted-foreground">
            The main laptop's gateway, e.g. http://192.168.1.20:3001. Default: this page's host, port 3001.
          </span>
        </label>
        <label className="block space-y-1">
          <span className="text-xs font-medium">Access token (if the server requires one)</span>
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            autoComplete="off"
            className="w-full rounded border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </label>
        {formError && <p className="text-xs text-destructive">{formError}</p>}
        <div className="flex justify-end gap-2">
          <Button size="sm" variant="outline" onClick={refresh}>
            Test
          </Button>
          <Button size="sm" onClick={save}>
            Save
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
