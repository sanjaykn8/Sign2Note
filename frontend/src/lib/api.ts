const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:3001";

export interface ProcessResult {
  notes_md: string;
  gloss_list: string[];
}

export async function uploadVideo(
  file: File,
  options: { useLlama?: boolean; useOllama?: boolean } = {}
): Promise<ProcessResult> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("use_llama", String(options.useLlama ?? false));
  fd.append("use_ollama", String(options.useOllama ?? false));

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Server error ${res.status}`);
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
