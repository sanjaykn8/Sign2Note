const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:3001";

export interface ProcessResult {
  notes_md: string;
  gloss_list: string[];
  segments: { window: number; label: string; confidence: number }[];
  confidence: number;
  backend: string;
  providers?: string[];
  low_confidence?: boolean;
}

export async function uploadVideo(file: File, options: {
  notesMode?: "template" | "llama_cpp";
  llmModel?: string;
  style?: "concise" | "detailed" | "academic";
  threshold?: number;
  frameSkip?: number;
  stride?: number;
} = {}): Promise<ProcessResult> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("notes_mode", options.notesMode || "template");
  fd.append("llm_model", options.llmModel || "gemma4");
  fd.append("style", options.style || "concise");
  if (options.threshold !== undefined) fd.append("threshold", String(options.threshold));
  if (options.frameSkip !== undefined) fd.append("frame_skip", String(options.frameSkip));
  if (options.stride !== undefined) fd.append("stride", String(options.stride));

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
  } catch (e) {
    throw new Error(
      "Couldn't reach the backend. Is it running at " + API_BASE + "?"
    );
  }

  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
  return body;
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
