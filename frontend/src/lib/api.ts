const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:3001";

export interface ProcessResult {
  notes_md: string;
  gloss_list: string[];
  segments: { window: number; label: string; confidence: number }[];
  confidence: number;
  backend: string;
  providers?: string[];
}

export async function uploadVideo(file: File, options: {
  notesMode?: "template" | "ollama";
  ollamaModel?: string;
  style?: "concise" | "detailed" | "academic";
} = {}): Promise<ProcessResult> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("notes_mode", options.notesMode || "template");
  fd.append("ollama_model", options.ollamaModel || "llama3.2:3b");
  fd.append("style", options.style || "concise");

  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
  return body;
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
