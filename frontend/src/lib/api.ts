const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:3001";

export interface ProcessEvent {
  label: string;
  confidence: number;
  start_time: number;
  end_time: number;
}

export interface ProcessResult {
  notes_md: string;
  gloss_list: string[];
  events?: ProcessEvent[];
  segments: { window: number; label: string; confidence: number; start_time?: number; end_time?: number }[];
  confidence: number;
  backend: string;
  providers?: string[];
  low_confidence?: boolean;
  video_fps?: number;
}

export interface ModelMeta {
  max_len: number;
  input_dim: number;
  num_classes: number;
  label2id: Record<string, number>;
  id2label: Record<string, string>;
}

export type NotesMode = "template" | "llm" | "llama_cpp" | "ollama";

export async function uploadVideo(file: File, options: {
  notesMode?: NotesMode;
  llmModel?: string;
  style?: "concise" | "detailed" | "academic";
  threshold?: number;
  frameSkip?: number;
  stride?: number;
} = {}): Promise<ProcessResult> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("notes_mode", options.notesMode || "template");
  // Only send llm_model if the caller actually picked one -- otherwise let
  // the backend fall back to its own LLM_MODEL env-configured default
  // instead of us silently forcing a hard-coded model name.
  if (options.llmModel) fd.append("llm_model", options.llmModel);
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

/** Model metadata for the live webcam demo (max_len, input_dim, label
 * vocabulary) -- fetched once at session start so the browser knows how
 * to shape its sliding window and how to map predicted class indices back
 * to human-readable gloss labels. */
export async function getModelMeta(): Promise<ModelMeta> {
  const res = await fetch(`${API_BASE}/model/meta`);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
  return body;
}

/** Raw ONNX model bytes for onnxruntime-web to load and run entirely
 * client-side. */
export async function getModelOnnxBytes(): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}/model/onnx`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Server error ${res.status}`);
  }
  return res.arrayBuffer();
}

/** Generate notes directly from an already-recognized gloss sequence (the
 * live webcam session's "Generate Notes" button) -- no video/keypoints
 * involved, just the final recognized words. */
export async function generateNotesFromGlosses(
  glossList: string[],
  options: { notesMode?: NotesMode; llmModel?: string; style?: string } = {}
): Promise<{ notes_md: string }> {
  const res = await fetch(`${API_BASE}/notes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gloss_list: glossList,
      notes_mode: options.notesMode || "template",
      llm_model: options.llmModel,
      style: options.style || "concise",
    }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
  return body;
}
