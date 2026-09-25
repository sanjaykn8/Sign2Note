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
  feature_schema_version?: string;
  architecture?: string;
  vocab_hash?: string;
  best_val_accuracy?: number | null;
  trained_at?: string | null;
  /** Non-fatal notes about why the currently-loaded checkpoint might not
   * match the current feature schema / vocab.json -- see RULE 26. Empty
   * array means no known issues. */
  compatibility_warnings?: string[];
}

export interface RecognizeEvent {
  label: string;
  confidence: number;
  start_time?: number;
  end_time?: number;
}

export interface RecognizeResult {
  glosses: string[];
  events: RecognizeEvent[];
  segments: { window: number; label: string; confidence: number; start_time?: number; end_time?: number }[];
  confidence: number;
  backend: string;
  model_version: string;
  feature_schema_version: string;
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

/** Generate a natural-language TRANSCRIPT from an already-recognized gloss
 * sequence (Live Transcription mode's "Stop & Generate" step) -- flowing
 * prose, distinct from generateNotesFromGlosses()'s structured notes. See
 * ml_service/notes_generator.py's module docstring on the "three output
 * layers" for why these are separate calls, not one with a style flag. */
export async function generateTranscriptFromGlosses(
  glossList: string[],
  options: { notesMode?: NotesMode; llmModel?: string } = {}
): Promise<{ transcript: string }> {
  const res = await fetch(`${API_BASE}/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gloss_list: glossList,
      notes_mode: options.notesMode || "template",
      llm_model: options.llmModel,
    }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Server error ${res.status}`);
  return body;
}

/**
 * Recognition-only fallback: send already-extracted keypoints (never raw
 * video/frames) to the main server's recognition model and get back
 * glosses -- no notes are generated here (recognition and note generation
 * are deliberately separate concerns, see RULE 27). This is the
 * distributed-fallback path (ARCHITECTURE.md "Distributed architecture" /
 * RULE 8) for a client that can extract landmarks locally (MediaPipe runs
 * in every modern browser) but doesn't have a working local recognition
 * model yet -- e.g. the ONNX model hasn't finished downloading, or failed
 * a compatibility check against the current feature schema.
 *
 * `features` must already be in the canonical hand+body+face layout (see
 * FEATURE_SCHEMA.md / frontend/src/lib/featureSchema.ts) -- this function
 * does not reshape or validate the vectors itself beyond what the server
 * reports back on a 422.
 */
export async function recognizeFromFeatures(
  features: number[][],
  options: { fps?: number; frameSkip?: number; stride?: number; threshold?: number } = {}
): Promise<RecognizeResult> {
  const res = await fetch(`${API_BASE}/recognize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features,
      fps: options.fps,
      frame_skip: options.frameSkip ?? 8,
      stride: options.stride ?? 12,
      threshold: options.threshold,
    }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      body.error ||
        `Server error ${res.status}` +
          (body.expected_feature_dim
            ? ` (expected ${body.expected_feature_dim}-dim features, got ${body.got_feature_dim})`
            : "")
    );
  }
  return body;
}
