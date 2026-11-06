/**
 * Client for the Sign2Notes gateway. Two separate services sit behind it:
 *
 *   Recognition (signs -> glosses):     getModelMeta, getModelOnnxBytes,
 *                                       recognizeFromFeatures (server also
 *                                       offers POST /recognize/video)
 *   Language generation (glosses -> text): generateNotesFromGlosses,
 *                                       generateTranscriptFromGlosses
 *
 * uploadVideo() is the upload page's combined call (recognition + notes).
 * Server address / token / timeouts: see serverConfig.ts.
 */
import { apiFetch, apiJson, ApiError, getApiToken, getServerUrl } from "./serverConfig";

export { getServerUrl } from "./serverConfig";

export type NoteStyle = "concise" | "detailed" | "academic";
export type NotesMode = "template" | "llm" | "llama_cpp" | "ollama";

export interface ProcessEvent {
  label: string;
  confidence: number;
  start_time: number;
  end_time: number;
}

export interface Segment {
  window: number;
  label: string;
  confidence: number;
  start_time?: number;
  end_time?: number;
  alternatives?: { label: string; confidence: number }[];
}

/** Which generator actually produced a piece of text -- shown to the user
 * so a deterministic fallback is never presented as LLM output. */
export interface GenerationMeta {
  generator: "llm" | "template";
  fallback_reason: string | null;
  style?: NoteStyle;
  prompt_version?: string;
  llm_model?: string | null;
}

export interface ProcessResult {
  notes_md: string;
  gloss_list: string[];
  events?: ProcessEvent[];
  segments: Segment[];
  confidence: number;
  backend: string;
  providers?: string[];
  low_confidence?: boolean;
  video_fps?: number;
  video_duration_seconds?: number | null;
  generation?: GenerationMeta;
  model_version?: string;
  feature_schema_version?: string;
}

export interface ModelMeta {
  max_len: number;
  input_dim: number;
  num_classes: number;
  label2id: Record<string, number>;
  id2label: Record<string, string>;
  feature_schema_version?: string;
  feature_type?: string;
  is_legacy_schema?: boolean;
  architecture?: string;
  vocab_hash?: string;
  best_val_accuracy?: number | null;
  trained_at?: string | null;
  model_version?: string;
  ready?: boolean;
  onnx_available?: boolean;
  /** Samples per second the model was trained on (7.5 for FDMSE-ISL at frame_skip 8). */
  sample_hz?: number;
  /** Real frames per recognition window (zero-padded to max_len). */
  window_frames?: number;
  window_stride?: number;
  /** Non-fatal notes about vocab drift / model problems. */
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
  segments: Segment[];
  confidence: number;
  backend: string;
  model_version: string;
  feature_schema_version: string;
}

export interface HealthResult {
  gateway: string;
  auth_required?: boolean;
  ml:
    | "unavailable"
    | {
        status: string;
        model_ready: boolean;
        model_problem?: string | null;
        model_version?: string | null;
        feature_schema_version?: string | null;
        is_legacy_schema?: boolean | null;
        onnx_ready?: boolean;
        llm_provider?: string;
        llm_model?: string;
      };
  error?: string;
}

export const MAX_UPLOAD_MB = Number(import.meta.env?.VITE_MAX_UPLOAD_MB || 200);
export const ACCEPTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

/** Client-side check before uploading (the server enforces the same rules). */
export function validateVideoFile(file: File): string | null {
  const name = file.name.toLowerCase();
  const extOk = ACCEPTED_VIDEO_EXTENSIONS.some((e) => name.endsWith(e));
  if (!extOk && !file.type.startsWith("video/")) {
    return `“${file.name}” is not a supported video. Use ${ACCEPTED_VIDEO_EXTENSIONS.join(", ")}.`;
  }
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `The video is ${(file.size / 1024 / 1024).toFixed(0)} MB; the limit is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

export interface UploadOptions {
  notesMode?: NotesMode;
  llmModel?: string;
  style?: NoteStyle;
  threshold?: number;
  frameSkip?: number;
  stride?: number;
  /** Upload progress 0..1 (bytes sent). Processing time after the upload
   * completes is not reported by the server, only "processing". */
  onUploadProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

/**
 * Upload a video for recognition + notes. Uses XMLHttpRequest (not fetch)
 * because fetch has no upload-progress events.
 */
export function uploadVideo(file: File, options: UploadOptions = {}): Promise<ProcessResult> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("notes_mode", options.notesMode || "template");
  if (options.llmModel) fd.append("llm_model", options.llmModel);
  fd.append("style", options.style || "concise");
  if (options.threshold !== undefined) fd.append("threshold", String(options.threshold));
  // frame_skip is only sent if the caller explicitly chose one: by default
  // the server derives it from the video's fps to match training sampling.
  if (options.frameSkip !== undefined) fd.append("frame_skip", String(options.frameSkip));
  if (options.stride !== undefined) fd.append("stride", String(options.stride));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${getServerUrl()}/upload`);
    const token = getApiToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.timeout = 35 * 60 * 1000;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) options.onUploadProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(xhr.responseText || "{}");
      } catch {
        /* non-JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as unknown as ProcessResult);
      else
        reject(
          new ApiError(
            typeof body.error === "string"
              ? body.error
              : xhr.status === 401
                ? "The server requires an access token (Connection settings)."
                : `Server error ${xhr.status}`,
            xhr.status,
            body
          )
        );
    };
    xhr.onerror = () =>
      reject(new ApiError(`Couldn't reach the server at ${getServerUrl()}. Is it running? (Connection settings)`, 0));
    xhr.ontimeout = () => reject(new ApiError("The server took too long to process this video.", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled.", 0));
    options.signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(fd);
  });
}

export async function checkHealth(): Promise<HealthResult> {
  const res = await apiFetch("/health", {}, { timeoutMs: 8000 });
  return res.json();
}

/** Metadata of the server's recognition model (max_len, input_dim, label
 * vocabulary, feature schema, sampling rate) -- fetched once per session. */
export function getModelMeta(): Promise<ModelMeta> {
  return apiJson<ModelMeta>("/model/meta", {}, { timeoutMs: 15000, retries: 2 });
}

/** Raw ONNX model bytes for onnxruntime-web to run recognition locally. */
export async function getModelOnnxBytes(): Promise<ArrayBuffer> {
  const res = await apiFetch("/model/onnx", {}, { timeoutMs: 120000, retries: 1 });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.error || `Server error ${res.status}`, res.status, body);
  }
  return res.arrayBuffer();
}

export interface NotesResult {
  notes_md: string;
  style?: NoteStyle;
  generation?: GenerationMeta;
}

/** Language generation: ordered glosses -> structured notes. Only gloss
 * words (plus optional confidences/context) are sent -- no video. */
export function generateNotesFromGlosses(
  glossList: string[],
  options: {
    notesMode?: NotesMode;
    llmModel?: string;
    style?: NoteStyle | string;
    confidences?: number[];
    context?: string;
  } = {}
): Promise<NotesResult> {
  return apiJson<NotesResult>(
    "/notes",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gloss_list: glossList,
        notes_mode: options.notesMode || "template",
        llm_model: options.llmModel,
        style: options.style || "concise",
        confidences: options.confidences && options.confidences.length === glossList.length ? options.confidences : undefined,
        context: options.context?.trim() || undefined,
      }),
    },
    { timeoutMs: 240000 }
  );
}

/** Language generation: ordered glosses -> natural-language transcript. */
export function generateTranscriptFromGlosses(
  glossList: string[],
  options: { notesMode?: NotesMode; llmModel?: string; confidences?: number[] } = {}
): Promise<{ transcript: string; generation?: GenerationMeta }> {
  return apiJson(
    "/transcript",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gloss_list: glossList,
        notes_mode: options.notesMode || "template",
        llm_model: options.llmModel,
        confidences: options.confidences && options.confidences.length === glossList.length ? options.confidences : undefined,
      }),
    },
    { timeoutMs: 240000 }
  );
}

/**
 * Recognition on the SERVER from keypoints extracted in this browser (never
 * raw video). Used when no compatible local recognizer is available.
 * `features` must be in the served model's schema (ModelMeta.input_dim).
 */
export async function recognizeFromFeatures(
  features: number[][],
  options: { fps?: number; frameSkip?: number; stride?: number; threshold?: number; windowFrames?: number; signal?: AbortSignal } = {}
): Promise<RecognizeResult> {
  try {
    return await apiJson<RecognizeResult>(
      "/recognize",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features,
          fps: options.fps,
          frame_skip: options.frameSkip ?? 8,
          stride: options.stride ?? 4,
          threshold: options.threshold,
          window_frames: options.windowFrames,
        }),
      },
      { timeoutMs: 20000, signal: options.signal }
    );
  } catch (err) {
    if (err instanceof ApiError && err.body?.expected_feature_dim) {
      throw new ApiError(
        `${err.message} (expected ${err.body.expected_feature_dim}-dim features, got ${err.body.got_feature_dim})`,
        err.status,
        err.body
      );
    }
    throw err;
  }
}
