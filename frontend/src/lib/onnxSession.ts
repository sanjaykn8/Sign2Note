/**
 * Thin wrapper around onnxruntime-web for the live webcam demo. Loads the
 * model bytes from the backend's /model/onnx endpoint (see api.ts) and runs
 * inference entirely in the browser -- no keypoints or predictions are ever
 * sent to a server for this flow (see PRIVACY.md).
 */
import * as ort from "onnxruntime-web";

let sessionPromise: Promise<ort.InferenceSession> | null = null;

/** Loads (and caches) the ONNX session from raw model bytes. Call once per
 * page load; subsequent calls reuse the same session. */
export function loadOnnxSession(modelBytes: ArrayBuffer): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

export function resetOnnxSession() {
  sessionPromise = null;
}

/**
 * Runs one forward pass. `normalizedWindow` must already be normalized
 * (see webcamPipeline.normalizeWindow) and shaped as a flat Float32Array of
 * length maxLen*inputDim -- this function just wraps it into the tensor
 * shape the model expects ([1, maxLen, inputDim], input name "keypoints",
 * matching the ONNX export in train.py) and reads back "logits".
 */
export async function runInference(
  session: ort.InferenceSession,
  normalizedWindow: Float32Array,
  maxLen: number,
  inputDim: number
): Promise<Float32Array> {
  const tensor = new ort.Tensor("float32", normalizedWindow, [1, maxLen, inputDim]);
  const results = await session.run({ keypoints: tensor });
  return results.logits.data as Float32Array;
}
