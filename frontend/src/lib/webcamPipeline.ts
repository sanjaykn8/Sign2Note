/**
 * Pure, browser-API-free logic for the live webcam recognition pipeline.
 * Kept separate from onnxSession.ts/handLandmarker.ts (which touch actual
 * browser APIs) specifically so it can be unit-tested with plain Vitest --
 * no DOM, no WASM, no camera required.
 *
 * Every function here must stay numerically consistent with the Python
 * training/inference pipeline (ml_service/feature_extraction.py and
 * ml_service/infer.py) -- see the docstring on each function for exactly
 * which Python function it mirrors and why.
 */

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/**
 * Mirrors feature_extraction.py's `_extract_keypoints()`: 126-dim vector =
 * first-detected hand's 21 landmarks (x,y,z flattened) in the first 63
 * slots, second-detected hand's landmarks in the next 63 slots, zero-padded
 * if fewer than 2 hands are visible. NOTE: this is ordered by MediaPipe's
 * detection order, NOT by actual left/right handedness -- the Python
 * extractor doesn't use the handedness label either, so this must not be
 * "improved" to use real handedness, or it will no longer match what the
 * model was trained on.
 */
export function keypointsFromLandmarks(hands: Landmark[][]): Float32Array {
  const out = new Float32Array(126); // already zero-filled
  for (let h = 0; h < Math.min(hands.length, 2); h++) {
    const offset = h * 63;
    const lm = hands[h];
    for (let i = 0; i < 21 && i < lm.length; i++) {
      out[offset + i * 3] = lm[i].x;
      out[offset + i * 3 + 1] = lm[i].y;
      out[offset + i * 3 + 2] = lm[i].z;
    }
  }
  return out;
}

/**
 * Mirrors infer.py's `_normalize()`: per-window z-score across the TIME
 * axis independently for each of the 126 feature dimensions (mean/std
 * computed over the window's frames, not across features). `window` is a
 * flat Float32Array of length frames*126 (row-major: frame 0's 126 values,
 * then frame 1's, ...).
 */
export function normalizeWindow(window: Float32Array, frames: number, dims = 126): Float32Array {
  const mean = new Float64Array(dims);
  const std = new Float64Array(dims);
  for (let f = 0; f < frames; f++) {
    for (let d = 0; d < dims; d++) mean[d] += window[f * dims + d];
  }
  for (let d = 0; d < dims; d++) mean[d] /= frames;

  for (let f = 0; f < frames; f++) {
    for (let d = 0; d < dims; d++) {
      const diff = window[f * dims + d] - mean[d];
      std[d] += diff * diff;
    }
  }
  for (let d = 0; d < dims; d++) std[d] = Math.sqrt(std[d] / frames) + 1e-5;

  const out = new Float32Array(frames * dims);
  for (let f = 0; f < frames; f++) {
    for (let d = 0; d < dims; d++) {
      out[f * dims + d] = (window[f * dims + d] - mean[d]) / std[d];
    }
  }
  return out;
}

/**
 * Fixed-size ring buffer of the last `maxLen` per-frame keypoint vectors.
 * Mirrors infer.py's `_pad_window()` when the buffer isn't full yet (zero-
 * pads at the END, matching Python's np.vstack([x, zeros]) -- i.e. this
 * buffer's window is oldest-frame-first, and short windows are padded
 * after the real frames, not before).
 */
export class KeypointBuffer {
  private frames: Float32Array[] = [];
  constructor(private maxLen: number, private dims = 126) {}

  push(vec: Float32Array) {
    this.frames.push(vec);
    if (this.frames.length > this.maxLen) this.frames.shift();
  }

  get length() {
    return this.frames.length;
  }

  clear() {
    this.frames = [];
  }

  /** Returns a normalized (maxLen * dims) flat Float32Array ready to feed
   * the model, or null if the buffer is completely empty. */
  getNormalizedWindow(): Float32Array | null {
    if (this.frames.length === 0) return null;
    const flat = new Float32Array(this.maxLen * this.dims);
    for (let f = 0; f < this.frames.length; f++) {
      flat.set(this.frames[f], f * this.dims);
    }
    // frames beyond this.frames.length stay zero -- matches _pad_window
    return normalizeWindow(flat, this.maxLen, this.dims);
  }
}

export interface PredictionEvent {
  label: string;
  confidence: number;
  timestamp: number; // seconds since session start
}

export interface SmoothingConfig {
  /** Predictions below this confidence are never appended to the session
   * and are surfaced as "uncertain" instead. Spec default: 0.70. */
  acceptThreshold: number;
  /** How many consecutive raw predictions must agree on the same label
   * (each individually above acceptThreshold) before it's committed to the
   * session as one event. This is what collapses "QUESTION QUESTION
   * QUESTION QUESTION" (repeated raw predictions of one held sign) into a
   * single QUESTION event, and rejects one-off flickers/noise. */
  stableCount: number;
}

export const DEFAULT_SMOOTHING: SmoothingConfig = {
  acceptThreshold: 0.7,
  stableCount: 3,
};

/**
 * Simplified real-time smoothing for the LIVE webcam session. This is
 * intentionally NOT the same algorithm as the backend's Viterbi/HMM
 * smoothing (inference_viterbi.py) used for uploaded-video processing --
 * that DP needs the whole clip's window probabilities up front, which
 * doesn't exist yet in a live stream. Instead this uses a simple
 * "N consecutive agreeing high-confidence predictions -> commit one event,
 * don't commit again until the label changes" rule. This is a deliberate
 * simplification (see dev principle "do not over-engineer") documented
 * here and in ARCHITECTURE.md, not a claim of true HMM decoding in-browser.
 */
export class SessionSmoother {
  private recentLabel: string | null = null;
  private recentCount = 0;
  private lastCommittedLabel: string | null = null;

  constructor(private config: SmoothingConfig = DEFAULT_SMOOTHING) {}

  /**
   * Feed one raw (label, confidence) prediction. Returns:
   *   - {status: "uncertain"} if confidence is below threshold (caller
   *     should show "Low confidence -- please repeat" and NOT touch the
   *     session history)
   *   - {status: "pending"} if confident but not yet stable for
   *     `stableCount` consecutive frames, or if it repeats the already-
   *     committed label (avoids re-appending a sign that's still being held)
   *   - {status: "committed", event} exactly once when a new stable,
   *     confident, label-change event should be appended to the session
   */
  update(label: string, confidence: number, timestamp: number):
    { status: "uncertain" } | { status: "pending" } | { status: "committed"; event: PredictionEvent } {
    if (confidence < this.config.acceptThreshold) {
      this.recentLabel = null;
      this.recentCount = 0;
      return { status: "uncertain" };
    }

    if (label === this.recentLabel) {
      this.recentCount += 1;
    } else {
      this.recentLabel = label;
      this.recentCount = 1;
    }

    const isStable = this.recentCount >= this.config.stableCount;
    const isNewSign = label !== this.lastCommittedLabel;

    if (isStable && isNewSign) {
      this.lastCommittedLabel = label;
      return { status: "committed", event: { label, confidence, timestamp } };
    }
    return { status: "pending" };
  }

  reset() {
    this.recentLabel = null;
    this.recentCount = 0;
    this.lastCommittedLabel = null;
  }
}

/** Softmax + argmax over raw model logits (Float32Array of length
 * numClasses). Mirrors the numerically-stable softmax used in infer.py
 * (subtract max before exponentiating). */
export function softmaxArgmax(logits: Float32Array | number[]): { index: number; confidence: number; probs: number[] } {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sum);
  let index = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;
  return { index, confidence: probs[index], probs };
}
