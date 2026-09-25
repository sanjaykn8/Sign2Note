/**
 * Pure, browser-API-free logic for the live webcam recognition pipeline.
 * Kept separate from onnxSession.ts/holisticLandmarker.ts (which touch
 * actual browser APIs) specifically so it can be unit-tested with plain
 * Vitest -- no DOM, no WASM, no camera required.
 *
 * Every function here must stay numerically consistent with the Python
 * training/inference pipeline (ml_service/feature_extraction.py and
 * ml_service/infer.py) -- see the docstring on each function for exactly
 * which Python function it mirrors and why.
 *
 * Schema v2 note: the actual hand+body+face feature vector construction
 * and normalization now live in featureSchema.ts (the canonical schema
 * shared with ml_service/feature_schema.py -- see FEATURE_SCHEMA.md), not
 * here. This file re-exports what it needs from there so existing
 * imports of `Landmark` from this module keep working, and focuses on
 * what's actually specific to the live session: buffering frames and the
 * real-time stability/commit state machine.
 */

import { FEATURE_DIM, normalizeSequence } from "./featureSchema";
export type { Landmark } from "./featureSchema";
export { buildFeatureVector, FEATURE_DIM } from "./featureSchema";

/**
 * Fixed-size ring buffer of the last `maxLen` per-frame keypoint vectors.
 * Mirrors ml_service's pad_or_trim() when the buffer isn't full yet (zero-
 * pads at the END, matching Python's np.vstack([x, zeros]) -- i.e. this
 * buffer's window is oldest-frame-first, and short windows are padded
 * after the real frames, not before). Default `dims` is FEATURE_DIM (285,
 * the hand+body+face schema) -- pass a different value only for tests.
 */
export class KeypointBuffer {
  private frames: Float32Array[] = [];
  constructor(private maxLen: number, private dims = FEATURE_DIM) {}

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
   * the model, or null if the buffer is completely empty. Normalization
   * is featureSchema.normalizeSequence() -- the same function
   * ml_service/feature_schema.py's normalize_sequence() mirrors, including
   * excluding pose-visibility dims from z-scoring when dims === FEATURE_DIM
   * (see FEATURE_SCHEMA.md "Normalization"). */
  getNormalizedWindow(): Float32Array | null {
    if (this.frames.length === 0) return null;
    const flat = new Float32Array(this.maxLen * this.dims);
    for (let f = 0; f < this.frames.length; f++) {
      flat.set(this.frames[f], f * this.dims);
    }
    // frames beyond this.frames.length stay zero -- matches pad_or_trim
    return normalizeSequence(flat, this.maxLen, this.dims);
  }
}

export interface PredictionEvent {
  label: string;
  confidence: number;
  timestamp: number; // seconds since session start
}

export interface SmoothingConfig {
  /** Below this confidence, a prediction is treated as NO_SIGN -- i.e. the
   * classifier is essentially guessing at noise (idle hands, a transition
   * between signs, nobody signing). NO_SIGN never contributes to the
   * stability streak and immediately resets it, the same way going back to
   * an IDLE state would. Spec zone: 0.00-0.49. */
  ignoreThreshold: number;
  /** At or above this confidence, a prediction is "confident" and can
   * accumulate stability toward a commit. Between `ignoreThreshold` and
   * `acceptThreshold` is the "uncertain" zone (spec: 0.50-0.74) -- probably
   * a real sign, but not reliable enough to count as a vote; it's a soft
   * tick that neither advances nor resets the current streak, so a brief
   * confidence dip in the middle of a held sign doesn't force the user to
   * start over. Spec zone: 0.75-1.00. */
  acceptThreshold: number;
  /** How many consecutive confident (>= acceptThreshold) raw predictions
   * must agree on the same label before it's committed to the session as
   * one event. This is what collapses "QUESTION QUESTION QUESTION
   * QUESTION" (repeated raw predictions of one held sign) into a single
   * QUESTION event, and rejects one-off flickers/noise. */
  stableCount: number;
}

export const DEFAULT_SMOOTHING: SmoothingConfig = {
  ignoreThreshold: 0.5,
  acceptThreshold: 0.75,
  stableCount: 3,
};

/**
 * Simplified real-time smoothing for the LIVE webcam session. This is
 * intentionally NOT the same algorithm as the backend's Viterbi/HMM
 * smoothing (inference_viterbi.py) used for uploaded-video processing --
 * that DP needs the whole clip's window probabilities up front, which
 * doesn't exist yet in a live stream. Instead this uses a simple
 * "N consecutive agreeing confident predictions -> commit one event, don't
 * commit again until the label changes" rule, gated by a three-zone
 * confidence read (NO_SIGN / uncertain / confident). This is a deliberate
 * simplification (see dev principle "do not over-engineer") documented
 * here and in ARCHITECTURE.md, not a claim of true HMM decoding in-browser.
 */
export class SessionSmoother {
  private recentLabel: string | null = null;
  private recentCount = 0;
  private lastCommittedLabel: string | null = null;
  private config: SmoothingConfig;

  constructor(config: Partial<SmoothingConfig> = {}) {
    this.config = { ...DEFAULT_SMOOTHING, ...config };
  }

  /**
   * Feed one raw (label, confidence) prediction. Returns:
   *   - {status: "no_sign"} if confidence is below `ignoreThreshold` --
   *     caller should show "No sign detected" and NOT touch the session
   *     history. Resets the stability streak (equivalent to returning to
   *     an IDLE state).
   *   - {status: "uncertain"} if confidence is in the middle zone -- caller
   *     should show "Uncertain -- hold steady". Does not touch the session
   *     history, and does not reset an in-progress streak either.
   *   - {status: "pending"} if confident but not yet stable for
   *     `stableCount` consecutive frames, or if it repeats the already-
   *     committed label (avoids re-appending a sign that's still being held)
   *   - {status: "committed", event} exactly once when a new stable,
   *     confident, label-change event should be appended to the session
   */
  update(label: string, confidence: number, timestamp: number):
    | { status: "no_sign" }
    | { status: "uncertain" }
    | { status: "pending" }
    | { status: "committed"; event: PredictionEvent } {
    if (confidence < this.config.ignoreThreshold) {
      this.recentLabel = null;
      this.recentCount = 0;
      return { status: "no_sign" };
    }

    if (confidence < this.config.acceptThreshold) {
      // Uncertain: a soft tick. Deliberately does NOT reset recentLabel/
      // recentCount -- a single low-ish-confidence frame in the middle of
      // an otherwise-held sign shouldn't force the stability streak to
      // restart from zero.
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

  /** Lets a committed label be re-committed again -- used when the caller
   * undoes or deletes a history event, so the smoother "forgets" that it
   * already emitted that label and won't silently swallow a genuine repeat
   * of the same sign later in the session. Pass the label of whatever is
   * now the last remaining event (or null if the history is now empty). */
  forgetLastCommitted(newLastLabel: string | null) {
    this.lastCommittedLabel = newLastLabel;
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
