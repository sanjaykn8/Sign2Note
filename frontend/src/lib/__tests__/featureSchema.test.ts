import { describe, it, expect } from "vitest";
import {
  FEATURE_DIM,
  HAND_FEATURE_DIM,
  POSE_FEATURE_DIM,
  FACE_FEATURE_DIM,
  FEATURE_GROUPS,
  POSE_LANDMARK_INDICES,
  FACE_LANDMARK_INDICES,
  POSE_VISIBILITY_INDICES,
  buildFeatureVector,
  normalizeSequence,
  assertFeatureDim,
  FeatureSchemaError,
  type Landmark,
} from "../featureSchema";

function hand(seed: number): Landmark[] {
  return Array.from({ length: 21 }, (_, i) => ({ x: seed + i, y: seed + i + 0.1, z: seed + i + 0.2 }));
}

function fullPose(): Landmark[] {
  // 33 landmarks, value = index itself, so tests can verify exactly which
  // indices got pulled into the feature vector.
  return Array.from({ length: 33 }, (_, i) => ({ x: i, y: i * 10, z: i * 100, visibility: i / 32 }));
}

function fullFace(): Landmark[] {
  return Array.from({ length: 468 }, (_, i) => ({ x: i, y: i * 10, z: i * 100 }));
}

describe("feature dimension bookkeeping", () => {
  it("matches the documented group sizes", () => {
    expect(HAND_FEATURE_DIM).toBe(63);
    expect(POSE_FEATURE_DIM).toBe(36);
    expect(FACE_FEATURE_DIM).toBe(123);
    expect(FEATURE_DIM).toBe(63 + 63 + 36 + 123);
    expect(FEATURE_DIM).toBe(285);
  });

  it("feature groups cover the whole vector with no gaps or overlap", () => {
    const covered = new Array(FEATURE_DIM).fill(false);
    for (const [, start, end] of FEATURE_GROUPS) {
      for (let i = start; i < end; i++) {
        expect(covered[i]).toBe(false); // no overlap
        covered[i] = true;
      }
    }
    expect(covered.every(Boolean)).toBe(true); // no gaps
  });

  it("pose and face index lists have no duplicates", () => {
    expect(new Set(POSE_LANDMARK_INDICES).size).toBe(POSE_LANDMARK_INDICES.length);
    expect(new Set(FACE_LANDMARK_INDICES).size).toBe(FACE_LANDMARK_INDICES.length);
    expect(FACE_LANDMARK_INDICES.length).toBe(41);
  });
});

describe("buildFeatureVector -- missing-landmark behavior", () => {
  it("both hands present", () => {
    const vec = buildFeatureVector(hand(0), hand(100), null, null);
    expect(vec.length).toBe(FEATURE_DIM);
    expect(vec[0]).toBeCloseTo(0); // left wrist x
    expect(vec[HAND_FEATURE_DIM]).toBeCloseTo(100); // right wrist x
    for (let i = 2 * HAND_FEATURE_DIM; i < FEATURE_DIM; i++) expect(vec[i]).toBe(0);
  });

  it("one hand missing is a deterministic zero, not a shift", () => {
    const vec = buildFeatureVector(null, hand(100), null, null);
    for (let i = 0; i < HAND_FEATURE_DIM; i++) expect(vec[i]).toBe(0);
    expect(vec[HAND_FEATURE_DIM]).toBeCloseTo(100);
  });

  it("nothing detected -> all zeros, correct length", () => {
    const vec = buildFeatureVector(null, null, null, null);
    expect(vec.length).toBe(FEATURE_DIM);
    expect(Array.from(vec).every((v) => v === 0)).toBe(true);
  });

  it("empty arrays behave the same as null", () => {
    const a = buildFeatureVector(null, null, null, null);
    const b = buildFeatureVector([], [], [], []);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("pose pulls exactly the documented 9 indices, in order, including visibility", () => {
    const vec = buildFeatureVector(null, null, fullPose(), null);
    const poseStart = FEATURE_GROUPS[2][1];
    POSE_LANDMARK_INDICES.forEach((srcIdx, slot) => {
      const base = poseStart + slot * 4;
      expect(vec[base]).toBeCloseTo(srcIdx); // x == landmark's own index
      expect(vec[base + 3]).toBeCloseTo(srcIdx / 32); // visibility carried through
    });
  });

  it("face pulls exactly the documented 41 indices, in order", () => {
    const vec = buildFeatureVector(null, null, null, fullFace());
    const faceStart = FEATURE_GROUPS[3][1];
    FACE_LANDMARK_INDICES.forEach((srcIdx, slot) => {
      const base = faceStart + slot * 3;
      expect(vec[base]).toBeCloseTo(srcIdx);
    });
  });

  it("an out-of-range landmark index degrades to zero, not a crash", () => {
    const shortPose: Landmark[] = [{ x: 1, y: 1, z: 1, visibility: 1 }]; // only index 0
    const vec = buildFeatureVector(null, null, shortPose, null);
    const poseStart = FEATURE_GROUPS[2][1];
    expect(vec[poseStart]).toBeCloseTo(1); // NOSE (index 0) present
    expect(vec[poseStart + 4]).toBe(0); // LEFT_SHOULDER (index 11) -> zero, no throw
  });
});

describe("assertFeatureDim", () => {
  it("accepts the current schema dim", () => {
    expect(() => assertFeatureDim(FEATURE_DIM)).not.toThrow();
  });

  it("rejects the old v1 hand-only dim with a helpful hint", () => {
    expect(() => assertFeatureDim(126)).toThrow(FeatureSchemaError);
    try {
      assertFeatureDim(126, "some context");
    } catch (e) {
      expect(String(e)).toMatch(/126/);
      expect(String(e)).toMatch(/hand-only/);
    }
  });

  it("rejects an arbitrary wrong dim", () => {
    expect(() => assertFeatureDim(999)).toThrow(FeatureSchemaError);
  });
});

describe("normalizeSequence", () => {
  it("z-scores coordinate dimensions (zero mean, unit-ish variance over time)", () => {
    const frames = 50;
    const flat = new Float32Array(frames * FEATURE_DIM);
    // deterministic pseudo-random-ish values via a simple LCG so this
    // doesn't depend on Math.random()
    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return (seed / 0x7fffffff) * 4 - 2;
    };
    for (let i = 0; i < flat.length; i++) flat[i] = rand();

    const visSet = new Set(POSE_VISIBILITY_INDICES);
    const z = normalizeSequence(flat, frames, FEATURE_DIM);
    for (let d = 0; d < FEATURE_DIM; d++) {
      if (visSet.has(d)) continue;
      let mean = 0;
      for (let f = 0; f < frames; f++) mean += z[f * FEATURE_DIM + d];
      mean /= frames;
      expect(Math.abs(mean)).toBeLessThan(1e-2);
    }
  });

  it("does not z-score pose visibility -- passes a near-constant value through", () => {
    const frames = 30;
    const flat = new Float32Array(frames * FEATURE_DIM).fill(0.1);
    const visIdx = POSE_VISIBILITY_INDICES[0];
    for (let f = 0; f < frames; f++) flat[f * FEATURE_DIM + visIdx] = 0.97;

    const z = normalizeSequence(flat, frames, FEATURE_DIM);
    for (let f = 0; f < frames; f++) {
      expect(z[f * FEATURE_DIM + visIdx]).toBeCloseTo(0.97, 5);
    }
  });

  it("clips out-of-range visibility defensively", () => {
    const flat = new Float32Array(5 * FEATURE_DIM);
    const visIdx = POSE_VISIBILITY_INDICES[0];
    for (let f = 0; f < 5; f++) flat[f * FEATURE_DIM + visIdx] = 1.4;
    const z = normalizeSequence(flat, 5, FEATURE_DIM);
    for (let f = 0; f < 5; f++) expect(z[f * FEATURE_DIM + visIdx]).toBeLessThanOrEqual(1);
  });

  it("does not divide by zero for a constant (zero-variance) dimension", () => {
    const flat = new Float32Array(5 * FEATURE_DIM).fill(3.0);
    const z = normalizeSequence(flat, 5, FEATURE_DIM);
    expect(Array.from(z).every((v) => Number.isFinite(v))).toBe(true);
  });
});
