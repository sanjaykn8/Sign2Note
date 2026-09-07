import { describe, it, expect } from "vitest";
import {
  keypointsFromLandmarks,
  normalizeWindow,
  KeypointBuffer,
  SessionSmoother,
  softmaxArgmax,
  DEFAULT_SMOOTHING,
} from "../webcamPipeline";

describe("keypointsFromLandmarks", () => {
  it("returns all zeros when no hands are detected", () => {
    const vec = keypointsFromLandmarks([]);
    expect(vec.length).toBe(126);
    expect(Array.from(vec).every((v) => v === 0)).toBe(true);
  });

  it("places the first detected hand in slot 0 (first 63 values), zero-pads the rest", () => {
    const hand = Array.from({ length: 21 }, (_, i) => ({ x: i * 0.01, y: i * 0.02, z: i * 0.03 }));
    const vec = keypointsFromLandmarks([hand]);
    expect(vec[0]).toBeCloseTo(0);
    expect(vec[1]).toBeCloseTo(0);
    expect(vec[2]).toBeCloseTo(0);
    expect(vec[60]).toBeCloseTo(20 * 0.01); // landmark 20's x
    // second hand slot (indices 63..125) must be all zero
    expect(Array.from(vec.slice(63)).every((v) => v === 0)).toBe(true);
  });

  it("places a second detected hand starting at index 63, by detection order not handedness", () => {
    const handA = Array.from({ length: 21 }, () => ({ x: 1, y: 1, z: 1 }));
    const handB = Array.from({ length: 21 }, () => ({ x: 2, y: 2, z: 2 }));
    const vec = keypointsFromLandmarks([handA, handB]);
    expect(vec[0]).toBe(1);
    expect(vec[63]).toBe(2);
  });

  it("ignores hands beyond the first two", () => {
    const h = (v: number) => Array.from({ length: 21 }, () => ({ x: v, y: v, z: v }));
    const vec = keypointsFromLandmarks([h(1), h(2), h(3)]);
    expect(vec.length).toBe(126);
    expect(vec[0]).toBe(1);
    expect(vec[63]).toBe(2);
  });
});

describe("normalizeWindow", () => {
  it("produces zero-mean, unit-ish-variance per feature dimension across time", () => {
    const frames = 10;
    const dims = 4;
    const window = new Float32Array(frames * dims);
    for (let f = 0; f < frames; f++) {
      for (let d = 0; d < dims; d++) window[f * dims + d] = f * (d + 1); // linear ramp per dim
    }
    const normalized = normalizeWindow(window, frames, dims);
    for (let d = 0; d < dims; d++) {
      let mean = 0;
      for (let f = 0; f < frames; f++) mean += normalized[f * dims + d];
      mean /= frames;
      expect(mean).toBeCloseTo(0, 5);
    }
  });

  it("does not divide by zero for a constant (zero-variance) dimension", () => {
    const frames = 5;
    const dims = 1;
    const window = new Float32Array(frames).fill(3.0);
    const normalized = normalizeWindow(window, frames, dims);
    expect(Array.from(normalized).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("matches a hand-computed reference for a small case", () => {
    // frames=3, dims=1: values [1, 2, 3] -> mean=2, std=sqrt(((1)^2+0+1)/3)=sqrt(0.6667)
    const window = new Float32Array([1, 2, 3]);
    const normalized = normalizeWindow(window, 3, 1);
    const std = Math.sqrt(((1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 2) ** 2) / 3) + 1e-5;
    expect(normalized[0]).toBeCloseTo((1 - 2) / std, 4);
    expect(normalized[1]).toBeCloseTo((2 - 2) / std, 4);
    expect(normalized[2]).toBeCloseTo((3 - 2) / std, 4);
  });
});

describe("KeypointBuffer", () => {
  it("returns null when empty", () => {
    const buf = new KeypointBuffer(4, 2);
    expect(buf.getNormalizedWindow()).toBeNull();
  });

  it("zero-pads at the END when fewer than maxLen frames have been pushed", () => {
    const buf = new KeypointBuffer(4, 2);
    buf.push(new Float32Array([1, 1]));
    buf.push(new Float32Array([1, 1]));
    const win = buf.getNormalizedWindow()!;
    expect(win.length).toBe(4 * 2);
    // last two frames (padding) should normalize the SAME zero value
    // differently from the real frames -- just check shape/finiteness here,
    // exact values are covered by the normalizeWindow tests above.
    expect(Array.from(win).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("drops the oldest frame once maxLen is exceeded (ring-buffer behavior)", () => {
    const buf = new KeypointBuffer(2, 1);
    buf.push(new Float32Array([10]));
    buf.push(new Float32Array([20]));
    buf.push(new Float32Array([30])); // should evict the "10" frame
    expect(buf.length).toBe(2);
  });
});

describe("softmaxArgmax", () => {
  it("picks the highest-logit class and returns a valid probability distribution", () => {
    const { index, confidence, probs } = softmaxArgmax([1, 5, 2]);
    expect(index).toBe(1);
    expect(confidence).toBeGreaterThan(0.5);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("is numerically stable for large logits", () => {
    const { confidence } = softmaxArgmax([1000, 1001, 999]);
    expect(Number.isFinite(confidence)).toBe(true);
  });
});

describe("SessionSmoother", () => {
  it("reports 'uncertain' below the accept threshold and does not commit", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.7, stableCount: 3 });
    const r = s.update("QUESTION", 0.4, 1.0);
    expect(r.status).toBe("uncertain");
  });

  it("does not commit until stableCount consecutive agreeing predictions arrive", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.7, stableCount: 3 });
    expect(s.update("QUESTION", 0.9, 0).status).toBe("pending");
    expect(s.update("QUESTION", 0.85, 1).status).toBe("pending");
    const r3 = s.update("QUESTION", 0.88, 2);
    expect(r3.status).toBe("committed");
    if (r3.status === "committed") {
      expect(r3.event.label).toBe("QUESTION");
    }
  });

  it("collapses a long run of the same held sign into exactly ONE committed event", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.7, stableCount: 3 });
    const results = [];
    for (let i = 0; i < 10; i++) results.push(s.update("QUESTION", 0.9, i));
    const committed = results.filter((r) => r.status === "committed");
    expect(committed.length).toBe(1); // NOT ten QUESTION events
  });

  it("resets the stability counter when the raw label flickers, then commits once it settles", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.7, stableCount: 3 });
    s.update("DEFINITION", 0.9, 0);
    s.update("DEFINITION", 0.9, 1);
    s.update("EXAMPLE", 0.9, 2); // flicker resets the DEFINITION run
    s.update("DEFINITION", 0.9, 3);
    s.update("DEFINITION", 0.9, 4);
    const r = s.update("DEFINITION", 0.9, 5);
    expect(r.status).toBe("committed");
  });

  it("commits a new sign after a different sign has already been committed", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.7, stableCount: 2 });
    s.update("DEFINITION", 0.9, 0);
    const r1 = s.update("DEFINITION", 0.9, 1);
    expect(r1.status).toBe("committed");

    s.update("EXAMPLE", 0.9, 2);
    const r2 = s.update("EXAMPLE", 0.9, 3);
    expect(r2.status).toBe("committed");
    if (r2.status === "committed") expect(r2.event.label).toBe("EXAMPLE");
  });

  it("an uncertain blip in the middle of a held sign resets stability (documented behavior)", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.7, stableCount: 3 });
    s.update("QUESTION", 0.9, 0);
    s.update("QUESTION", 0.9, 1);
    s.update("QUESTION", 0.3, 2); // uncertain blip
    // stability must restart from here
    expect(s.update("QUESTION", 0.9, 3).status).toBe("pending");
    expect(s.update("QUESTION", 0.9, 4).status).toBe("pending");
    expect(s.update("QUESTION", 0.9, 5).status).toBe("committed");
  });

  it("default config matches the spec's 0.70 confidence threshold", () => {
    expect(DEFAULT_SMOOTHING.acceptThreshold).toBe(0.7);
  });
});
