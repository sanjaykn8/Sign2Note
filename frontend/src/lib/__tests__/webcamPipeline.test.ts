import { describe, it, expect } from "vitest";
import {
  KeypointBuffer,
  SessionSmoother,
  softmaxArgmax,
  DEFAULT_SMOOTHING,
} from "../webcamPipeline";

// NOTE: feature-vector construction (buildFeatureVector) and normalization
// (normalizeSequence) used to be tested here directly, back when they
// lived in this file as hand-only (126-dim) functions. They've since
// moved to featureSchema.ts as part of the hand+body+face schema v2
// migration -- see featureSchema.test.ts for their unit tests, and
// featureSchema.crosscheck.test.ts for the cross-language golden-vector
// test against ml_service/feature_schema.py.

describe("KeypointBuffer", () => {
  it("returns null when empty", () => {
    const buf = new KeypointBuffer(4, 2);
    expect(buf.getNormalizedWindow()).toBeNull();
  });

  it("defaults to FEATURE_DIM (285, hand+body+face) when dims isn't specified", () => {
    const buf = new KeypointBuffer(3);
    buf.push(new Float32Array(285).fill(1));
    const win = buf.getNormalizedWindow()!;
    expect(win.length).toBe(3 * 285);
  });

  it("zero-pads at the END when fewer than maxLen frames have been pushed", () => {
    const buf = new KeypointBuffer(4, 2);
    buf.push(new Float32Array([1, 1]));
    buf.push(new Float32Array([1, 1]));
    const win = buf.getNormalizedWindow()!;
    expect(win.length).toBe(4 * 2);
    // last two frames (padding) get normalized along with the real ones --
    // just check shape/finiteness here; exact normalization values are
    // covered by featureSchema.test.ts's normalizeSequence tests.
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
  it("reports 'no_sign' below the ignore threshold and does not commit", () => {
    const s = new SessionSmoother({ ignoreThreshold: 0.5, acceptThreshold: 0.75, stableCount: 3 });
    const r = s.update("QUESTION", 0.3, 1.0);
    expect(r.status).toBe("no_sign");
  });

  it("reports 'uncertain' in the middle zone (between ignore and accept thresholds) and does not commit", () => {
    const s = new SessionSmoother({ ignoreThreshold: 0.5, acceptThreshold: 0.75, stableCount: 3 });
    const r = s.update("QUESTION", 0.6, 1.0);
    expect(r.status).toBe("uncertain");
  });

  it("does not commit until stableCount consecutive agreeing confident predictions arrive", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.75, stableCount: 3 });
    expect(s.update("QUESTION", 0.9, 0).status).toBe("pending");
    expect(s.update("QUESTION", 0.85, 1).status).toBe("pending");
    const r3 = s.update("QUESTION", 0.88, 2);
    expect(r3.status).toBe("committed");
    if (r3.status === "committed") {
      expect(r3.event.label).toBe("QUESTION");
    }
  });

  it("collapses a long run of the same held sign into exactly ONE committed event", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.75, stableCount: 3 });
    const results = [];
    for (let i = 0; i < 10; i++) results.push(s.update("QUESTION", 0.9, i));
    const committed = results.filter((r) => r.status === "committed");
    expect(committed.length).toBe(1); // NOT ten QUESTION events
  });

  it("resets the stability counter when the raw label flickers, then commits once it settles", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.75, stableCount: 3 });
    s.update("DEFINITION", 0.9, 0);
    s.update("DEFINITION", 0.9, 1);
    s.update("EXAMPLE", 0.9, 2); // flicker resets the DEFINITION run
    s.update("DEFINITION", 0.9, 3);
    s.update("DEFINITION", 0.9, 4);
    const r = s.update("DEFINITION", 0.9, 5);
    expect(r.status).toBe("committed");
  });

  it("commits a new sign after a different sign has already been committed", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.75, stableCount: 2 });
    s.update("DEFINITION", 0.9, 0);
    const r1 = s.update("DEFINITION", 0.9, 1);
    expect(r1.status).toBe("committed");

    s.update("EXAMPLE", 0.9, 2);
    const r2 = s.update("EXAMPLE", 0.9, 3);
    expect(r2.status).toBe("committed");
    if (r2.status === "committed") expect(r2.event.label).toBe("EXAMPLE");
  });

  it("a NO_SIGN blip in the middle of a held sign resets stability (documented behavior)", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.75, stableCount: 3 });
    s.update("QUESTION", 0.9, 0);
    s.update("QUESTION", 0.9, 1);
    s.update("QUESTION", 0.2, 2); // drops into NO_SIGN zone
    // stability must restart from here
    expect(s.update("QUESTION", 0.9, 3).status).toBe("pending");
    expect(s.update("QUESTION", 0.9, 4).status).toBe("pending");
    expect(s.update("QUESTION", 0.9, 5).status).toBe("committed");
  });

  it("an UNCERTAIN blip (mid-zone) does NOT reset stability -- only NO_SIGN does", () => {
    const s = new SessionSmoother({ ignoreThreshold: 0.5, acceptThreshold: 0.75, stableCount: 3 });
    s.update("QUESTION", 0.9, 0);
    s.update("QUESTION", 0.9, 1);
    expect(s.update("QUESTION", 0.6, 2).status).toBe("uncertain"); // mid-zone blip, streak untouched
    const r = s.update("QUESTION", 0.9, 3); // 3rd confident QUESTION in the still-live streak
    expect(r.status).toBe("committed");
  });

  it("forgetLastCommitted lets an undone/deleted label be committed again", () => {
    const s = new SessionSmoother({ acceptThreshold: 0.75, stableCount: 2 });
    s.update("QUESTION", 0.9, 0);
    const r1 = s.update("QUESTION", 0.9, 1);
    expect(r1.status).toBe("committed");
    // without forgetting, repeating QUESTION stays "pending" forever
    expect(s.update("QUESTION", 0.9, 2).status).toBe("pending");

    s.forgetLastCommitted(null); // simulate the user undoing that event
    // the sign is still being held (already stable), so it re-commits on
    // the very next tick once the smoother has "forgotten" it was logged
    const r2 = s.update("QUESTION", 0.9, 3);
    expect(r2.status).toBe("committed");
  });

  it("default config matches the spec's three confidence zones (0.50 / 0.75)", () => {
    expect(DEFAULT_SMOOTHING.ignoreThreshold).toBe(0.5);
    expect(DEFAULT_SMOOTHING.acceptThreshold).toBe(0.75);
  });
});
