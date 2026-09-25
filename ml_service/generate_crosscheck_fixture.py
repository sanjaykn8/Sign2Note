"""
Generates frontend/src/lib/__tests__/crosscheckFixture.json -- the fixture
that proves feature_schema.py and frontend/src/lib/featureSchema.ts produce
numerically identical output from the same synthetic landmark input.

Run this whenever feature_schema.py's landmark indices, dims, or
normalization logic change, then re-run the frontend cross-check test
(`npm run test -- featureSchema.crosscheck`) to confirm the TS twin was
updated to match. This script is NOT itself a test -- it regenerates the
fixture from the Python side, which is treated as the source of truth
(feature_schema.py's own tests, test_feature_schema.py, already cover it
independently with plain assertions, not by round-tripping this fixture).

Usage:
    cd ml_service && python3 generate_crosscheck_fixture.py
"""

import json
from pathlib import Path

import numpy as np

import feature_schema as fs


class FakeLM:
    def __init__(self, x, y, z, visibility=None):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def make_hand(seed: float):
    return [
        [seed + i * 0.01, seed + i * 0.01 + 0.001, seed + i * 0.01 + 0.002]
        for i in range(fs.NUM_HAND_LANDMARKS)
    ]


def make_full_pose():
    # 33 landmarks (full MediaPipe Pose topology), value derived from index
    # so the subsetting logic is exercised identically in both languages.
    return [[i * 0.02, i * 0.02 + 0.3, i * 0.02 - 0.1, min(1.0, 0.5 + i / 64)] for i in range(33)]


def make_full_face():
    # 468 landmarks (full FaceMesh topology), same idea.
    return [[i * 0.001, i * 0.001 + 0.4, i * 0.001 - 0.2] for i in range(468)]


def to_fake_landmarks(rows, with_visibility):
    if with_visibility:
        return [FakeLM(r[0], r[1], r[2], visibility=r[3]) for r in rows]
    return [FakeLM(r[0], r[1], r[2]) for r in rows]


def main():
    left_hand = make_hand(0.1)
    right_hand = make_hand(0.6)
    pose = make_full_pose()
    face = make_full_face()

    vec = fs.build_feature_vector(
        to_fake_landmarks(left_hand, False),
        to_fake_landmarks(right_hand, False),
        to_fake_landmarks(pose, True),
        to_fake_landmarks(face, False),
    )

    # A tiny synthetic 4-frame sequence (built by scaling the single frame
    # above) to exercise normalize_sequence()'s per-dimension z-score +
    # pose-visibility passthrough across time, not just a single frame.
    rng_scale = [0.8, 1.0, 1.2, 0.95]
    sequence = np.stack([vec * s for s in rng_scale]).astype(np.float32)
    normalized = fs.normalize_sequence(sequence)

    fixture = {
        "schema_version": fs.FEATURE_SCHEMA_VERSION,
        "feature_dim": fs.FEATURE_DIM,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pose": pose,
        "face": face,
        "expected_feature_vector": vec.tolist(),
        "sequence_frame_scales": rng_scale,
        "expected_normalized_sequence": normalized.tolist(),
    }

    out_path = Path(__file__).resolve().parent.parent / "frontend" / "src" / "lib" / "__tests__" / "crosscheckFixture.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
