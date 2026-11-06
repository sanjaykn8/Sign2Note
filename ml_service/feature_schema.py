"""
Canonical feature schema v2: "hand_body_face".

This module is the ONE place that defines how a MediaPipe HolisticLandmarker
result becomes a flat per-frame feature vector, and how a sequence of those
vectors is normalized. Every other module that touches features
(feature_extraction.py, dataset.py, infer.py, api.py, train.py) imports the
constants and functions here instead of re-deriving them -- that is the fix
for the "one ordering in Python, a different one in TypeScript" risk called
out in the project brief.

The browser has its own copy, frontend/src/lib/featureSchema.ts, because a
Python module can't run in a browser. The two are kept in lockstep by:
  1. identical constants (landmark index lists, dims, ordering) -- see
     FEATURE_SCHEMA.md for the side-by-side table, and
  2. a cross-language golden-vector test: the same synthetic landmark input
     is fed to both this module and the TS module, and the two output
     vectors are asserted numerically identical (see
     ml_service/tests/test_feature_schema.py and
     frontend/src/lib/__tests__/featureSchema.crosscheck.json, which this
     module's test suite GENERATES and the TS suite CONSUMES).

Full history: schema v1 ("hand_only") was 2 hands x 21 landmarks x (x,y,z)
= 126 dims, built from mp.solutions.hands (see git history / ARCHITECTURE.md
"What Was Changed" for that era). v1 is retired, not extended -- v1 feature
files are numerically incompatible with v2 and must never be silently mixed
in (see assert_feature_dim() and FeatureSchemaError below).
"""

from __future__ import annotations

import numpy as np

FEATURE_SCHEMA_VERSION = "2.0"
FEATURE_TYPE = "hand_body_face"

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
# A single MediaPipe Tasks HolisticLandmarker call produces all four groups
# below in one pass, from one model, in one coordinate space. This is a
# deliberate choice over running separate Hands/Pose/FaceMesh detectors:
# Holistic already resolves left/right hand identity from its own body
# tracking (no "first detected = left hand" guessing), and guarantees all
# four groups share one frame's coordinate frame. See FEATURE_SCHEMA.md.
#
#   Python:  mediapipe.tasks.python.vision.HolisticLandmarker
#   Browser: @mediapipe/tasks-vision HolisticLandmarker (same task, same
#            model asset family, confirmed present in the pinned npm
#            version -- see FEATURE_SCHEMA.md "Verified against the
#            installed SDKs" section)

# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------
# All landmarks are MediaPipe "normalized" landmarks: x, y in [0, 1] relative
# to the image frame (NOT raw pixels -- already scale-appropriate for a
# frame of any resolution), z roughly on the same scale as x, with the
# origin MediaPipe defines internally per task. We do NOT use the
# "world landmarks" (metric, camera-relative) variants Holistic also
# exposes -- normalized landmarks are what schema v1 already used for
# hands, so keeping them for all four groups preserves that working choice
# instead of changing coordinate systems for no evidenced reason (see
# project brief: "do not destroy working code merely to make it look
# cleaner").

# ---------------------------------------------------------------------------
# LEFT_HAND / RIGHT_HAND -- 21 landmarks x (x, y, z) = 63 dims each
# ---------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21
HAND_DIMS_PER_LANDMARK = 3  # x, y, z (no visibility -- MediaPipe hand
                            # landmarks don't populate it meaningfully)
HAND_FEATURE_DIM = NUM_HAND_LANDMARKS * HAND_DIMS_PER_LANDMARK  # 63

# ---------------------------------------------------------------------------
# POSE -- 9 selected landmarks x (x, y, z, visibility) = 36 dims
# ---------------------------------------------------------------------------
# Indices are from MediaPipe Pose's standard 33-point topology (this
# numbering is part of the published Pose model contract, stable across
# mediapipe versions -- see FEATURE_SCHEMA.md for the citation).
#
# We deliberately use a REDUCED set, not the full 33: legs/feet (25-32) and
# the pose model's own low-resolution hand/finger points (17-22, which
# duplicate what the 21-point hand landmarker already gives us at far
# higher fidelity) carry no signal for seated/standing sign language and
# would just add dimensions and noise. What's kept covers exactly what the
# brief asked for -- shoulders, elbows, wrists -- plus nose and hips as
# head/torso reference points for future normalization work.
POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]
POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_DIMS_PER_LANDMARK = 4  # x, y, z, visibility
POSE_FEATURE_DIM = len(POSE_LANDMARK_INDICES) * POSE_DIMS_PER_LANDMARK  # 36

# ---------------------------------------------------------------------------
# FACE -- 41 selected landmarks x (x, y, z) = 123 dims
# ---------------------------------------------------------------------------
# The full FaceMesh topology is 468 points (1,404 dims at x,y,z) -- per the
# project brief, that would let face dimensions swamp the other three
# groups combined (63+63+36=162) for no accuracy benefit, since most of
# those 468 points are dense cheek/forehead tessellation with no expressive
# content. Instead we take a curated subset covering exactly the regions
# that carry ISL-relevant non-manual signal (eyebrow raises for question
# marking, eye widening, mouth/lip shape) plus head-pose reference points,
# using the community-standard FaceMesh landmark indices (these are fixed
# properties of the published 468-point face topology, not something we
# invented -- see FEATURE_SCHEMA.md for the per-region citation table).
# No visibility dim for face: FaceMesh doesn't populate it.
FACE_LANDMARK_GROUPS = {
    # Head-pose / scale reference points.
    "HEAD_REFERENCE": {
        "NOSE_TIP": 1,
        "NOSE_BRIDGE": 168,
        "NOSE_BOTTOM": 2,
        "FOREHEAD": 10,
        "CHIN": 152,
        "LEFT_CHEEK": 234,
        "RIGHT_CHEEK": 454,
    },
    # Eyebrow raise/furrow -- grammatically significant in ISL (e.g.
    # yes/no vs wh-question marking uses eyebrow position).
    "EYEBROWS": {
        "LEFT_EYEBROW_OUTER": 70,
        "LEFT_EYEBROW_MID_OUTER": 63,
        "LEFT_EYEBROW_MID": 105,
        "LEFT_EYEBROW_MID_INNER": 66,
        "LEFT_EYEBROW_INNER": 107,
        "RIGHT_EYEBROW_INNER": 336,
        "RIGHT_EYEBROW_MID_INNER": 296,
        "RIGHT_EYEBROW_MID": 334,
        "RIGHT_EYEBROW_MID_OUTER": 293,
        "RIGHT_EYEBROW_OUTER": 300,
    },
    # Eye openness / gaze direction.
    "EYES": {
        "LEFT_EYE_OUTER": 33,
        "LEFT_EYE_TOP_OUTER": 160,
        "LEFT_EYE_TOP_INNER": 158,
        "LEFT_EYE_INNER": 133,
        "LEFT_EYE_BOTTOM_INNER": 153,
        "LEFT_EYE_BOTTOM_OUTER": 144,
        "RIGHT_EYE_INNER": 362,
        "RIGHT_EYE_TOP_INNER": 385,
        "RIGHT_EYE_TOP_OUTER": 387,
        "RIGHT_EYE_OUTER": 263,
        "RIGHT_EYE_BOTTOM_OUTER": 373,
        "RIGHT_EYE_BOTTOM_INNER": 380,
    },
    # Mouth shape / mouthing -- many ISL signs carry an accompanying mouth
    # pattern, and lip aperture is part of non-manual grammar.
    "LIPS": {
        "MOUTH_LEFT_CORNER": 61,
        "UPPER_LIP_LEFT_OUTER": 40,
        "UPPER_LIP_LEFT_INNER": 37,
        "UPPER_LIP_CENTER": 0,
        "UPPER_LIP_RIGHT_INNER": 267,
        "UPPER_LIP_RIGHT_OUTER": 270,
        "MOUTH_RIGHT_CORNER": 291,
        "LOWER_LIP_RIGHT_OUTER": 321,
        "LOWER_LIP_RIGHT_INNER": 314,
        "LOWER_LIP_CENTER": 17,
        "LOWER_LIP_LEFT_INNER": 84,
        "LOWER_LIP_LEFT_OUTER": 91,
    },
}

# Flattened, in a fixed deterministic order (dict insertion order in Python
# 3.7+ is guaranteed, and the literal above is written in the exact order
# we want -- HEAD_REFERENCE, EYEBROWS, EYES, LIPS). This flattened list is
# what actually gets used; the grouped dict above exists purely for
# documentation and is asserted to stay in sync with it in tests.
FACE_LANDMARK_NAMES: list[str] = [
    name for group in FACE_LANDMARK_GROUPS.values() for name in group
]
FACE_LANDMARK_INDICES: list[int] = [
    idx for group in FACE_LANDMARK_GROUPS.values() for idx in group.values()
]
FACE_DIMS_PER_LANDMARK = 3  # x, y, z
FACE_FEATURE_DIM = len(FACE_LANDMARK_INDICES) * FACE_DIMS_PER_LANDMARK  # 123

# ---------------------------------------------------------------------------
# Full per-frame layout
# ---------------------------------------------------------------------------
# Concatenation order is fixed and must never change without bumping
# FEATURE_SCHEMA_VERSION: LEFT_HAND, RIGHT_HAND, POSE, FACE.
FEATURE_GROUPS = [
    ("left_hand", 0, HAND_FEATURE_DIM),
    ("right_hand", HAND_FEATURE_DIM, 2 * HAND_FEATURE_DIM),
    ("pose", 2 * HAND_FEATURE_DIM, 2 * HAND_FEATURE_DIM + POSE_FEATURE_DIM),
    (
        "face",
        2 * HAND_FEATURE_DIM + POSE_FEATURE_DIM,
        2 * HAND_FEATURE_DIM + POSE_FEATURE_DIM + FACE_FEATURE_DIM,
    ),
]
FEATURE_DIM = 2 * HAND_FEATURE_DIM + POSE_FEATURE_DIM + FACE_FEATURE_DIM  # 285

# Absolute indices (within the 285-dim vector) of the 9 pose-visibility
# values -- x,y,z,visibility per pose landmark, visibility is the 4th.
# Used by normalize_sequence() to exclude these from z-scoring (see
# "Normalization" below for why).
_POSE_START = FEATURE_GROUPS[2][1]
POSE_VISIBILITY_INDICES: list[int] = [
    _POSE_START + 4 * i + 3 for i in range(len(POSE_LANDMARK_INDICES))
]


class FeatureSchemaError(ValueError):
    """Raised when loaded/received features don't match the current schema
    (wrong dimension, or a version tag that doesn't match). This is meant
    to fail LOUDLY and EARLY -- see project brief section 14/66: never
    silently mix old and new feature representations."""


# ---------------------------------------------------------------------------
# Legacy schema v1 ("hand_only") -- kept ONLY so the previously trained
# hand-only model (models/sign_recog, 126 dims) stays usable for rollback
# and comparison. It is a separate, explicitly-versioned schema: a v1 model
# is only ever fed v1 vectors and a v2 model only v2 vectors (see
# SCHEMA_DIMS / assert_feature_dim(expected_dim=...)). It is never mixed
# into v2 training data.
#
# v1 layout (reproduced from the original feature_extraction.py in git
# history, commit b6a11c7): two 21x(x,y,z) hand slots = 126 dims, filled in
# MediaPipe's DETECTION order (slot 0 = first detected hand, slot 1 =
# second), NOT by handedness. That order-dependence is a known weakness of
# v1 and one of the reasons v2 exists.
# ---------------------------------------------------------------------------
LEGACY_SCHEMA_VERSION = "1.0"
LEGACY_FEATURE_TYPE = "hand_only"
LEGACY_FEATURE_DIM = 2 * HAND_FEATURE_DIM  # 126

SCHEMA_DIMS = {
    LEGACY_SCHEMA_VERSION: LEGACY_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION: FEATURE_DIM,
}
SCHEMA_TYPES = {
    LEGACY_SCHEMA_VERSION: LEGACY_FEATURE_TYPE,
    FEATURE_SCHEMA_VERSION: FEATURE_TYPE,
}


def feature_dim_for_schema(schema_version: str) -> int:
    """Feature dimension of a known schema version; raises
    FeatureSchemaError for an unknown version instead of guessing."""
    try:
        return SCHEMA_DIMS[str(schema_version)]
    except KeyError:
        raise FeatureSchemaError(
            f"Unknown feature_schema_version {schema_version!r}; known versions: "
            f"{sorted(SCHEMA_DIMS)}"
        ) from None


def build_legacy_v1_vector(hands_in_detection_order) -> np.ndarray:
    """Schema v1 (126-dim) vector from up to two hand landmark lists given
    in detection order. Missing hands -> zeros. Mirrors the original v1
    `_extract_keypoints` exactly (slot 0 = first hand, slot 1 = second)."""
    hands = [h for h in (hands_in_detection_order or []) if h][:2]
    out = np.zeros(LEGACY_FEATURE_DIM, dtype=np.float32)
    for slot, hand in enumerate(hands):
        out[slot * HAND_FEATURE_DIM:(slot + 1) * HAND_FEATURE_DIM] = _flatten_hand(hand)
    return out


def legacy_v1_from_v2(vec_or_seq: np.ndarray) -> np.ndarray:
    """Approximate v1 vector(s) from v2 vector(s): the detected hands are
    packed into the v1 slots in (left, right) order, skipping a missing
    hand -- so a single visible hand always lands in slot 0, which is what
    v1's detection-order layout produced for one-handed signing. With two
    hands the v1 order was arbitrary (whichever MediaPipe found first), so
    this is an approximation, documented in FEATURE_SCHEMA.md."""
    x = np.asarray(vec_or_seq, dtype=np.float32)
    single = x.ndim == 1
    x2 = x[None, :] if single else x
    assert_feature_dim(x2.shape[-1], context="legacy_v1_from_v2 input")
    out = np.zeros((len(x2), LEGACY_FEATURE_DIM), dtype=np.float32)
    for i, row in enumerate(x2):
        left = row[0:HAND_FEATURE_DIM]
        right = row[HAND_FEATURE_DIM:2 * HAND_FEATURE_DIM]
        present = [h for h in (left, right) if np.any(h)]
        for slot, h in enumerate(present):
            out[i, slot * HAND_FEATURE_DIM:(slot + 1) * HAND_FEATURE_DIM] = h
    return out[0] if single else out


def assert_feature_dim(dim: int, context: str = "", expected_dim: int | None = None) -> None:
    """Fail clearly (not with a downstream shape-mismatch crash three
    layers away) when a loaded feature array doesn't match the expected
    schema's dimension. `expected_dim` defaults to the current schema v2
    (FEATURE_DIM); pass a model's own input_dim to validate against the
    schema that model was trained on. `context` should be something like a
    file path or 'checkpoint' so the error is actionable."""
    expected = FEATURE_DIM if expected_dim is None else int(expected_dim)
    if dim != expected:
        where = f" ({context})" if context else ""
        hint = ""
        if dim == LEGACY_FEATURE_DIM:
            hint = " These look like old hand-only features (schema v1, 126 dims)."
        elif dim == FEATURE_DIM:
            hint = " These look like hand+body+face features (schema v2, 285 dims)."
        raise FeatureSchemaError(
            f"Feature dimension mismatch{where}: got {dim}, expected {expected}. "
            f"Schema v{FEATURE_SCHEMA_VERSION} ('{FEATURE_TYPE}') is {FEATURE_DIM} dims "
            f"({HAND_FEATURE_DIM} left hand + {HAND_FEATURE_DIM} right hand + "
            f"{POSE_FEATURE_DIM} pose + {FACE_FEATURE_DIM} face); legacy schema "
            f"v{LEGACY_SCHEMA_VERSION} ('{LEGACY_FEATURE_TYPE}') is {LEGACY_FEATURE_DIM} dims."
            f"{hint} Features and model must use the same schema (see FEATURE_SCHEMA.md)."
        )


# ---------------------------------------------------------------------------
# Missing-landmark behavior
# ---------------------------------------------------------------------------
# If a landmark group isn't detected in a frame (hand out of frame, person
# fully occluded, etc.), that group's slice is a deterministic all-zero
# vector -- never NaN, never a different length, never left out of the
# concatenation. This keeps every frame's feature vector exactly
# FEATURE_DIM long regardless of what was or wasn't detected, which is
# required for both training (fixed-shape tensors) and normalization
# (see below).


def _flatten_landmarks(
    landmarks,
    indices: list[int] | None,
    dims_per_landmark: int,
    include_visibility: bool,
) -> np.ndarray:
    """Flatten a list of landmark-like objects (must expose .x/.y/.z, and
    .visibility if include_visibility) into a flat float32 vector, in the
    exact order given by `indices` (or natural order if indices is None).
    Missing/out-of-range landmarks -> zeros for that landmark's slice.
    `landmarks` may be None or empty (not detected) -> an all-zero vector
    of the correct total length.
    """
    order = indices if indices is not None else range(dims_per_landmark and 0)
    n = len(indices) if indices is not None else 0
    out = np.zeros(n * dims_per_landmark, dtype=np.float32)
    if not landmarks:
        return out
    for slot, src_idx in enumerate(order):
        if src_idx >= len(landmarks):
            continue  # defensive: leave this landmark's slice as zeros
        lm = landmarks[src_idx]
        base = slot * dims_per_landmark
        out[base + 0] = lm.x if lm.x is not None else 0.0
        out[base + 1] = lm.y if lm.y is not None else 0.0
        out[base + 2] = lm.z if lm.z is not None else 0.0
        if include_visibility:
            vis = getattr(lm, "visibility", None)
            out[base + 3] = vis if vis is not None else 0.0
    return out


def _flatten_hand(landmarks) -> np.ndarray:
    """All 21 hand landmarks, in MediaPipe's native order (0=wrist ...
    20=pinky tip) -- no subsetting, unlike pose/face."""
    out = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    if not landmarks:
        return out
    for i, lm in enumerate(landmarks[:NUM_HAND_LANDMARKS]):
        base = i * HAND_DIMS_PER_LANDMARK
        out[base + 0] = lm.x if lm.x is not None else 0.0
        out[base + 1] = lm.y if lm.y is not None else 0.0
        out[base + 2] = lm.z if lm.z is not None else 0.0
    return out


def build_feature_vector(
    left_hand_landmarks,
    right_hand_landmarks,
    pose_landmarks,
    face_landmarks,
) -> np.ndarray:
    """Build one canonical FEATURE_DIM-length vector for a single frame from
    a HolisticLandmarker result's four landmark lists. Each argument is
    either None/empty (not detected -> zeros) or a list of landmark-like
    objects exposing .x/.y/.z (and .visibility for pose).

    This function takes plain landmark lists rather than a
    HolisticLandmarkerResult object so it works identically whether the
    caller is live Python HolisticLandmarker output (result.left_hand_landmarks
    etc, already flat lists) or a replayed/synthetic fixture in tests.
    """
    left = _flatten_hand(left_hand_landmarks)
    right = _flatten_hand(right_hand_landmarks)
    pose = _flatten_landmarks(
        pose_landmarks, POSE_LANDMARK_INDICES, POSE_DIMS_PER_LANDMARK, include_visibility=True
    )
    face = _flatten_landmarks(
        face_landmarks, FACE_LANDMARK_INDICES, FACE_DIMS_PER_LANDMARK, include_visibility=False
    )
    vec = np.concatenate([left, right, pose, face]).astype(np.float32)
    assert vec.shape[0] == FEATURE_DIM, (
        f"internal error: built a {vec.shape[0]}-dim vector, expected {FEATURE_DIM}"
    )
    return vec


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
# Per-clip/per-window z-score, per dimension, over time (axis=0) -- this is
# schema v1's existing normalization, PRESERVED here rather than replaced,
# because nothing in the v1->v2 audit showed it was broken for coordinate
# dimensions: z-scoring per-dimension is already scale- and translation-
# invariant per coordinate channel regardless of which landmark group that
# channel belongs to, so hand/pose/face coordinates don't need a separate
# spatial re-referencing step on top of it.
#
# A second thing checked while consolidating this (not a bug fix -- the
# existing behavior was already correct and already consistent between
# dataset.py and infer.py, verified by reading both directly): short
# clips/windows (fewer real frames than max_len) are zero-padded to
# max_len FIRST, and the per-dimension mean/std is computed over the
# padded array, padding included -- both dataset.py's SignDataset and
# infer.py's _window_batch already did this, in this order, deliberately
# (infer.py's own docstring says so). Long clips/windows never need
# padding in either file (a window is already exactly max_len real
# frames), so the pad step is a no-op there and order doesn't matter.
# prepare_window() below just gives this one shared implementation
# instead of two independently-written, easy-to-drift copies of the same
# two-line pattern.


def normalize_sequence(x: np.ndarray) -> np.ndarray:
    """Per-dimension z-score over time (axis=0), excluding pose-visibility
    dims (passed through, clipped to [0, 1] instead). `x` is
    (num_frames, FEATURE_DIM) or (num_frames, any_dim) -- dimension is not
    asserted here so this can also be unit-tested against toy vector sizes;
    callers that care about schema compatibility should call
    assert_feature_dim() themselves first.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-5
    z = (x - mean) / std

    if x.shape[1] == FEATURE_DIM:
        for idx in POSE_VISIBILITY_INDICES:
            z[:, idx] = np.clip(x[:, idx], 0.0, 1.0)
    return z


def pad_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    """Trim to the first max_len frames, or zero-pad at the end to reach
    max_len. Frame dimension (x.shape[1]) is preserved as-is."""
    if len(x) >= max_len:
        return x[:max_len]
    pad = np.zeros((max_len - len(x), x.shape[1]), dtype=np.float32)
    return np.vstack([x, pad])


def prepare_window(x: np.ndarray, max_len: int) -> np.ndarray:
    """The ONE canonical pad/trim -> normalize pipeline for turning a raw
    (num_real_frames, FEATURE_DIM) sequence into a fixed (max_len,
    FEATURE_DIM) model input: pad_or_trim() first, normalize_sequence()
    second -- matching what dataset.py (training) and infer.py
    (inference) already both did independently. Used by both so they
    can't independently drift apart in the future; does not change either
    one's numeric behavior today.
    """
    return normalize_sequence(pad_or_trim(x, max_len))
