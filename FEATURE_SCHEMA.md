# Feature Schema (v2 `hand_body_face`, legacy v1 `hand_only`)

This is the canonical, single-source-of-truth description of what a
"feature vector" is in Sign2Notes. It exists because the project runs the
same representation in three places (Python training, Python backend
inference, and browser live inference) and those three **must** agree
bit-for-bit on layout — a mismatch here doesn't crash, it just silently
produces garbage predictions.

The two implementations are:

| | File |
|---|---|
| Python (training + server inference) | `ml_service/feature_schema.py` |
| Browser (live webcam) | `frontend/src/lib/featureSchema.ts` |

They are proven identical by a **cross-language golden-vector test**, not
just code review: `ml_service/generate_crosscheck_fixture.py` builds a
feature vector from a fixed synthetic landmark input using the Python
module (source of truth) and writes the input + expected output to
`frontend/src/lib/__tests__/crosscheckFixture.json`.
`frontend/src/lib/__tests__/featureSchema.crosscheck.test.ts` then feeds
that same input through the TypeScript module and asserts the output
matches to 4 decimal places. **If you change a landmark index, a
dimension, or the normalization formula in one file, you must regenerate
the fixture (`cd ml_service && python3 generate_crosscheck_fixture.py`)
and re-run the frontend test** — if it fails, the two schemas drifted.

## Schema versions

| Version | Type | Dims | Status |
|---|---|---|---|
| `2.0` | `hand_body_face` | 285 | current; everything new is trained on it |
| `1.0` | `hand_only` | 126 | legacy; only to serve/compare the old model in `models/sign_recog` |

**v1 layout** (reproduced from the original extractor, git `b6a11c7`): two
21×(x,y,z) hand slots filled in MediaPipe **detection order** (slot 0 =
first detected hand), not by handedness. `feature_schema.LEGACY_*`,
`build_legacy_v1_vector()`; Python extraction for v1 uses the original
`mp.solutions.hands` settings (model_complexity 0, max 2 hands).

v1 and v2 are never mixed: every checkpoint records
`feature_schema_version`; the API extracts uploads in the served model's
schema and rejects keypoints of another dimension; feature directories
carry `feature_meta.json`; `assert_feature_dim(..., expected_dim=...)`
fails loudly on a mismatch.

**Browser → v1** (`legacyV1FromV2` / `legacy_v1_from_v2`): the browser
only runs Holistic, so for a v1 model it packs the detected hands into the
v1 slots in (left, right) order, skipping a missing hand. One visible hand
→ slot 0, which is what v1 produced for one-handed signing; with two hands
v1's order was arbitrary, so this is an approximation.

The 83.5 % figure quoted in older docs belongs to the v1 model trained on
a random split of all clips; it is not a v2 or official-split result.

## Detector

Python and browser run the same MediaPipe Holistic graph, through
different APIs:

| | API | Why |
|---|---|---|
| Python (training, uploads) | `mediapipe.solutions.holistic` (classic), `model_complexity=1`, `refine_face_landmarks=False`, tracking per video | The Tasks `HolisticLandmarker` in mediapipe 0.10.21 hard-crashes the process on FDMSE-ISL clips (`packet.cc ... The packet is empty`), reproduced in VIDEO and IMAGE mode. Selectable with `HOLISTIC_BACKEND=tasks`. |
| Browser (live modes) | `@mediapipe/tasks-vision` `HolisticLandmarker` | No such crash; runs in WASM/GPU |

Both resolve left/right hands from the body pose (no detection-order
guessing) and use the same landmark numbering (21 per hand, 33 pose,
468 face — Tasks adds 10 iris points at 468–477, which the schema does
not use). Numeric outputs of the two implementations are close but not
identical — a documented train/serve approximation for the live modes.
Uploads are extracted with the same backend as training. The backend used
is recorded in each feature directory's `feature_meta.json`
(`mediapipe_api`).

## Coordinate system

All landmarks are MediaPipe **normalized landmarks**: `x`, `y` ∈ [0, 1]
relative to the image frame (not raw pixels), `z` roughly on the same
scale. This is schema v1's existing choice for hands, kept for pose and
face too rather than switched to "world" (metric) landmarks, since v1's
choice was already working and nothing in this audit showed a concrete
reason to change coordinate systems.

## Per-frame layout (285 dims total)

Concatenation order is fixed: **LEFT_HAND → RIGHT_HAND → POSE → FACE**.
Changing this order, or any index list below, requires bumping
`FEATURE_SCHEMA_VERSION`.

| Group | Landmarks | Dims/landmark | Group dims | Offset |
|---|---:|---:|---:|---|
| `left_hand` | 21 | 3 (x,y,z) | 63 | 0–62 |
| `right_hand` | 21 | 3 (x,y,z) | 63 | 63–125 |
| `pose` | 9 (selected) | 4 (x,y,z,visibility) | 36 | 126–161 |
| `face` | 41 (selected) | 3 (x,y,z) | 123 | 162–284 |

**FEATURE_DIM = 285**

### Hands (63 dims each)

All 21 MediaPipe hand landmarks (0=wrist … 20=pinky tip), in native
MediaPipe order — no subsetting, x/y/z only (hand landmarks don't
populate `visibility` meaningfully).

### Pose (36 dims) — a reduced, not full, 33-point set

Indices are from MediaPipe Pose's standard 33-point topology (a fixed,
published numbering, stable across mediapipe versions):

| Name | Pose index |
|---|---:|
| NOSE | 0 |
| LEFT_SHOULDER | 11 |
| RIGHT_SHOULDER | 12 |
| LEFT_ELBOW | 13 |
| RIGHT_ELBOW | 14 |
| LEFT_WRIST | 15 |
| RIGHT_WRIST | 16 |
| LEFT_HIP | 23 |
| RIGHT_HIP | 24 |

Legs/feet (indices 25–32) and the pose model's own low-resolution
finger points (17–22, which duplicate the 21-point hand landmarker at
far lower fidelity) are deliberately excluded — no signal for seated/
standing signing, just extra dims and noise. `visibility` is included
per landmark (the 4th value) since Pose populates it meaningfully and
it's a useful occlusion signal — see "Normalization" below for why it's
handled differently from coordinates.

### Face (123 dims) — a curated 41-point subset, not the full 468-point mesh

The brief explicitly warned against "randomly concatenating thousands of
raw face coordinates" — the full mesh (468 × 3 = 1,404 dims) would let
face alone outweigh hands+pose combined (162 dims) for no accuracy
benefit, since most of those 468 points are dense cheek/forehead
tessellation with no expressive content for sign language. The 41 chosen
points cover exactly the regions with ISL-relevant non-manual signal,
using MediaPipe FaceMesh's standard (fixed, published) 468-point index
numbering:

| Group | Count | Purpose |
|---|---:|---|
| Head reference | 7 | nose tip/bridge/bottom, forehead, chin, both cheeks — head pose & scale reference |
| Eyebrows | 10 | eyebrow raise/furrow — grammatically significant in ISL (e.g. question marking) |
| Eyes | 12 | eye openness / gaze |
| Lips | 12 | mouth shape / mouthing — many ISL signs carry an accompanying mouth pattern |

Exact indices are in `FACE_LANDMARK_GROUPS` in both `feature_schema.py`
and `featureSchema.ts` (kept as named dicts, not just a bare index list,
specifically so a reader can see which index means what without
cross-referencing this doc). No `visibility` dim for face — FaceMesh
doesn't populate it.

## Missing-landmark behavior

If a group isn't detected in a frame (hand out of frame, person fully
occluded, face turned away, etc.), that group's slice is **deterministic
all-zero** — never `NaN`, never a shorter vector, never left out of the
concatenation. A missing left hand does **not** shift the right hand's
data into the left hand's slot; it stays zero-filled in place. This keeps
every frame exactly `FEATURE_DIM` long regardless of what was or wasn't
detected — required both for fixed-shape training tensors and for
per-dimension normalization to mean the same thing across frames.
Covered by `test_feature_schema.py`'s "both hands / one hand / no hands /
pose present / face present / out-of-range index" tests, and mirrored in
`featureSchema.ts`'s logic (defensive zero-fill on any missing or
out-of-range landmark, never a thrown error).

## Normalization

Per-clip/per-window **z-score, per dimension, over time** (mean/std
computed across the frames of that window, independently for each of the
285 dimensions) — this is schema v1's existing scheme, **preserved**, not
replaced: z-scoring per-dimension is already scale- and
translation-invariant per coordinate channel regardless of which landmark
group that channel belongs to, so there was no concrete evidence it broke
by adding pose/face on top of hands.

Two things worth calling out on top of that scheme:

1. **Pose visibility is excluded from z-scoring** (a real, evidence-based
   fix). Visibility is a confidence score already in [0, 1], not a
   coordinate. It's often near-constant (≈1.0) across a whole clip when a
   landmark stays visible the entire time, driving its temporal std toward
   zero — z-scoring a near-constant channel divides by `(~0 + 1e-5)`,
   amplifying tiny fluctuations into huge spurious values. Visibility is
   instead passed through as-is (clipped to [0, 1] defensively). See
   `test_normalize_sequence_does_not_zscore_pose_visibility`.
2. **Pad/trim-then-normalize is now one shared function, not two
   independent copies** (a consolidation, not a bug fix — checked both
   files directly before touching anything, per RULE 18). `dataset.py`
   (training) already padded/trimmed to `max_len` and then computed that
   sample's own per-clip mean/std; `infer.py`'s live `_window_batch`
   docstring says explicitly that its short-clip path does the same for
   exactly this reason, and its multi-window path for long clips never
   pads at all (a window is already exactly `max_len` real frames), so
   there was nothing to reconcile there. Both were already correct and
   already consistent with each other. `prepare_window(x, max_len)` —
   `pad_or_trim()` then `normalize_sequence()`, in that order — is now the
   one implementation both files call, so this can't independently drift
   apart in the future even though it doesn't change today's behavior.

(Earlier drafts of this document claimed a real train/inference
normalization-order mismatch existed here. On closer reading of both
files side by side, that was wrong — the harmless artifact actually
present was `infer.py` having two copies of `_window_batch` defined back
to back, the first one dead code silently shadowed by the second; that's
what's actually been cleaned up, not a numeric behavior change.)

## Handedness handling

Solved by the detector, not by post-processing (see "Detector" above) —
`HolisticLandmarker` reports which hand is which directly. There is no
"assume detection order = left-then-right" logic anywhere in v2.

## Frame sampling, padding, temporal window length

- Training: FDMSE-ISL clips are 60 fps; `frame_skip=8` → 7.5 samples/s;
  clips are ~20–35 samples, zero-padded at the end to `max_len=64`, then
  normalized over the padded array.
- Uploads: `frame_skip = round(fps / 7.5)` so any frame rate is sampled at
  the training rate; windows of 24 real samples (stride 4) padded to 64.
- Browser: one sample every `1000 / sample_hz` ms (133 ms); the buffer
  keeps the last 24 samples, padded to 64.

## Verified against the installed SDKs

- `mediapipe==0.10.21` (Python, the pinned version): provides both
  `mediapipe.solutions.holistic` / `.hands` and the Tasks
  `HolisticLandmarker`; the Tasks one crashes as described above. Later
  releases (e.g. 0.10.33) removed `mediapipe.solutions`.
- `@mediapipe/tasks-vision@1.0.1` (npm): `HolisticLandmarker` with
  `leftHandLandmarks`, `rightHandLandmarks`, `poseLandmarks`,
  `faceLandmarks`.
