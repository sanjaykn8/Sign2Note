# Feature Schema v2 — `hand_body_face`

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

## History

Schema v1 (`hand_only`) was 2 hands × 21 landmarks × (x,y,z) = **126
dims**, built from `mp.solutions.hands`. It's retired, not extended. v1
`.npy` feature files are **numerically incompatible** with v2 and must
never be silently mixed in — `assert_feature_dim()` /
`assertFeatureDim()` raise a clear, actionable error (naming the 126-dim
case specifically) instead of a downstream shape-mismatch crash three
layers away. See `ARCHITECTURE.md`'s changelog for v1's era; the 83.5%
validation-accuracy result referenced by earlier project docs was
measured against v1 (hand-only) features and should be labeled as such
wherever it's cited, not silently reused as if it applied to v2.

There's also a real reason `mp.solutions.hands` had to go regardless of
the schema change: it doesn't exist in current `mediapipe` releases —
`pip install mediapipe` today ships Tasks-API-only, no `mp.solutions`.
`requirements.txt`'s unpinned `mediapipe>=0.10.0` meant `feature_extraction.py`
would crash on import against a fresh install. Migrating to the Tasks API
(`HolisticLandmarker`) fixes this as a side effect of the v2 schema work.

## Detector: one `HolisticLandmarker` call, not three separate detectors

Both languages use MediaPipe's Tasks-API **`HolisticLandmarker`** —
confirmed present in both the installed Python `mediapipe` package and the
pinned npm package (`@mediapipe/tasks-vision@1.0.1`, checked against its
shipped `.d.ts`) — rather than running separate `HandLandmarker` +
`PoseLandmarker` + `FaceLandmarker` instances. Two concrete reasons:

1. **Handedness is solved by construction.** `HolisticLandmarker` reports
   `left_hand_landmarks` / `right_hand_landmarks` (Python) and
   `leftHandLandmarks` / `rightHandLandmarks` (JS) directly, resolved from
   its own body tracking. Schema v1 ordered hands by "whichever MediaPipe
   detected first," which is not the same thing as which hand is actually
   left or right — that ambiguity doesn't exist in v2 at all, because the
   API itself tells you.
2. **One coordinate frame.** Hand, pose, and face landmarks from one
   `HolisticLandmarker` call are guaranteed to come from the same frame's
   single detection pass, rather than three independent detectors
   potentially disagreeing about where the person is.

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

Unchanged from schema v1 — nothing in this audit showed a concrete reason
to change `frame_skip`, sliding-window stride, or `max_len` handling, so
they're preserved as-is (see `feature_extraction.py`'s CLI flags and
`ARCHITECTURE.md`'s inference-pipeline section for the current values).
Only the per-frame vector's *contents* changed (126 → 285 dims); the
*temporal* pipeline around it (windowing, Viterbi decoding for uploaded
video, stability gating for live webcam) is schema-agnostic and required
no changes.

## Verified against the installed SDKs, not assumed

Both of the following were checked against the actual installed
packages while writing this schema, not recalled from memory:

- `mediapipe==0.10.33` (Python): `mp.solutions` does not exist;
  `mediapipe.tasks.python.vision.HolisticLandmarker` does, and its result
  dataclass (`HolisticLandmarkerResult`) has `left_hand_landmarks`,
  `right_hand_landmarks`, `pose_landmarks`, `face_landmarks` fields
  (confirmed by reading `holistic_landmarker.py`'s source directly).
- `@mediapipe/tasks-vision@1.0.1` (npm, the version pinned in
  `frontend/package.json`): its shipped `vision.d.ts` declares
  `HolisticLandmarker`, `HolisticLandmarkerOptions`, and
  `HolisticLandmarkerResult` with `leftHandLandmarks`,
  `rightHandLandmarks`, `poseLandmarks`, `faceLandmarks` fields
  (confirmed by unpacking the actual npm tarball and grepping its type
  declarations).

Model asset (the `.task` bundle `HolisticLandmarker` loads) is fetched
from Google's model CDN at first use, same pattern as v1's
`hand_landmarker.task` — see `TROUBLESHOOTING.md` for the exact URL and
`ARCHITECTURE.md`'s "Model download / cache" section for what is and
isn't cached locally after that first fetch.
