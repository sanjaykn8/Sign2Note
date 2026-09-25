# Privacy

Sign2Notes is designed so that the sensitive data — your video and your
hands — never has to leave your machine. This document says exactly what
that does and does not mean, for the upload, live webcam, and live
transcription flows, because the honest answer is more precise than a
blanket "100% local."

## Video upload flow

1. The React frontend sends the video file to the Node gateway
   (`backend/server.js`), which holds it in memory only (never writes it to
   disk) and forwards it to the FastAPI ML service.
2. FastAPI (`ml_service/api.py`) writes the video to a temporary file only
   long enough to extract keypoints frame-by-frame, then:
   - deletes the temporary video file
   - deletes the extracted keypoints file (`.npy`)
   both in a `finally` block, so they're removed even if extraction or
   inference raises an exception.
3. Nothing about the video or its content is sent anywhere outside your
   machine (no third-party API, no analytics, no telemetry). If you've
   configured `notes_mode="llm"`, only the **recognized gloss words**
   (plain text like `"Market"` or `"Whistle"`) are sent to your **local**
   Ollama/llama.cpp server — never the video or keypoints, and never to a
   remote/cloud LLM unless you deliberately point `LLM_BASE_URL` at one.

## Live webcam flow

This is the stricter case, and the architecture reflects it: **all keypoint
extraction and ONNX inference for the webcam session run inside your
browser tab.** Concretely:

- `getUserMedia()` gives the page a live video stream; it is never
  recorded, never uploaded, and never touches `fetch`/`XMLHttpRequest`.
- MediaPipe's HolisticLandmarker (`@mediapipe/tasks-vision`, running via
  WASM) processes video frames locally to get hand, body-pose, and
  face-landmark coordinates in one pass — see `FEATURE_SCHEMA.md` for
  exactly which points and why.
- The ONNX model runs locally via `onnxruntime-web` (also WASM), on the
  keypoints extracted from those landmarks.
- Only when you click **Generate Notes** does anything leave the browser
  tab — and what leaves is the final list of recognized gloss words (e.g.
  `["DEFINITION", "EXAMPLE", "QUESTION"]`), sent to the backend's `/notes`
  endpoint. No video frame, image, or keypoint array is ever part of that
  request.

### What "local" does NOT mean here

To load the landmark-tracking WASM runtime and model file, and the ONNX
runtime's WASM binary, the browser does make a small number of one-time
network requests to public CDNs (jsDelivr for the WASM runtimes, Google's
model storage for the holistic-landmark model file) — the same way loading
any web page's JS/CSS or a compiled app's shared libraries involves network
requests. **These are static, generic runtime/model assets, not your
data** — no video, image, or keypoint ever appears in these requests, and
the browser caches them after the first load. If you need a fully
air-gapped setup with zero network requests after initial page load, see
"Self-hosting the runtime assets" below.

## Live Transcription flow

Same architecture as live webcam above — same landmark extraction, same
local ONNX inference, same "only recognized glosses leave the browser, and
only on Stop & Generate" rule (now sent to two endpoints, `/transcript`
and `/notes`, both gloss-only, never video/keypoints). The one difference
is the input source:

- **Webcam** source: identical to the live webcam flow.
- **Screen/tab/window** source (`getDisplayMedia`): whatever you choose to
  share in the browser's own picker is treated exactly like webcam video
  by this app — processed locally, frame by frame, for landmark
  extraction only, never recorded or uploaded. The browser itself shows
  its own persistent sharing indicator/border while a share is active
  (outside this app's control) so it's always visible when your screen is
  being captured, same as any other screen-share.

## Exports (Markdown / Text / PDF / Word)

All four export formats are generated **entirely client-side**, from the
notes/transcript text already sitting in the browser tab — no export
requires a network request to the backend at all. `jspdf` and `docx` (the
libraries used for PDF and Word generation) are loaded on demand from
your own `node_modules` at build time, then run as ordinary in-browser
JavaScript; nothing about the notes content is sent anywhere to produce
the downloaded file.

## What is stored on disk (and why)

| Data | Stored? | Where | Deleted when |
|---|---|---|---|
| Uploaded video (upload flow) | Temporarily | `ml_service/data/tmp/` | Immediately after processing (`finally` block) |
| Extracted keypoints, per-request (upload flow) | Temporarily | `ml_service/data/tmp/features/` | Immediately after processing |
| Webcam/screen-capture video or frames | Never | — | N/A (never leaves the browser tab) |
| Training dataset keypoints (`data/features_v2/*.npy`) | Yes, persistently | `data/features_v2/` | Manually, if you choose — these are training artifacts, not user session data |
| Trained model checkpoint/ONNX export | Yes, persistently | `models/sign_recog_v2/` | Manually |
| Session gloss history (webcam / live transcription) | In-memory only, in the browser tab | React state | Cleared on page reload, "Clear Session"/"Start New Session", or navigating away |

The training-dataset `.npy` files are intentionally kept — they're the
model-training artifact (like a compiled dataset), not personal data
collected during app usage.

## In-app privacy messaging

All three pages (upload, live webcam, live transcription) display a
visible privacy banner (`frontend/src/components/PrivacyBanner.tsx`)
summarizing the relevant policy above for that specific flow, so this
isn't just documentation — it's stated in the UI itself.

### Self-hosting the runtime assets (optional, for air-gapped setups)

Both `onnxruntime-web` and `@mediapipe/tasks-vision` ship their WASM
binaries inside their npm packages (`node_modules/onnxruntime-web/dist/`
and `node_modules/@mediapipe/tasks-vision/wasm/`). To avoid the CDN
fetches entirely:

1. Copy those directories into `frontend/public/ort/` and
   `frontend/public/mediapipe-wasm/` respectively.
2. Update `WASM_BASE` in `frontend/src/lib/holisticLandmarker.ts` and set
   `ort.env.wasm.wasmPaths` in `frontend/src/lib/onnxSession.ts` to point
   at the local `/ort/` path instead of the jsDelivr URL.
3. Download the `holistic_landmarker.task` model file once and serve it
   from `frontend/public/` too, updating `MODEL_URL` accordingly — this
   is the same model asset family `ml_service/feature_extraction.py`
   downloads server-side (see `FEATURE_SCHEMA.md` "Detector"), so a
   single manually-downloaded copy can be reused for both if you're
   setting up an air-gapped training+inference environment.

This isn't done by default because it adds a manual asset-management step
that's easy to get out of sync with the installed package version — the
current default (CDN, version-pinned to match `package.json`) is simpler
to keep correct for a demo.
