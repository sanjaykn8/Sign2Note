# Privacy

What leaves each device, in each mode. "Main laptop" = the machine running
the gateway and the ML service. Nothing is sent to any cloud service by
Sign2Notes itself; the optional LLM is a local server you configure.

## Summary

| Mode / path | Leaves the client device | Stored on the main laptop |
|---|---|---|
| Video Upload | the video file (to the main laptop) | temporarily: the upload and its extracted features in `data/tmp/`, deleted when the request finishes (success or error). Gateway keeps the upload in memory only. |
| Live modes, **local** recognizer | nothing while signing; recognized gloss words (+ their confidences, optional context you type) when you click Generate / Stop & Generate | nothing |
| Live modes, **server** recognizer (fallback) | landmark coordinates of the current window (numbers: hand/pose/face points, ~24 frames, about every 0.27 s) to `/recognize` while signing; gloss words when generating | nothing (features are processed in memory) |
| Notes / transcript generation | gloss words, confidences, optional context → main laptop → optional local LLM server | nothing |
| Exports | nothing — files are created in the browser | nothing |

Video frames from the camera or a shared screen/tab are **never** sent
anywhere in the live modes. The Live Webcam / Live Transcription pages
show which recognizer is active and say which of the two rows above
applies.

## Details

**Video Upload.** The browser uploads the file to the gateway (memory
only), which streams it to the ML service. The ML service writes it to a
temporary file under `data/tmp/` (needed by OpenCV to decode it), extracts
landmarks to a temporary `.npy`, recognizes, generates notes, and deletes
both files in a `finally` block. Upload size and type are limited.

**Live modes.** MediaPipe Holistic runs in the browser (WebAssembly). The
first time, the browser downloads the MediaPipe runtime from
`cdn.jsdelivr.net` and the Holistic model from
`storage.googleapis.com` (no user data is sent with these requests; they
are ordinary file downloads, cached afterwards). If the recognition model
can run locally, only gloss words leave the device. Otherwise the current
window of landmark coordinates is sent to the main laptop for recognition
— coordinates of tracked points, not images, but they do describe body,
hand and facial movement and should be treated as personal data.

**Screen/tab capture** uses the browser's own picker; only what you choose
is captured, and the same rules as the webcam apply.

**Language generation.** The ML service sends the glosses (and the
optional context) to the LLM server at `LLM_BASE_URL`. Keep that a local
server (llama.cpp/Ollama on the main laptop). If you point it at a remote
service, gloss text leaves your network.

**Network.** With `SIGN2NOTES_API_TOKEN` unset the gateway only listens on
127.0.0.1. Setting the token opens it to the LAN; traffic is plain HTTP
unless you put TLS in front of it (see SETUP.md). Do not expose it to the
internet.

**Logs.** The ML service logs request lines, errors with tracebacks, and
— when no window passes the confidence threshold — the top predicted
labels. It does not log video, landmarks or notes text.

## What is stored on disk

- `data/tmp/` — transient files during an upload only.
- Browser `localStorage` — the server address and access token you entered
  in Connection settings (clear them by saving empty values).
- Nothing else is persisted: no session history, no notes, no recordings.

## Self-hosting the browser assets (air-gapped setups)

Copy `node_modules/@mediapipe/tasks-vision/wasm` and the
`holistic_landmarker.task` model (also in `models/mediapipe/`) to a
location served by the frontend and change `WASM_BASE` / `MODEL_URL` in
`frontend/src/lib/holisticLandmarker.ts`.
