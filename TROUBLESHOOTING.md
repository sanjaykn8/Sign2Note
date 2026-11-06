# Troubleshooting

## Status indicator says "Server offline"

The page cannot reach the gateway.
1. Is `backend` running? `curl http://localhost:3001/health` on the main laptop.
2. On another device: open **Connection settings** and check the address is
   `http://<main-laptop-ip>:3001` (not `localhost`, which means the device
   itself). Is the gateway listening on the network? Without
   `SIGN2NOTES_API_TOKEN` (or an explicit `HOST`) it only listens on
   127.0.0.1 — by design.
3. Windows Firewall: allow inbound TCP 3001 and 8080 on the Private profile.

## "Server: limited" / "The server has no usable recognition model"

`/health` → `ml.model_problem` says why. Typical causes: no checkpoint at
`MODEL_PATH`; a checkpoint whose `input_dim` doesn't match its own feature
schema; the ML service not running (`ml: "unavailable"`; start it, check
its console).

## "Missing or invalid access token" (401)

The gateway has `SIGN2NOTES_API_TOKEN` set. Enter the same value under
Connection settings on each device.

## `import mediapipe` fails (TensorFlow / NumPy / `_ARRAY_API` / `GetPrototype` errors)

mediapipe 0.10.21 needs NumPy < 2 and protobuf < 5, and it imports
TensorFlow if TensorFlow is installed (TensorFlow 2.17 also needs NumPy < 2).
Install the pinned versions:
`py -3.11 -m pip install "numpy>=1.24,<2" "protobuf>=4.25.3,<5"`
(or re-run `py -3.11 -m pip install -r ml_service\requirements.txt`).

## Python extraction crashes: `packet.cc ... Check failed: holder_ != nullptr The packet is empty`

That is the MediaPipe **Tasks** `HolisticLandmarker` (0.10.21, Python,
Windows). Use the default `HOLISTIC_BACKEND=solutions` (classic
`mp.solutions.holistic`). Keep `mediapipe==0.10.21` — later releases
removed `mediapipe.solutions`.

## Live modes: "Camera and screen capture need a secure page"

Browsers allow camera/screen capture only on `https://` or `localhost`.
On other devices, use HTTPS (SETUP.md "Connecting other devices"), or use
Video Upload, which works over plain http.

## Live modes: "Local recognizer unavailable … using server recognition"

The ONNX model couldn't be downloaded or failed its test inference in
this browser. Recognition still works via the server (landmarks, not
video, are sent — see PRIVACY.md). Reload after fixing the cause (e.g. a
missing `sign_recog.onnx`, or an old browser without WebAssembly SIMD).

## Live modes: "The server's model uses feature schema … which this version of the app cannot produce"

The served model's schema is unknown to this frontend build. Update the
frontend, or serve a model with schema 1.0 or 2.0.

## Live modes keep showing "No sign detected" / "Uncertain"

- "No sign" appears whenever no hand is visible for 3 samples — check
  framing and lighting; keep both hands and the upper body in view.
- "Uncertain" = confidence 0.50–0.75. The models are trained on 20
  signers and a fixed vocabulary; unfamiliar signers, angles or signs
  produce low confidence. That is expected, not a bug.
- If recognition is consistently poor, check which model is served
  (`/health`): the legacy hand-only model is weaker and was evaluated on a
  random split.

## "Server recognition is failing"

The live page couldn't reach `/recognize` 8 times in a row. Check the
status indicator and the ML service console. Recognized glosses so far are
kept.

## Upload: "No frames could be processed" / "could not be opened as a video"

The file is corrupt or uses a codec OpenCV can't decode. Re-encode to
H.264 MP4 (`ffmpeg -i in.mov -c:v libx264 out.mp4`).

## Upload: 413 / "exceeds the … limit"

Raise `MAX_UPLOAD_MB` (gateway), `MAX_UPLOAD_SIZE` (ML service) and
`VITE_MAX_UPLOAD_MB` (frontend) together, or trim the video. Very long
videos are also limited by `MAX_VIDEO_SECONDS`.

## Upload: "No confident sign — best guess shown"

No window passed the confidence threshold; the upload page shows the most
frequent prediction, flagged. Lower the threshold under Advanced settings,
or edit the glosses before regenerating notes.

## Notes always say "LLM unavailable — deterministic fallback used"

Hover the badge for the reason, or check the ML service console. Usual
causes: LLM server not running, wrong `LLM_BASE_URL`/`LLM_MODEL`, model
slower than `LLM_TIMEOUT` (raise it), or output without Markdown structure
(the model ignored the instructions — try a different model).

## PDF export: "uses a Latin-only font and can't show …"

The PDF exporter's built-in font has no glyphs for that script. Use Word
(.docx), Markdown or Text, which keep Unicode.

## Training: `ModuleNotFoundError: onnxscript` during export

Fixed in `train.py` (TorchScript exporter, `dynamo=False`). Update the file.

## Training/evaluation: "No indexed videos have extracted feature files"

The index CSV and `--feature_dir` don't match (e.g. a v1 index with v2
features). Build a v2 index with `build_index.py --feature_dir
data/features_v2 --out_csv data/index_v2.csv --vocab_json config/vocab_v2.json`.

## Training: "Feature dimension mismatch … 126 … 285"

You are mixing schemas. v1 (126) and v2 (285) features must live in
different directories and be used with matching models; pass
`--feature_schema_version 1.0` only for genuinely legacy features.

## `evaluate.py` warns "NOT trained with the official train/val split"

The checkpoint (e.g. the legacy model) was trained on a random split, so
evaluated samples may have been in its training data. The numbers are not
a clean held-out result.

## Frontend tests time out on a busy machine

The first rendered-component test can exceed vitest's 5 s default while
the CPU is saturated (e.g. during feature extraction). Re-run when idle or
pass `--testTimeout=30000`.
