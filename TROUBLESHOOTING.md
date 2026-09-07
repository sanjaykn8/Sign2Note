# Troubleshooting

## "No confident signs detected" / low_confidence: true on /process

The model's per-window confidence never crossed `threshold`. This is
normal for a demo-scale model with limited training data or an unfamiliar
camera angle/lighting. `/process` still returns a best-guess result
(`low_confidence: true`) rather than a dead end — see `ARCHITECTURE.md`
for why the upload flow and the webcam flow handle this differently. To
reduce how often this happens: lower `threshold` on the request, or
retrain with more samples per class / a cleaner camera setup matching your
training data's conditions.

## Live webcam keeps showing "Low confidence — please repeat"

This is the intended behavior at low confidence (Acceptance Test 4) — the
prediction simply isn't added to the session. If it happens constantly:

- Check lighting and that both hands are in frame.
- Confirm `/model/meta` and `/model/onnx` are reachable (open
  `http://localhost:3001/model/meta` directly in a browser — you should
  see JSON, not an error).
- The webcam pipeline's hand-detection sampling rate (`DETECTION_INTERVAL_MS`
  in `Webcam.tsx`) approximates the training-time `frame_skip` cadence; if
  your model was trained with a very different `frame_skip`, consider
  adjusting this constant to match more closely.
- Lower `SessionSmoother`'s `acceptThreshold` in `webcamPipeline.ts` if
  your model's confidence calibration runs lower than 0.70 in general
  (this is a code constant, not currently exposed as a UI control).

## Camera permission denied / camera not found

The Webcam page surfaces these as readable messages, not stack traces:
- **Permission denied**: check your browser's site settings (usually the
  lock/camera icon in the address bar) and allow camera access, then click
  Start Session again.
- **No camera found**: confirm a webcam is actually connected/enabled — on
  laptops, check for a physical privacy shutter or a function-key camera
  toggle.

## "Couldn't load the recognition model" on the Webcam page

This means `GET /model/meta` or `GET /model/onnx` failed. Check:
1. Is the FastAPI ML service running (`http://127.0.0.1:8000/health`)?
2. Is the Node gateway running and pointed at it
   (`ML_SERVICE_URL` env var, default `http://127.0.0.1:8000/process`)?
3. Have you actually trained a model yet? `models/sign_recog/checkpoints/demo.pt`
   and `models/sign_recog/sign_recog.onnx` must exist — see "Training" in
   `README.md`.

## LLM notes always fall back to template mode

`generate_notes(mode="llm", ...)` catches ANY failure (connection refused,
model not loaded, timeout, malformed response) and falls back silently,
printing the reason to the ML service's console — check that log first.
Common causes:
- The LLM server (llama.cpp / Ollama) isn't running.
- `LLM_BASE_URL` in `.env` doesn't match the port your server is actually
  listening on.
- `LLM_MODEL` doesn't match a model your server actually has loaded
  (for llama.cpp, the `--alias` you passed; for Ollama, a model you've
  actually `ollama pull`ed).
- `openai` Python package isn't installed (`pip install openai` — it's in
  `requirements.txt`, but confirm your active virtualenv has it).

## `torch.onnx.export` prints warnings or produces a `.onnx.data` sidecar file

Newer PyTorch versions (2.9+ as of this writing) default to a
"dynamo"-based ONNX exporter, which behaves slightly differently from the
older TorchScript-based one `train.py` was originally written against:
warnings about opset conversion are typically harmless noise, and for
larger models it may write external tensor data into a `<model>.onnx.data`
file alongside the `.onnx` file (this is normal ONNX behavior for
external data, not a bug — just make sure both files travel together if
you move/copy the exported model). The resulting `.onnx` file still loads
and runs correctly with `onnxruntime`. If you hit an actual export
failure (not just warnings), `pip install onnxscript` — the dynamo
exporter depends on it and some environments don't pull it in
automatically as a transitive dependency.

## `mediapipe.solutions` doesn't exist / AttributeError on `mp.solutions.hands`

Some `mediapipe` wheel builds — particularly certain Linux
sandboxed/minimal environments — ship without the legacy `solutions` API
(`feature_extraction.py` uses `mediapipe.solutions.hands`, the same API
MediaPipe has supported for years). If you hit this:
- Confirm you're on a standard desktop install (Windows/Mac/typical Linux
  desktop) — this is what the project is developed and tested against.
- Try `pip install --force-reinstall mediapipe` to get a full wheel for
  your platform.
- This is unrelated to the browser webcam feature, which uses a completely
  different, browser-native MediaPipe package
  (`@mediapipe/tasks-vision`) and doesn't depend on the Python
  `mediapipe.solutions` API at all.

## `npm install` fails in `frontend/` with a peer dependency conflict

If you see a conflict between `vite` and `@vitejs/plugin-react-swc`: this
project pins `@vitejs/plugin-react-swc@^4.3.3`, which supports Vite 8. If
you've modified `vite`'s version pin, keep the plugin version compatible
(check `npm view @vitejs/plugin-react-swc peerDependencies` for the
current compatibility matrix).

## CORS errors in the browser console

FastAPI's CORS middleware (`ml_service/api.py`) explicitly allows
`localhost:8080`/`3000` and their `127.0.0.1` equivalents. If you're
running the frontend on a different port or host, add it to the
`allow_origins` list in `api.py`.

## Long video processing is slow

- Increase `frame_skip` (fewer keypoint frames extracted per second of
  video) at some cost to temporal resolution.
- Increase `stride` (fewer, more widely-spaced sliding windows).
- `INFER_CHUNK_SIZE` (env var) controls memory/throughput tradeoff for the
  model forward pass, not overall speed for a given window count — lower
  it only if you're hitting memory limits, not for general speed.

## Empty gloss sequence but hands were clearly visible in the video

Check the FastAPI console output — `/process` prints the top 5 window
predictions with their confidence when nothing crosses `threshold`
(`[api] No window crossed threshold=...`), which tells you whether the
model is close-but-under-threshold (lower the threshold) or genuinely
confused (retrain / check the vocabulary actually includes that sign).
