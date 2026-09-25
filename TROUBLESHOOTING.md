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

## Live webcam keeps showing "No sign detected" or "Uncertain — hold steady"

This is the intended behavior at low/mid confidence (Acceptance Test 4) —
the prediction simply isn't added to the session. If it happens
constantly:

- Check lighting and that both hands are in frame.
- Confirm `/model/meta` and `/model/onnx` are reachable (open
  `http://localhost:3001/model/meta` directly in a browser — you should
  see JSON, not an error).
- The webcam pipeline's hand-detection sampling rate (`DETECTION_INTERVAL_MS`
  in `Webcam.tsx`) approximates the training-time `frame_skip` cadence; if
  your model was trained with a very different `frame_skip`, consider
  adjusting this constant to match more closely.
- Lower `SessionSmoother`'s `ignoreThreshold`/`acceptThreshold` in
  `webcamPipeline.ts` if your model's confidence calibration runs lower
  than the 0.50/0.75 defaults in general (these are code constants, not
  currently exposed as a UI control).
- If a sign was misread, you don't have to restart the whole session —
  click the pencil icon on that row in Sign History to fix the label, the
  trash icon to remove it, or **Undo** to drop the most recent event.

## Camera permission denied / camera not found

The Webcam and Live Transcription pages surface these as readable
messages, not stack traces:
- **Permission denied**: check your browser's site settings (usually the
  lock/camera icon in the address bar) and allow camera access, then click
  Start Session again.
- **No camera found**: confirm a webcam is actually connected/enabled — on
  laptops, check for a physical privacy shutter or a function-key camera
  toggle.

## Screen/tab capture (Live Transcription's "Screen / Tab" source) doesn't work

- **The option is disabled/greyed out**: your browser doesn't support
  `getDisplayMedia` (or you're on an insecure `http://` origin other than
  `localhost` — screen capture requires a secure context). This is a real
  browser limitation, not a bug to work around; use Webcam instead, or
  switch to current desktop Chrome, Edge, or Firefox.
- **"Screen/tab sharing permission was denied"**: you cancelled or denied
  the browser's own share picker. Click "Screen / Tab" and Start again and
  choose something in the dialog.
- **The session stops unexpectedly**: if you stop sharing from the
  browser's own "Stop sharing" indicator (not this app's Stop button),
  the session ends the same way clicking Stop would — this is intentional,
  not a crash.
- Mobile browsers generally don't support `getDisplayMedia` at all; use
  Webcam mode there.

## "Couldn't load the recognition model" on the Webcam page

This means `GET /model/meta` or `GET /model/onnx` failed. Check:
1. Is the FastAPI ML service running (`http://127.0.0.1:8000/health`)?
2. Is the Node gateway running and pointed at it
   (`ML_SERVICE_URL` env var, default `http://127.0.0.1:8000/process`)?
3. Have you actually trained a model yet? `models/sign_recog_v2/checkpoints/demo.pt`
   and `models/sign_recog_v2/sign_recog.onnx` must exist — see "Training" in
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

This isn't a platform quirk — the legacy `mediapipe.solutions` API (which
`feature_extraction.py` used to use) was **removed entirely** from
current `mediapipe` PyPI releases (confirmed directly: `mediapipe==0.10.33`
ships `mediapipe.tasks` only, no `solutions` attribute at all, on every
platform). `feature_extraction.py` has since been migrated to the Tasks
API (`mediapipe.tasks.python.vision.HolisticLandmarker`) for exactly this
reason, alongside the hand+body+face schema v2 migration — see
`FEATURE_SCHEMA.md`. If you're on an older checked-out version of this
repo that still imports `mp.solutions.hands`, update to the current
`feature_extraction.py`, or pin `mediapipe<0.10.0` as a stopgap (not
recommended — you'd also need to stay on schema v1 / 126-dim features).

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

## PyTorch won't install, or crashes on `import torch` after installing

On a disk-constrained machine (this happened while building this
project, in a sandboxed container), a plain `pip install torch` on Linux
can fail or leave a broken partial install: the default PyPI wheel pulls
in several GB of CUDA dependency packages (`nvidia-cublas-cu12`,
`nvidia-cudnn-*`, etc.) even if you don't have an NVIDIA GPU, and if that
download is interrupted partway (out of disk space), `import torch` then
fails with something like `ModuleNotFoundError: No module named
'torch._strobelight'` or `libcublasLt.so not found`. If you hit this:
- Free disk space and retry a full `pip install --force-reinstall torch`
  (the CUDA wheels are large — budget several GB free).
- If you don't have an NVIDIA GPU and don't need CUDA, install a CPU-only
  build instead: see
  [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and
  select the CPU option — this avoids the large CUDA downloads entirely.
- A `pip uninstall torch` followed by manually deleting any leftover
  `nvidia*`/`triton*` directories under your Python environment's
  `site-packages` clears a broken partial install before retrying.

## `evaluate.py` says "nothing to evaluate" or can't find a test split

`evaluate.py` deliberately has **no fallback** here (see
`split_utils.resolve_test_indices()`): if `data/index.csv` has no
`split` column, or the column has no rows labeled `test`, it raises an
error rather than silently evaluating against validation or training
rows instead — using a leakage-free held-out test set is the entire
point of this script (project brief RULE 11/16). Check that:
1. Your metadata CSV actually has a `split` column with `test` values
   (`data/data_meta/metadata*.csv`).
2. `build_index.py` was run against that metadata CSV, so `data/index.csv`
   carries the `split` column through.
3. Features were extracted for those specific test-split videos (not
   just train/val) — see `feature_extraction.py --splits test`.

## Word (.docx) export produces an error or a file Word won't open

- Check the browser console for the actual error — `NotesPanel`'s Export
  row shows a short message under the buttons if generation fails, but
  the console has the full stack trace.
- This uses the `docx` npm package, dynamically loaded on first use (not
  bundled into the initial page load) — if your network blocks that chunk
  from loading (e.g. a very restrictive corporate proxy), the PDF export
  and other formats will still work since they're separate code paths.
- If the download starts but the file won't open in Word: confirm the
  file actually finished downloading (check its size isn't 0 bytes) --
  a browser tab closed mid-generation can produce a truncated file.
