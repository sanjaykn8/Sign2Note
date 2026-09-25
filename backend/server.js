const express = require('express');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');

const app = express();
app.use(cors());
// Default express.json() limit (100kb) is far too small for /recognize's
// keypoints payload (a live-recognition burst of a few hundred frames at
// 285 floats each, as JSON text, can run into single-digit MB) -- raised
// to comfortably fit that without opening the door to arbitrarily large
// bodies (25mb is still bounded, matching the spirit of section 39).
app.use(express.json({ limit: '25mb' }));

// Memory storage: the gateway never writes uploaded video to disk.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
});

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000/process';
const ML_BASE_URL = ML_SERVICE_URL.replace(/\/process$/, '');

app.get('/health', async (_req, res) => {
  try {
    const r = await fetch(`${ML_BASE_URL}/health`);
    const json = await r.json();
    res.json({ gateway: 'ok', ml: json });
  } catch (err) {
    res.status(503).json({ gateway: 'ok', ml: 'unavailable', error: err.message });
  }
});

// Model metadata + weights, so the browser can run the webcam demo's
// inference client-side (onnxruntime-web) without ever uploading video.
app.get('/model/meta', async (_req, res) => {
  try {
    const r = await fetch(`${ML_BASE_URL}/model/meta`);
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (err) {
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

app.get('/model/onnx', async (_req, res) => {
  try {
    const r = await fetch(`${ML_BASE_URL}/model/onnx`);
    if (!r.ok) {
      const json = await r.json().catch(() => ({ error: `ML service returned ${r.status}` }));
      return res.status(r.status).json(json);
    }
    res.set('Content-Type', r.headers.get('content-type') || 'application/octet-stream');
    const buf = Buffer.from(await r.arrayBuffer());
    res.send(buf);
  } catch (err) {
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

app.post('/upload', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video file supplied.' });

  const fd = new FormData();
  fd.append('file', req.file.buffer, {
    filename: req.file.originalname || 'upload.mp4',
    contentType: req.file.mimetype || 'video/mp4',
  });
  fd.append('notes_mode', req.body.notes_mode || 'template');
  // Only forward llm_model if the caller actually chose one -- omitting it
  // lets the ML service fall back to its own LLM_MODEL env-configured
  // default instead of us silently overriding it with a hard-coded value.
  if (req.body.llm_model) fd.append('llm_model', req.body.llm_model);
  fd.append('style', req.body.style || 'concise');
  fd.append('frame_skip', req.body.frame_skip || '8');
  fd.append('stride', req.body.stride || '12');
  if (req.body.threshold) fd.append('threshold', req.body.threshold);

  try {
    const r = await fetch(ML_SERVICE_URL, { method: 'POST', body: fd, headers: fd.getHeaders() });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

// Generate notes directly from an already-recognized gloss sequence (no
// video/keypoints involved) -- used by the live webcam session's
// "Generate Notes" button. All keypoint extraction + ONNX inference for
// the webcam flow happens client-side in the browser; only the final
// recognized gloss words reach the backend here.
app.post('/notes', async (req, res) => {
  try {
    const r = await fetch(`${ML_BASE_URL}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (err) {
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

// Generate a natural-language transcript from an already-recognized gloss
// sequence -- Live Transcription mode's "Stop & Generate" step. Separate
// from /notes: a transcript (flowing prose) and structured notes
// (headed/bulleted) are different output layers built from different
// prompts, and the UI can request both from the same gloss list.
app.post('/transcript', async (req, res) => {
  try {
    const r = await fetch(`${ML_BASE_URL}/transcript`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (err) {
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

// Recognition-only: pre-extracted keypoints -> glosses, no notes, no video.
// Used by a client that can extract landmarks locally but doesn't have (or
// hasn't finished loading) a local recognition model -- the distributed
// fallback path (see ARCHITECTURE.md "Distributed architecture" / RULE 8):
// keypoints, never raw video, are sent here so the main server's model can
// do the recognition step instead.
app.post('/recognize', async (req, res) => {
  try {
    const r = await fetch(`${ML_BASE_URL}/recognize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (err) {
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
