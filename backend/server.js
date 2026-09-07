const express = require('express');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');

const app = express();
app.use(cors());
app.use(express.json());

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

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
