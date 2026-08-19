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

app.get('/health', async (_req, res) => {
  try {
    const r = await fetch(ML_SERVICE_URL.replace('/process', '/health'));
    const json = await r.json();
    res.json({ gateway: 'ok', ml: json });
  } catch (err) {
    res.status(503).json({ gateway: 'ok', ml: 'unavailable', error: err.message });
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
  fd.append('llm_model', req.body.llm_model || 'gemma4');
  fd.append('style', req.body.style || 'concise');
  fd.append('frame_skip', req.body.frame_skip || '8');
  fd.append('stride', req.body.stride || '12');
  fd.append('threshold', req.body.threshold || '0.55');

  try {
    const r = await fetch(ML_SERVICE_URL, { method: 'POST', body: fd, headers: fd.getHeaders() });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: `ML service unavailable: ${err.message}` });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
