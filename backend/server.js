// backend/server.js
const express   = require('express');
const multer    = require('multer');
const fetch     = require('node-fetch');
const FormData  = require('form-data');
const fs        = require('fs');
const path      = require('path');
const cors      = require('cors');

const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename:    (req, file, cb) => cb(null, Date.now() + '-' + file.originalname),
});
const upload = multer({ storage });

const app = express();
app.use(cors());
app.use(express.json());

const ML_SERVICE_URL = 'http://127.0.0.1:8000/process';

app.post('/upload', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'no file' });

  const fd = new FormData();
  fd.append('file',       fs.createReadStream(req.file.path), req.file.originalname);
  fd.append('use_llama',  req.body.use_llama  === 'true' ? 'true' : 'false');
  fd.append('use_ollama', req.body.use_ollama === 'true' ? 'true' : 'false');

  try {
    const r    = await fetch(ML_SERVICE_URL, { method: 'POST', body: fd });
    const json = await r.json();
    return res.json(json);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.toString() });
  }
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.listen(3001, () => console.log('Backend listening on http://localhost:3001'));
