/**
 * Sign2Notes gateway (Express). The browser talks only to this process; it
 * forwards to the ML service (FastAPI, bound to 127.0.0.1 by default) and
 * adds what a network-facing entry point needs:
 *
 *   - optional shared-token authentication (SIGN2NOTES_API_TOKEN)
 *   - safe default binding: 127.0.0.1 unless a token is configured or
 *     HOST is set explicitly (an unauthenticated server is never exposed
 *     to the LAN by accident)
 *   - configurable CORS origins
 *   - request size limits and JSON (not HTML) error responses
 *   - per-route upstream timeouts, so a hung ML/LLM call can't hold a
 *     request open forever
 *
 * Routes (same paths as the ML service):
 *   GET  /health            public; minimal status
 *   GET  /model/meta        served recognition model's metadata
 *   GET  /model/onnx        ONNX export for in-browser recognition
 *   POST /recognize         keypoints -> glosses       (recognition service)
 *   POST /recognize/video   video -> glosses           (recognition service)
 *   POST /notes             glosses -> notes           (language-generation service)
 *   POST /transcript        glosses -> transcript      (language-generation service)
 *   POST /upload            video -> glosses + notes   (combined; upload page)
 */
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');
const crypto = require('crypto');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000/process';
const ML_BASE_URL = ML_SERVICE_URL.replace(/\/process$/, '');
const API_TOKEN = process.env.SIGN2NOTES_API_TOKEN || '';
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 200);
const TIMEOUTS_MS = {
  meta: Number(process.env.TIMEOUT_META_MS || 15000),
  onnx: Number(process.env.TIMEOUT_ONNX_MS || 60000),
  recognize: Number(process.env.TIMEOUT_RECOGNIZE_MS || 60000),
  // LLM generation: ML side bounds the LLM call (LLM_TIMEOUT) and falls
  // back to the template, so this only needs headroom above that.
  generate: Number(process.env.TIMEOUT_GENERATE_MS || 240000),
  video: Number(process.env.TIMEOUT_VIDEO_MS || 30 * 60 * 1000),
};

function corsOptions() {
  const raw = process.env.CORS_ORIGINS;
  if (!raw || raw.trim() === '*') return { origin: true }; // reflect request origin
  const allowed = raw.split(',').map((s) => s.trim()).filter(Boolean);
  return {
    origin(origin, cb) {
      // Same-origin / non-browser requests have no Origin header.
      if (!origin || allowed.includes(origin)) return cb(null, true);
      return cb(null, false);
    },
  };
}

function tokenMatches(given) {
  if (!API_TOKEN) return true;
  if (typeof given !== 'string' || given.length !== API_TOKEN.length) return false;
  return crypto.timingSafeEqual(Buffer.from(given), Buffer.from(API_TOKEN));
}

function requireToken(req, res, next) {
  if (!API_TOKEN) return next();
  const header = req.get('authorization') || '';
  const bearer = header.startsWith('Bearer ') ? header.slice(7) : '';
  const token = bearer || req.get('x-sign2notes-token') || '';
  if (tokenMatches(token)) return next();
  return res.status(401).json({ error: 'Missing or invalid access token. Set it under Connection settings.' });
}

/** fetch with an abort timeout; maps network failures to clear messages. */
async function upstream(path, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${ML_BASE_URL}${path}`, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === 'AbortError') {
      const e = new Error(`ML service did not respond within ${Math.round(timeoutMs / 1000)} s`);
      e.status = 504;
      throw e;
    }
    const e = new Error(`ML service unavailable: ${err.message}`);
    e.status = 502;
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function relayJson(res, r) {
  const text = await r.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { error: `ML service returned a non-JSON response (HTTP ${r.status})` };
  }
  res.status(r.status).json(body);
}

function createApp() {
  const app = express();
  app.disable('x-powered-by');
  app.use(cors(corsOptions()));
  // /recognize keypoint payloads: a few hundred frames x 285 floats as JSON
  // text can reach single-digit MB; 25 MB keeps that comfortable but bounded.
  app.use(express.json({ limit: process.env.JSON_LIMIT || '25mb' }));

  // Memory storage: the gateway never writes uploaded video to disk; the
  // ML service streams it to a temp file and deletes it after processing.
  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: Math.max(1, Math.floor(MAX_UPLOAD_MB * 1024 * 1024)), files: 1 },
    fileFilter(_req, file, cb) {
      const ok = /^video\//.test(file.mimetype || '') || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.originalname || '');
      if (ok) return cb(null, true);
      const e = new Error('Only video files (mp4, mov, avi, mkv, webm, m4v) are accepted.');
      e.status = 415;
      return cb(e);
    },
  });

  app.get('/health', async (_req, res) => {
    try {
      const r = await upstream('/health', {}, TIMEOUTS_MS.meta);
      const json = await r.json();
      res.json({ gateway: 'ok', auth_required: Boolean(API_TOKEN), ml: json });
    } catch (err) {
      res.status(503).json({ gateway: 'ok', auth_required: Boolean(API_TOKEN), ml: 'unavailable', error: err.message });
    }
  });

  app.use(requireToken);

  app.get('/model/meta', async (_req, res, next) => {
    try {
      relayJson(res, await upstream('/model/meta', {}, TIMEOUTS_MS.meta));
    } catch (err) {
      next(err);
    }
  });

  app.get('/model/onnx', async (_req, res, next) => {
    try {
      const r = await upstream('/model/onnx', {}, TIMEOUTS_MS.onnx);
      if (!r.ok) return relayJson(res, r);
      res.set('Content-Type', r.headers.get('content-type') || 'application/octet-stream');
      res.set('Cache-Control', 'no-cache');
      res.send(Buffer.from(await r.arrayBuffer()));
    } catch (err) {
      next(err);
    }
  });

  const forwardVideo = (mlPath, fields) => async (req, res, next) => {
    if (!req.file) return res.status(400).json({ error: 'No video file supplied.' });
    const fd = new FormData();
    fd.append('file', req.file.buffer, {
      filename: req.file.originalname || 'upload.mp4',
      contentType: req.file.mimetype || 'video/mp4',
    });
    // Forward only what the client actually set: omitted fields let the ML
    // service apply its own defaults (notably frame_skip, which it derives
    // from the video's fps to match the training sampling rate).
    for (const key of fields) {
      const v = req.body[key];
      if (v !== undefined && v !== null && v !== '') fd.append(key, String(v));
    }
    try {
      relayJson(res, await upstream(mlPath, { method: 'POST', body: fd, headers: fd.getHeaders() }, TIMEOUTS_MS.video));
    } catch (err) {
      next(err);
    }
  };

  const videoFields = ['frame_skip', 'stride', 'threshold', 'window_frames'];
  app.post('/upload', upload.single('file'), forwardVideo('/process', ['notes_mode', 'llm_model', 'style', ...videoFields]));
  app.post('/recognize/video', upload.single('file'), forwardVideo('/recognize/video', videoFields));

  const forwardJson = (mlPath, timeoutMs) => async (req, res, next) => {
    try {
      relayJson(res, await upstream(mlPath, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body ?? {}),
      }, timeoutMs));
    } catch (err) {
      next(err);
    }
  };

  app.post('/notes', forwardJson('/notes', TIMEOUTS_MS.generate));
  app.post('/transcript', forwardJson('/transcript', TIMEOUTS_MS.generate));
  app.post('/recognize', forwardJson('/recognize', TIMEOUTS_MS.recognize));

  app.use((_req, res) => res.status(404).json({ error: 'Not found' }));

  // Errors always come back as JSON (Express's default handler sends HTML,
  // which the frontend could only show as "Server error 500").
  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    let status = err.status || 500;
    let message = err.message || 'Internal error';
    if (err instanceof multer.MulterError) {
      status = err.code === 'LIMIT_FILE_SIZE' ? 413 : 400;
      message = err.code === 'LIMIT_FILE_SIZE' ? `Video exceeds the ${MAX_UPLOAD_MB} MB upload limit.` : `Upload error: ${err.message}`;
    } else if (err.type === 'entity.too.large') {
      status = 413;
      message = 'Request body is too large.';
    } else if (err.type === 'entity.parse.failed') {
      status = 400;
      message = 'Request body is not valid JSON.';
    }
    if (status >= 500) console.error('[gateway]', err);
    res.status(status).json({ error: message });
  });

  return app;
}

function resolveHost() {
  if (process.env.HOST) return process.env.HOST;
  return API_TOKEN ? '0.0.0.0' : '127.0.0.1';
}

if (require.main === module) {
  const PORT = Number(process.env.PORT || 3001);
  const HOST = resolveHost();
  createApp().listen(PORT, HOST, () => {
    console.log(`Backend listening on http://${HOST}:${PORT} (ML service: ${ML_BASE_URL})`);
    if (HOST === '127.0.0.1' && !process.env.HOST) {
      console.log('  Local-only. To let other devices connect, set SIGN2NOTES_API_TOKEN (binds 0.0.0.0) -- see SETUP.md.');
    }
    if (HOST !== '127.0.0.1' && HOST !== 'localhost' && !API_TOKEN) {
      console.warn('  WARNING: listening on the network WITHOUT authentication (HOST set, no SIGN2NOTES_API_TOKEN).');
    }
  });
}

module.exports = { createApp, resolveHost };
