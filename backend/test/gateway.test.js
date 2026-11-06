// Gateway tests (node:test, no extra dependencies). A tiny fake ML service
// stands in for FastAPI so these run without Python.
const test = require('node:test');
const assert = require('node:assert');
const http = require('node:http');

function startFakeMl(handler) {
  return new Promise((resolve) => {
    const srv = http.createServer(handler);
    srv.listen(0, '127.0.0.1', () => resolve(srv));
  });
}

function loadApp(env) {
  for (const [k, v] of Object.entries(env)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  delete require.cache[require.resolve('../server.js')];
  return require('../server.js');
}

async function listen(app) {
  return new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
}

const base = (s) => `http://127.0.0.1:${s.address().port}`;

test('proxies notes JSON and relays status codes', async () => {
  const ml = await startFakeMl((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      if (req.url === '/notes') {
        const parsed = JSON.parse(body);
        res.statusCode = parsed.gloss_list.length ? 200 : 422;
        res.end(JSON.stringify({ notes_md: `# ${parsed.gloss_list.join(' ')}` }));
      } else {
        res.end(JSON.stringify({ status: 'ok' }));
      }
    });
  });
  const { createApp } = loadApp({ ML_SERVICE_URL: `${base(ml)}/process`, SIGN2NOTES_API_TOKEN: undefined });
  const gw = await listen(createApp());
  try {
    let r = await fetch(`${base(gw)}/notes`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gloss_list: ['Hello'] }) });
    assert.strictEqual(r.status, 200);
    assert.strictEqual((await r.json()).notes_md, '# Hello');
    r = await fetch(`${base(gw)}/notes`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ gloss_list: [] }) });
    assert.strictEqual(r.status, 422);
    r = await fetch(`${base(gw)}/notes`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{not json' });
    assert.strictEqual(r.status, 400);
    assert.ok((await r.json()).error);
  } finally {
    gw.close();
    ml.close();
  }
});

test('token auth protects everything except /health', async () => {
  const ml = await startFakeMl((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ status: 'ok', url: req.url }));
  });
  const { createApp, resolveHost } = loadApp({ ML_SERVICE_URL: `${base(ml)}/process`, SIGN2NOTES_API_TOKEN: 'secret-token', HOST: undefined });
  assert.strictEqual(resolveHost(), '0.0.0.0');
  const gw = await listen(createApp());
  try {
    let r = await fetch(`${base(gw)}/health`);
    assert.strictEqual(r.status, 200);
    assert.strictEqual((await r.json()).auth_required, true);
    r = await fetch(`${base(gw)}/model/meta`);
    assert.strictEqual(r.status, 401);
    r = await fetch(`${base(gw)}/model/meta`, { headers: { Authorization: 'Bearer wrong-token!' } });
    assert.strictEqual(r.status, 401);
    r = await fetch(`${base(gw)}/model/meta`, { headers: { Authorization: 'Bearer secret-token' } });
    assert.strictEqual(r.status, 200);
    r = await fetch(`${base(gw)}/model/meta`, { headers: { 'X-Sign2Notes-Token': 'secret-token' } });
    assert.strictEqual(r.status, 200);
  } finally {
    gw.close();
    ml.close();
  }
  const fresh = loadApp({ SIGN2NOTES_API_TOKEN: undefined, HOST: undefined });
  assert.strictEqual(fresh.resolveHost(), '127.0.0.1');
});

test('ML outage gives 502 JSON; slow ML gives 504', async () => {
  const { createApp } = loadApp({ ML_SERVICE_URL: 'http://127.0.0.1:9/process', SIGN2NOTES_API_TOKEN: undefined });
  const gw = await listen(createApp());
  try {
    let r = await fetch(`${base(gw)}/transcript`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"gloss_list":["A"]}' });
    assert.strictEqual(r.status, 502);
    assert.match((await r.json()).error, /unavailable/);
    r = await fetch(`${base(gw)}/health`);
    assert.strictEqual(r.status, 503);
  } finally {
    gw.close();
  }

  const slow = await startFakeMl(() => { /* never answers */ });
  const app2 = loadApp({ ML_SERVICE_URL: `${base(slow)}/process`, TIMEOUT_RECOGNIZE_MS: '300' });
  const gw2 = await listen(app2.createApp());
  try {
    const r = await fetch(`${base(gw2)}/recognize`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"features":[[0]]}' });
    assert.strictEqual(r.status, 504);
  } finally {
    gw2.close();
    slow.closeAllConnections?.();
    slow.close();
    delete process.env.TIMEOUT_RECOGNIZE_MS;
  }
});

test('upload: size limit and type filter return JSON errors; optional fields only forwarded when set', async () => {
  let seen = null;
  const ml = await startFakeMl((req, res) => {
    let body = '';
    req.setEncoding('latin1');
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      seen = body;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ notes_md: '# ok', gloss_list: ['A'] }));
    });
  });
  const { createApp } = loadApp({ ML_SERVICE_URL: `${base(ml)}/process`, MAX_UPLOAD_MB: '0.001' });
  const gw = await listen(createApp());
  try {
    const big = new FormData();
    big.append('file', new Blob([Buffer.alloc(5000)], { type: 'video/mp4' }), 'a.mp4');
    let r = await fetch(`${base(gw)}/upload`, { method: 'POST', body: big });
    assert.strictEqual(r.status, 413);
    assert.match((await r.json()).error, /limit/);

    const txt = new FormData();
    txt.append('file', new Blob([Buffer.from('hi')], { type: 'text/plain' }), 'a.txt');
    r = await fetch(`${base(gw)}/upload`, { method: 'POST', body: txt });
    assert.strictEqual(r.status, 415);

    const ok = new FormData();
    ok.append('file', new Blob([Buffer.alloc(100)], { type: 'video/mp4' }), 'a.mp4');
    ok.append('style', 'academic');
    r = await fetch(`${base(gw)}/upload`, { method: 'POST', body: ok });
    assert.strictEqual(r.status, 200);
    assert.ok(seen.includes('name="style"'));
    assert.ok(!seen.includes('name="frame_skip"'), 'frame_skip must not be forced when the client omitted it');
  } finally {
    gw.close();
    ml.close();
    delete process.env.MAX_UPLOAD_MB;
  }
});
