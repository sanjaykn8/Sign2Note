import React, { useState } from 'react';
import axios from 'axios';

/* ── tiny markdown → HTML renderer (no extra deps) ─────────────────────── */
function renderMarkdown(md) {
  return md
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,    '<em>$1</em>')
    .replace(/^- (.+)$/gm,   '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    .replace(/\n{2,}/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');
}

function App() {
  const [file,    setFile]    = useState(null);
  const [notes,   setNotes]   = useState('');
  const [glosses, setGlosses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  const upload = async () => {
    if (!file) { alert('Please choose a video file first.'); return; }
    setLoading(true);
    setError('');
    setNotes('');
    setGlosses([]);

    const fd = new FormData();
    fd.append('file',       file);
    fd.append('use_llama',  'false');
    fd.append('use_ollama', 'false');  // set to true if Ollama is running

    try {
      const res = await axios.post('http://localhost:3001/upload', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 10 * 60 * 1000,   // 10 min timeout for longer videos
      });
      setNotes(res.data.notes_md   || '');
      setGlosses(res.data.gloss_list || []);
    } catch (e) {
      const msg = e.response?.data?.error || e.message || String(e);
      setError('Error: ' + msg);
    } finally {
      setLoading(false);
    }
  };

  const styles = {
    wrap:    { maxWidth: 800, margin: '0 auto', padding: 24, fontFamily: 'sans-serif' },
    title:   { fontSize: 24, fontWeight: 700, marginBottom: 8 },
    sub:     { color: '#555', marginBottom: 20 },
    row:     { display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 },
    btn:     { padding: '8px 20px', background: loading ? '#888' : '#1a73e8',
               color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 15 },
    err:     { color: 'red', marginBottom: 12 },
    glossBox:{ background: '#f0f4ff', borderRadius: 6, padding: '8px 14px', marginBottom: 16 },
    notesBox:{ border: '1px solid #ddd', borderRadius: 6, padding: 16,
               lineHeight: 1.6, background: '#fafafa' },
  };

  return (
    <div style={styles.wrap}>
      <div style={styles.title}>Sign2Notes — Demo</div>
      <div style={styles.sub}>Upload a sign-language video to generate structured notes.</div>

      <div style={styles.row}>
        <input type="file" accept="video/*" onChange={e => setFile(e.target.files[0])} />
        <button style={styles.btn} onClick={upload} disabled={loading}>
          {loading ? '⏳ Processing…' : '⬆ Upload & Generate Notes'}
        </button>
      </div>

      {error && <div style={styles.err}>{error}</div>}

      {glosses.length > 0 && (
        <div style={styles.glossBox}>
          <strong>Detected signs:</strong> {glosses.join(' · ')}
        </div>
      )}

      {notes && (
        <div>
          <h3>Generated Notes</h3>
          <div
            style={styles.notesBox}
            dangerouslySetInnerHTML={{ __html: renderMarkdown(notes) }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
