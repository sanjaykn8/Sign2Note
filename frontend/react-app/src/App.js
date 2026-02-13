import React, { useState } from 'react';
import axios from 'axios';

function App(){
  const [file, setFile] = useState(null);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);

  const upload = async () => {
    if (!file) return alert("Choose a file");
    const fd = new FormData();
    fd.append('file', file);
    // set use_llama to false for demo
    fd.append('use_llama', 'false');
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:3000/upload', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 10 * 60 * 1000
      });
      setNotes(res.data.notes_md || JSON.stringify(res.data));
    } catch (e) {
      setNotes("Error: " + e.toString());
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Sign2Notes — Demo</h2>
      <input type="file" accept="video/*" onChange={e => setFile(e.target.files[0])} />
      <br/><br/>
      <button onClick={upload} disabled={loading}>{loading ? "Processing..." : "Upload & Generate Notes"}</button>
      <hr/>
      <div>
        <h3>Generated Notes</h3>
        <div style={{ whiteSpace: 'pre-wrap', border: '1px solid #ddd', padding: 10 }}>
          {notes}
        </div>
      </div>
    </div>
  );
}

export default App;
