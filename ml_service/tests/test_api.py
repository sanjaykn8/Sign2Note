"""End-to-end API tests using FastAPI's TestClient. Uses a tiny real
TemporalCNN checkpoint (structurally valid, not meaningfully trained) so
these exercise the actual model-loading and inference code paths, not
mocks of them. Only the video->keypoints step is faked (feature
extraction depends on a real video file + a working MediaPipe hand
detector, which is out of scope for a fast unit test) -- everything
downstream of that (windowing, inference, Viterbi, notes) is real.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from model import TemporalCNN


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    import api as api_module
    import infer as infer_module

    models_dir = tmp_path / "models" / "sign_recog" / "checkpoints"
    models_dir.mkdir(parents=True)
    checkpoint_path = models_dir / "demo.pt"

    label2id = {"DEFINITION": 0, "EXAMPLE": 1, "QUESTION": 2}
    model = TemporalCNN(126, len(label2id))
    torch.save({"model": model.state_dict(), "label2id": label2id,
               "input_dim": 126, "max_len": 16}, checkpoint_path)

    monkeypatch.setattr(api_module, "CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(api_module, "ONNX", tmp_path / "does_not_exist.onnx")
    infer_module._torch_cache = None
    infer_module._onnx_cache = None

    def fake_extract_single_video(video_path, out_dir, frame_skip=8):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(video_path).stem + ".npy")
        arr = np.random.rand(40, 126).astype(np.float32)
        np.save(out_path, arr)
        return out_path

    monkeypatch.setattr(api_module, "extract_single_video", fake_extract_single_video)

    return TestClient(api_module.app)


def test_health_reports_model_ready(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_model_meta_matches_checkpoint(api_client):
    r = api_client.get("/model/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["max_len"] == 16
    assert body["input_dim"] == 126
    assert body["num_classes"] == 3
    assert set(body["label2id"].keys()) == {"DEFINITION", "EXAMPLE", "QUESTION"}


def test_model_onnx_returns_404_when_no_export_exists(api_client):
    r = api_client.get("/model/onnx")
    assert r.status_code == 404


def test_notes_endpoint_generates_template_notes(api_client):
    r = api_client.post("/notes", json={"gloss_list": ["DEFINITION", "QUESTION"], "notes_mode": "template"})
    assert r.status_code == 200
    assert "DEFINITION" in r.json()["notes_md"]


def test_notes_endpoint_rejects_empty_gloss_list(api_client):
    r = api_client.post("/notes", json={"gloss_list": []})
    assert r.status_code == 422


def test_process_video_returns_notes_and_events(api_client, tmp_path):
    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"not a real video, extract_single_video is mocked")

    with open(video_path, "rb") as f:
        r = api_client.post(
            "/process",
            files={"file": ("fake.mp4", f, "video/mp4")},
            data={"notes_mode": "template", "threshold": "0.05"},
        )
    assert r.status_code == 200
    body = r.json()
    assert "notes_md" in body
    assert "gloss_list" in body
    assert "events" in body
    assert isinstance(body["gloss_list"], list)
    assert len(body["gloss_list"]) > 0  # always-produce-a-result fallback


def test_process_video_rejects_empty_upload(api_client, tmp_path):
    empty_path = tmp_path / "empty.mp4"
    empty_path.write_bytes(b"")
    with open(empty_path, "rb") as f:
        r = api_client.post("/process", files={"file": ("empty.mp4", f, "video/mp4")})
    assert r.status_code == 422
