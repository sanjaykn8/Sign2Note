"""End-to-end API tests using FastAPI's TestClient. Uses a tiny real
TemporalCNN checkpoint (structurally valid, not meaningfully trained) so
these exercise the actual model-loading and inference code paths, not
mocks of them. Only the video->keypoints step is faked (feature
extraction depends on a real video file + a working MediaPipe hand
detector, which is out of scope for a fast unit test) -- everything
downstream of that (windowing, inference, Viterbi, notes) is real.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from model import TemporalCNN
import feature_schema as fs
from checkpoint_meta import build_checkpoint_metadata


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    import api as api_module
    import infer as infer_module

    models_dir = tmp_path / "models" / "sign_recog_v2" / "checkpoints"
    models_dir.mkdir(parents=True)
    checkpoint_path = models_dir / "demo.pt"

    label2id = {"DEFINITION": 0, "EXAMPLE": 1, "QUESTION": 2}
    model = TemporalCNN(fs.FEATURE_DIM, len(label2id))
    torch.save(build_checkpoint_metadata(
        model_state=model.state_dict(), label2id=label2id,
        input_dim=fs.FEATURE_DIM, max_len=16, best_val_accuracy=0.5,
        training_config={"note": "synthetic test fixture, not a real trained model"},
    ), checkpoint_path)

    monkeypatch.setattr(api_module, "CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(api_module, "ONNX", tmp_path / "does_not_exist.onnx")
    infer_module._torch_cache = None
    infer_module._onnx_cache = None

    def fake_extract_single_video(video_path, out_dir, frame_skip=8, model_asset_path=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(video_path).stem + ".npy")
        arr = np.random.rand(40, fs.FEATURE_DIM).astype(np.float32)
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
    assert body["input_dim"] == fs.FEATURE_DIM
    assert body["num_classes"] == 3
    assert set(body["label2id"].keys()) == {"DEFINITION", "EXAMPLE", "QUESTION"}
    assert body["feature_schema_version"] == fs.FEATURE_SCHEMA_VERSION
    assert body["architecture"] == "temporal_cnn"


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


def test_process_video_rejects_invalid_notes_mode(api_client, tmp_path):
    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"not a real video, extract_single_video is mocked")
    with open(video_path, "rb") as f:
        r = api_client.post(
            "/process", files={"file": ("fake.mp4", f, "video/mp4")},
            data={"notes_mode": "not_a_real_mode"},
        )
    assert r.status_code == 422


def test_process_video_rejects_invalid_style(api_client, tmp_path):
    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"not a real video, extract_single_video is mocked")
    with open(video_path, "rb") as f:
        r = api_client.post(
            "/process", files={"file": ("fake.mp4", f, "video/mp4")},
            data={"style": "not_a_real_style"},
        )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /notes -- validation (empty list, too many glosses, invalid mode/style)
# ---------------------------------------------------------------------------

def test_notes_endpoint_rejects_too_many_glosses(api_client):
    import api as api_module
    # NotesRequest's Field(max_length=...) is bound to MAX_GLOSS_COUNT at
    # class-definition time, so this test exercises the actual configured
    # limit directly rather than trying to monkeypatch it after the fact.
    too_many = ["A"] * (api_module.MAX_GLOSS_COUNT + 1)
    r = api_client.post("/notes", json={"gloss_list": too_many})
    assert r.status_code == 422


def test_notes_endpoint_rejects_invalid_notes_mode(api_client):
    r = api_client.post("/notes", json={"gloss_list": ["DEFINITION"], "notes_mode": "not_a_real_mode"})
    assert r.status_code == 422


def test_notes_endpoint_rejects_invalid_style(api_client):
    r = api_client.post("/notes", json={"gloss_list": ["DEFINITION"], "style": "not_a_real_style"})
    assert r.status_code == 422


def test_notes_endpoint_accepts_each_supported_style(api_client):
    from notes_generator import SUPPORTED_STYLES
    for style in SUPPORTED_STYLES:
        r = api_client.post("/notes", json={"gloss_list": ["DEFINITION"], "style": style})
        assert r.status_code == 200, f"style={style} should be accepted"


# ---------------------------------------------------------------------------
# /transcript -- natural-language transcript (Layer 2), separate from /notes
# ---------------------------------------------------------------------------

def test_transcript_endpoint_returns_a_transcript_field_not_notes_md(api_client):
    r = api_client.post("/transcript", json={"gloss_list": ["DEFINITION", "QUESTION"]})
    assert r.status_code == 200
    body = r.json()
    assert "transcript" in body
    assert "notes_md" not in body


def test_transcript_endpoint_rejects_empty_gloss_list(api_client):
    r = api_client.post("/transcript", json={"gloss_list": []})
    assert r.status_code == 422


def test_transcript_endpoint_rejects_invalid_notes_mode(api_client):
    r = api_client.post("/transcript", json={"gloss_list": ["DEFINITION"], "notes_mode": "not_a_real_mode"})
    assert r.status_code == 422


def test_transcript_endpoint_has_no_style_field_requirement(api_client):
    # Transcript has no style parameter at all (unlike /notes) -- omitting
    # it entirely must still succeed.
    r = api_client.post("/transcript", json={"gloss_list": ["DEFINITION"]})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /recognize -- keypoints-only recognition, no video, no notes
# ---------------------------------------------------------------------------

def test_recognize_accepts_correctly_shaped_features_and_returns_glosses_not_notes(api_client):
    import feature_schema as fs
    features = np.random.rand(40, fs.FEATURE_DIM).astype(np.float32).tolist()
    r = api_client.post("/recognize", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    assert "glosses" in body
    assert "notes_md" not in body  # RULE: recognition and notes are separate concerns
    assert body["feature_schema_version"] == fs.FEATURE_SCHEMA_VERSION


def test_recognize_rejects_wrong_feature_dimension_with_a_clear_message(api_client):
    # Simulates an old (schema v1, 126-dim) client hitting a v2 server.
    features = np.random.rand(40, 126).astype(np.float32).tolist()
    r = api_client.post("/recognize", json={"features": features})
    assert r.status_code == 422
    body = r.json()
    assert "126" in body["error"] or body.get("got_feature_dim") == 126
    assert "incompatible" in body["error"].lower() or "mismatch" in body["error"].lower()


def test_recognize_rejects_empty_features(api_client):
    r = api_client.post("/recognize", json={"features": []})
    assert r.status_code == 422


def test_recognize_rejects_ragged_feature_rows(api_client):
    import feature_schema as fs
    features = [[0.0] * fs.FEATURE_DIM, [0.0] * (fs.FEATURE_DIM - 1)]  # inconsistent row length
    r = api_client.post("/recognize", json={"features": features})
    assert r.status_code == 422


def test_recognize_rejects_excessively_large_requests(api_client, monkeypatch):
    import api as api_module
    monkeypatch.setattr(api_module, "MAX_RECOGNIZE_FRAMES", 10)
    import feature_schema as fs
    features = np.zeros((50, fs.FEATURE_DIM), dtype=np.float32).tolist()
    r = api_client.post("/recognize", json={"features": features})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /model/meta -- v2 fields
# ---------------------------------------------------------------------------

def test_model_meta_includes_v2_fields_and_compatibility_warnings(api_client):
    r = api_client.get("/model/meta")
    assert r.status_code == 200
    body = r.json()
    assert "vocab_hash" in body
    assert "best_val_accuracy" in body
    assert "compatibility_warnings" in body
    assert body["compatibility_warnings"] == []  # fixture checkpoint is self-consistent
