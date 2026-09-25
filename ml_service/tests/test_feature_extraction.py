"""feature_extraction.py tests. build_feature_vector()'s own missing-
landmark/dimension behavior is covered exhaustively in
test_feature_schema.py; this file covers what's specific to
feature_extraction.py itself: the HolisticLandmarkerResult -> feature
vector glue, the feature-directory metadata/compatibility guard, and the
model-asset-download helper's error handling. None of this needs a real
video, camera, or downloaded model weights, so it runs anywhere.
"""
import json

import numpy as np
import pytest

import feature_extraction as fe
import feature_schema as fs


class _FakeLandmark:
    def __init__(self, x, y, z, visibility=None):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeHolisticResult:
    """Stand-in for mediapipe's HolisticLandmarkerResult -- just needs the
    four flat landmark-list attributes _result_to_feature_vector reads."""

    def __init__(self, left_hand=None, right_hand=None, pose=None, face=None):
        self.left_hand_landmarks = left_hand
        self.right_hand_landmarks = right_hand
        self.pose_landmarks = pose
        self.face_landmarks = face


def _hand(seed):
    return [_FakeLandmark(seed + i * 0.01, seed + i * 0.02, seed + i * 0.03) for i in range(21)]


# ---------------------------------------------------------------------------
# _result_to_feature_vector -- glue between HolisticLandmarkerResult and
# the canonical schema
# ---------------------------------------------------------------------------

def test_result_to_feature_vector_empty_result_is_zero_and_correct_length():
    vec = fe._result_to_feature_vector(_FakeHolisticResult())
    assert vec.shape == (fs.FEATURE_DIM,)
    assert np.all(vec == 0)


def test_result_to_feature_vector_matches_build_feature_vector_directly():
    result = _FakeHolisticResult(left_hand=_hand(1.0), right_hand=_hand(2.0))
    via_extraction = fe._result_to_feature_vector(result)
    via_schema = fs.build_feature_vector(_hand(1.0), _hand(2.0), None, None)
    assert np.array_equal(via_extraction, via_schema)


# ---------------------------------------------------------------------------
# feature_meta.json -- write + compatibility check
# ---------------------------------------------------------------------------

def test_write_and_check_feature_meta_round_trips(tmp_path):
    out_dir = tmp_path / "features_v2"
    fe._write_feature_meta(out_dir, frame_skip=8, dataset_format="fdmse")

    meta_path = out_dir / "feature_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["feature_schema_version"] == fs.FEATURE_SCHEMA_VERSION
    assert meta["feature_dim"] == fs.FEATURE_DIM
    assert meta["frame_skip"] == 8
    assert meta["dataset_format"] == "fdmse"

    fe.check_feature_dir_compatible(out_dir)  # should not raise


def test_check_feature_dir_compatible_is_a_noop_when_no_meta_exists(tmp_path):
    # A brand-new empty output dir has no feature_meta.json yet -- must not
    # raise (this is the "first run into a fresh dir" case).
    fe.check_feature_dir_compatible(tmp_path / "does_not_exist_yet")


def test_check_feature_dir_compatible_rejects_wrong_schema_version(tmp_path):
    out_dir = tmp_path / "old_features"
    out_dir.mkdir()
    (out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_schema_version": "1.0",
        "feature_dim": 126,
    }))
    with pytest.raises(fs.FeatureSchemaError, match="1.0"):
        fe.check_feature_dir_compatible(out_dir)


def test_check_feature_dir_compatible_rejects_wrong_dim_even_if_version_matches(tmp_path):
    # Defensive: a hand-edited or corrupted meta file claiming the current
    # version but the wrong dim should still be caught.
    out_dir = tmp_path / "weird_features"
    out_dir.mkdir()
    (out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_schema_version": fs.FEATURE_SCHEMA_VERSION,
        "feature_dim": 999,
    }))
    with pytest.raises(fs.FeatureSchemaError):
        fe.check_feature_dir_compatible(out_dir)


# ---------------------------------------------------------------------------
# ensure_model_asset -- already-cached short circuit + clear failure
# ---------------------------------------------------------------------------

def test_ensure_model_asset_short_circuits_if_already_present(tmp_path, monkeypatch):
    fake_model = tmp_path / "holistic_landmarker.task"
    fake_model.write_bytes(b"not a real model, just needs to exist")

    def _fail_if_called(*a, **k):
        raise AssertionError("should not attempt a download when the file already exists")

    monkeypatch.setattr(fe.urllib.request, "urlretrieve", _fail_if_called)
    result = fe.ensure_model_asset(str(fake_model))
    assert result == str(fake_model)


def test_ensure_model_asset_raises_a_clear_error_when_download_fails(tmp_path, monkeypatch):
    missing_model = tmp_path / "subdir" / "holistic_landmarker.task"

    def _boom(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr(fe.urllib.request, "urlretrieve", _boom)
    with pytest.raises(RuntimeError, match="download"):
        fe.ensure_model_asset(str(missing_model))
