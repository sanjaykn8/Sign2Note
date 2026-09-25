"""Tests for feature_schema.py. Deliberately torch-free: this module only
depends on numpy, so these tests run even in environments where PyTorch
can't be installed (e.g. disk-constrained CI containers)."""

import numpy as np
import pytest

import feature_schema as fs


class FakeLM:
    """Stand-in for mediapipe's NormalizedLandmark -- just needs .x/.y/.z
    (and .visibility for pose) to match what feature_schema.py reads."""

    def __init__(self, x, y, z, visibility=None):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def hand_landmarks(seed):
    """21 fake hand landmarks, each with a distinct, recoverable value."""
    return [FakeLM(seed + i, seed + i + 0.1, seed + i + 0.2) for i in range(21)]


def full_pose_landmarks():
    """All 33 pose landmarks; value = index itself, so a test can verify
    exactly which 9 indices got pulled into the feature vector."""
    return [FakeLM(float(i), float(i) * 10, float(i) * 100, visibility=float(i) / 32) for i in range(33)]


def full_face_landmarks():
    """All 468 face-mesh landmarks; value = index itself, same purpose."""
    return [FakeLM(float(i), float(i) * 10, float(i) * 100) for i in range(468)]


# ---------------------------------------------------------------------------
# Dimension bookkeeping
# ---------------------------------------------------------------------------

def test_feature_dim_matches_group_sums():
    assert fs.HAND_FEATURE_DIM == 63
    assert fs.POSE_FEATURE_DIM == 36
    assert fs.FACE_FEATURE_DIM == 123
    assert fs.FEATURE_DIM == 63 + 63 + 36 + 123 == 285


def test_feature_groups_cover_the_whole_vector_with_no_gaps_or_overlap():
    covered = np.zeros(fs.FEATURE_DIM, dtype=bool)
    for _name, start, end in fs.FEATURE_GROUPS:
        assert not covered[start:end].any(), "overlapping feature group ranges"
        covered[start:end] = True
    assert covered.all(), "feature groups leave a gap in the vector"


def test_pose_and_face_index_lists_have_no_duplicates():
    assert len(set(fs.POSE_LANDMARK_INDICES)) == len(fs.POSE_LANDMARK_INDICES)
    assert len(set(fs.FACE_LANDMARK_INDICES)) == len(fs.FACE_LANDMARK_INDICES)


def test_face_landmark_names_and_indices_stay_in_sync_with_the_grouped_dict():
    flat_names = [n for g in fs.FACE_LANDMARK_GROUPS.values() for n in g]
    flat_indices = [i for g in fs.FACE_LANDMARK_GROUPS.values() for i in g.values()]
    assert flat_names == fs.FACE_LANDMARK_NAMES
    assert flat_indices == fs.FACE_LANDMARK_INDICES


# ---------------------------------------------------------------------------
# build_feature_vector -- missing-landmark behavior (both hands / one hand /
# no hands / pose present / face present / missing landmarks)
# ---------------------------------------------------------------------------

def test_both_hands_present():
    vec = fs.build_feature_vector(hand_landmarks(0), hand_landmarks(100), None, None)
    assert vec.shape == (fs.FEATURE_DIM,)
    assert vec[0] == pytest.approx(0.0)  # left wrist x
    assert vec[fs.HAND_FEATURE_DIM] == pytest.approx(100.0)  # right wrist x
    assert (vec[2 * fs.HAND_FEATURE_DIM:] == 0).all()  # pose+face untouched


def test_one_hand_missing_is_deterministic_zero_not_a_shift():
    only_right = fs.build_feature_vector(None, hand_landmarks(100), None, None)
    assert (only_right[: fs.HAND_FEATURE_DIM] == 0).all(), "missing left hand must be zeros, not shifted-in right-hand data"
    assert only_right[fs.HAND_FEATURE_DIM] == pytest.approx(100.0)


def test_no_hands_no_pose_no_face_is_all_zero_and_correct_length():
    vec = fs.build_feature_vector(None, None, None, None)
    assert vec.shape == (fs.FEATURE_DIM,)
    assert (vec == 0).all()


def test_empty_list_treated_same_as_none():
    vec_none = fs.build_feature_vector(None, None, None, None)
    vec_empty = fs.build_feature_vector([], [], [], [])
    assert np.array_equal(vec_none, vec_empty)


def test_pose_pulls_exactly_the_documented_9_indices_in_order():
    vec = fs.build_feature_vector(None, None, full_pose_landmarks(), None)
    pose_start = fs.FEATURE_GROUPS[2][1]
    for slot, src_idx in enumerate(fs.POSE_LANDMARK_INDICES):
        base = pose_start + slot * fs.POSE_DIMS_PER_LANDMARK
        assert vec[base + 0] == pytest.approx(float(src_idx))  # x == landmark's own index
        assert vec[base + 3] == pytest.approx(float(src_idx) / 32)  # visibility carried through


def test_pose_visibility_included_but_hand_and_face_do_not_carry_visibility():
    # Hands: 3 dims/landmark (x,y,z) -- HAND_FEATURE_DIM already encodes this.
    assert fs.HAND_DIMS_PER_LANDMARK == 3
    # Face: 3 dims/landmark -- same.
    assert fs.FACE_DIMS_PER_LANDMARK == 3
    # Pose: 4 dims/landmark (x,y,z,visibility).
    assert fs.POSE_DIMS_PER_LANDMARK == 4


def test_face_pulls_exactly_the_documented_41_indices_in_order():
    vec = fs.build_feature_vector(None, None, None, full_face_landmarks())
    face_start = fs.FEATURE_GROUPS[3][1]
    assert len(fs.FACE_LANDMARK_INDICES) == 41
    for slot, src_idx in enumerate(fs.FACE_LANDMARK_INDICES):
        base = face_start + slot * fs.FACE_DIMS_PER_LANDMARK
        assert vec[base + 0] == pytest.approx(float(src_idx))


def test_out_of_range_landmark_index_degrades_to_zero_not_a_crash():
    # A pose result with fewer landmarks than POSE_LANDMARK_INDICES expects
    # (e.g. a truncated/corrupt detection) must not raise -- missing slots
    # stay zero.
    short_pose = [FakeLM(1.0, 1.0, 1.0, visibility=1.0)]  # only landmark 0
    vec = fs.build_feature_vector(None, None, short_pose, None)
    pose_start = fs.FEATURE_GROUPS[2][1]
    assert vec[pose_start] == pytest.approx(1.0)  # NOSE (index 0) present
    assert vec[pose_start + 4] == pytest.approx(0.0)  # LEFT_SHOULDER (index 11) -> zero, not IndexError


# ---------------------------------------------------------------------------
# assert_feature_dim -- fail clearly instead of a downstream shape crash
# ---------------------------------------------------------------------------

def test_assert_feature_dim_accepts_current_schema():
    fs.assert_feature_dim(fs.FEATURE_DIM)  # should not raise


def test_assert_feature_dim_rejects_v1_hand_only_dim_with_a_helpful_hint():
    with pytest.raises(fs.FeatureSchemaError, match="126"):
        fs.assert_feature_dim(126, context="some_video.npy")


def test_assert_feature_dim_rejects_arbitrary_wrong_dim():
    with pytest.raises(fs.FeatureSchemaError):
        fs.assert_feature_dim(999)


# ---------------------------------------------------------------------------
# normalize_sequence -- z-score coordinates, pass through pose visibility
# ---------------------------------------------------------------------------

def test_normalize_sequence_zscores_coordinate_dims():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(50, fs.FEATURE_DIM)).astype(np.float32)
    z = fs.normalize_sequence(x)
    coord_dims = [i for i in range(fs.FEATURE_DIM) if i not in fs.POSE_VISIBILITY_INDICES]
    assert np.abs(z[:, coord_dims].mean(axis=0)).max() < 1e-3
    assert np.abs(z[:, coord_dims].std(axis=0) - 1.0).max() < 1e-2


def test_normalize_sequence_does_not_zscore_pose_visibility():
    # Construct a sequence where a visibility channel is CONSTANT (the
    # exact case that made z-scoring dangerous: std -> 0).
    x = np.random.default_rng(1).normal(size=(30, fs.FEATURE_DIM)).astype(np.float32)
    vis_idx = fs.POSE_VISIBILITY_INDICES[0]
    x[:, vis_idx] = 0.97  # constant "mostly visible" value
    z = fs.normalize_sequence(x)
    # If this were z-scored like a coordinate, a constant channel would
    # collapse toward 0 (mean-subtracted) and (0/1e-5) noise would dominate.
    # Passed-through-and-clipped, it should just stay 0.97.
    assert np.allclose(z[:, vis_idx], 0.97)


def test_normalize_sequence_clips_out_of_range_visibility_defensively():
    x = np.zeros((5, fs.FEATURE_DIM), dtype=np.float32)
    x[:, fs.POSE_VISIBILITY_INDICES[0]] = 1.4  # shouldn't happen, but be defensive
    z = fs.normalize_sequence(x)
    assert z[:, fs.POSE_VISIBILITY_INDICES[0]].max() <= 1.0


# ---------------------------------------------------------------------------
# prepare_window -- the train/inference consolidation fix
# ---------------------------------------------------------------------------

def test_prepare_window_pads_short_sequences_to_max_len():
    x = np.ones((5, fs.FEATURE_DIM), dtype=np.float32)
    out = fs.prepare_window(x, max_len=10)
    assert out.shape == (10, fs.FEATURE_DIM)
    # Padding participates in normalization (matching dataset.py/infer.py's
    # existing pad-then-normalize order -- see the test below), so the
    # padded tail is NOT raw zero; it's whatever (0 - mean) / std comes out
    # to. It should, however, be identical across the padded rows (all
    # zeros in, same z-score out) and different from the real rows.
    assert np.allclose(out[5], out[6])
    assert not np.allclose(out[0], out[5])


def test_prepare_window_trims_long_sequences_to_max_len():
    x = np.arange(20 * fs.FEATURE_DIM, dtype=np.float32).reshape(20, fs.FEATURE_DIM)
    out = fs.prepare_window(x, max_len=10)
    assert out.shape == (10, fs.FEATURE_DIM)


def test_prepare_window_normalizes_over_the_padded_array_matching_dataset_py_and_infer_py():
    # This is intentionally checking the EXISTING (already correct, already
    # consistent) behavior of dataset.py's SignDataset.__getitem__ and
    # infer.py's _window_batch short-clip path: pad_or_trim() FIRST, THEN
    # normalize_sequence() over the padded array -- so padding zeros DO
    # participate in the mean/std for short sequences. prepare_window()
    # must match that exactly, not "improve" on it.
    real = np.array([[10.0] * fs.FEATURE_DIM, [12.0] * fs.FEATURE_DIM, [14.0] * fs.FEATURE_DIM], dtype=np.float32)
    out = fs.prepare_window(real, max_len=10)

    expected = fs.normalize_sequence(fs.pad_or_trim(real, 10))
    assert np.allclose(out, expected)
    assert out.shape == (10, fs.FEATURE_DIM)

    # Sanity check the actual numbers for one plain coordinate dimension
    # (not a pose-visibility dim): 3 real frames of value 10/12/14 plus 7
    # zero-padded frames -> mean and std computed over all 10 values.
    coord_dim = 0
    values = np.array([10.0, 12.0, 14.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    mean, std = values.mean(), values.std() + 1e-5
    assert np.allclose(out[:, coord_dim], (values - mean) / std, atol=1e-4)
