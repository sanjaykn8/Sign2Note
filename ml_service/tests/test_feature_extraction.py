"""Keypoint vector construction -- feature_extraction.py's
_extract_keypoints(). Uses a fake MediaPipe result object rather than a
real video, so this runs without a camera/video file and without
depending on mediapipe's hand-detection actually finding anything."""
import numpy as np

from feature_extraction import _extract_keypoints


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeHand:
    def __init__(self, seed):
        self.landmark = [_FakeLandmark(seed + i * 0.01, seed + i * 0.02, seed + i * 0.03) for i in range(21)]


class _FakeResults:
    def __init__(self, hands):
        self.multi_hand_landmarks = hands if hands else None


def test_no_hands_returns_zero_vector():
    vec = _extract_keypoints(_FakeResults([]))
    assert vec.shape == (126,)
    assert np.all(vec == 0)


def test_one_hand_fills_first_63_only():
    vec = _extract_keypoints(_FakeResults([_FakeHand(1.0)]))
    assert vec.shape == (126,)
    assert not np.all(vec[:63] == 0)
    assert np.all(vec[63:] == 0)


def test_two_hands_fill_both_halves_in_detection_order():
    vec = _extract_keypoints(_FakeResults([_FakeHand(1.0), _FakeHand(2.0)]))
    assert vec[0] == 1.0    # first detected hand's first landmark x
    assert vec[63] == 2.0   # second detected hand's first landmark x


def test_output_dtype_is_float32():
    vec = _extract_keypoints(_FakeResults([_FakeHand(1.0)]))
    assert vec.dtype == np.float32
