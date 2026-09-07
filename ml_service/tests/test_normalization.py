"""Keypoint window normalization and padding -- infer.py's _normalize(),
_pad_window(), and _window_batch()."""
import numpy as np
import pytest

from infer import _normalize, _pad_window, _window_batch


def test_normalize_zero_mean_per_dim():
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    out = _normalize(x)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-5)


def test_normalize_handles_zero_variance_dim():
    # a feature dimension that's perfectly constant across time must not
    # produce NaN/inf (the +1e-5 epsilon on std exists exactly for this)
    x = np.full((5, 3), 7.0, dtype=np.float32)
    out = _normalize(x)
    assert np.all(np.isfinite(out))


def test_pad_window_pads_at_end_with_zeros():
    x = np.ones((3, 2), dtype=np.float32)
    padded = _pad_window(x, 5)
    assert padded.shape == (5, 2)
    assert np.allclose(padded[:3], 1.0)
    assert np.allclose(padded[3:], 0.0)


def test_pad_window_truncates_when_too_long():
    x = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    truncated = _pad_window(x, 4)
    assert truncated.shape == (4, 2)
    assert np.array_equal(truncated, x[:4])


def test_window_batch_short_clip_produces_one_padded_window():
    x = np.random.rand(5, 126).astype(np.float32)
    batch, spans = _window_batch(x, max_len=10, stride=4)
    assert batch.shape == (1, 10, 126)
    assert spans == [(0, 5)]


def test_window_batch_long_clip_produces_overlapping_windows():
    x = np.random.rand(50, 126).astype(np.float32)
    batch, spans = _window_batch(x, max_len=10, stride=4)
    assert batch.shape[1:] == (10, 126)
    assert batch.shape[0] == len(spans)
    assert batch.shape[0] > 1
    # spans must be non-decreasing and cover up to the end of the clip
    assert spans[-1][1] == 50


def test_window_batch_normalizes_each_window_independently():
    """Regression test for the bug found while building long-video support:
    each window must be normalized on its OWN statistics, not the whole
    clip's. Two windows drawn from very different scale regions of the
    same clip should each end up zero-mean after normalization -- which
    would NOT be true if normalization happened once globally beforehand."""
    low = np.random.rand(30, 4).astype(np.float32) * 1.0
    high = np.random.rand(30, 4).astype(np.float32) * 100.0 + 500.0
    x = np.concatenate([low, high], axis=0)

    batch, spans = _window_batch(x, max_len=30, stride=30)
    assert batch.shape[0] == 2
    for window in batch:
        assert np.allclose(window.mean(axis=0), 0, atol=1e-3)
