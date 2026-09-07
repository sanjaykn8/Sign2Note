"""Viterbi smoothing -- both the flat-list decoder (viterbi_decode) and the
timestamped-event decoder (viterbi_events) -- and the duplicate-collapse
behavior long-video support depends on."""
import numpy as np

from inference_viterbi import viterbi_decode, viterbi_events


def _clear_run_probs():
    """5 windows, all confidently the same class -- must collapse to ONE
    label, not five repeats."""
    return np.tile([0.9, 0.05, 0.05], (5, 1))


def test_viterbi_decode_collapses_duplicate_consecutive_predictions():
    probs = _clear_run_probs()
    result = viterbi_decode(probs, threshold=0.5)
    assert result == [0]  # not [0, 0, 0, 0, 0]


def test_viterbi_decode_drops_low_confidence_windows():
    probs = np.array([[0.3, 0.3, 0.4]] * 5)  # nothing crosses 0.5
    result = viterbi_decode(probs, threshold=0.5)
    assert result == []


def test_viterbi_decode_and_viterbi_events_agree_on_label_order():
    id2label = {0: "DEFINITION", 1: "EXAMPLE", 2: "QUESTION"}
    probs = np.array([
        [0.9, 0.05, 0.05], [0.85, 0.1, 0.05],
        [0.1, 0.85, 0.05], [0.1, 0.8, 0.1], [0.1, 0.75, 0.15],
        [0.05, 0.1, 0.85], [0.05, 0.1, 0.85], [0.05, 0.15, 0.8],
    ])
    window_times = [(i * 1.0, i * 1.0 + 2.0) for i in range(len(probs))]

    ids = viterbi_decode(probs, threshold=0.55)
    events = viterbi_events(probs, id2label, window_times, threshold=0.55)

    assert [id2label[i] for i in ids] == [e["label"] for e in events]


def test_viterbi_events_produces_ordered_nonoverlapping_timestamps():
    id2label = {0: "A", 1: "B"}
    probs = np.array([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
    window_times = [(i * 2.0, i * 2.0 + 3.0) for i in range(8)]

    events = viterbi_events(probs, id2label, window_times, threshold=0.5)
    assert len(events) == 2
    assert events[0]["label"] == "A"
    assert events[1]["label"] == "B"
    # second event must start no earlier than the first one's start
    assert events[1]["start_time"] >= events[0]["start_time"]
    for e in events:
        assert e["end_time"] >= e["start_time"]


def test_viterbi_events_empty_input():
    assert viterbi_events(np.zeros((0, 3)), {}, []) == []


def test_viterbi_decode_empty_input():
    assert viterbi_decode(np.zeros((0, 3))) == []


def test_viterbi_events_confidence_is_mean_of_run_not_single_window():
    id2label = {0: "A", 1: "B"}
    # class A's confidence varies across 3 consecutive windows (0.9, 0.7, 0.8)
    # while a low, constant probability mass sits on the other class -- a
    # meaningful two-class distribution, unlike a single-column array (which
    # trivially normalizes to 1.0 regardless of the raw value).
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.8, 0.2]])
    window_times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    events = viterbi_events(probs, id2label, window_times, threshold=0.5)
    assert len(events) == 1
    assert abs(events[0]["confidence"] - np.mean([0.9, 0.7, 0.8])) < 1e-6
