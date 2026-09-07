import numpy as np


def _viterbi_path(probs: np.ndarray, stay_prob: float = 0.92) -> np.ndarray:
    """Core Viterbi DP shared by viterbi_decode() and viterbi_events():
    returns the raw per-timestep most-likely state sequence (T,), with NO
    threshold/collapse applied yet. A simple "stay in current state with
    probability stay_prob, otherwise switch uniformly to any other state"
    transition model is what gives the smoothing its temporal stickiness
    (it discourages flipping labels window-to-window)."""
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    T, C = probs.shape

    switch = (1.0 - stay_prob) / max(C - 1, 1)
    trans = np.full((C, C), switch, dtype=np.float64)
    np.fill_diagonal(trans, stay_prob)
    log_trans = np.log(np.clip(trans, 1e-9, 1.0))
    logp = np.log(probs)

    dp = np.full((T, C), -1e30)
    ptr = np.zeros((T, C), dtype=np.int32)
    dp[0] = logp[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + log_trans
        ptr[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + logp[t]

    state = int(dp[-1].argmax())
    path = [state]
    for t in range(T - 1, 0, -1):
        state = int(ptr[t, state])
        path.append(state)
    path.reverse()
    return np.array(path, dtype=np.int32)


def viterbi_decode(window_probs: np.ndarray, stay_prob: float = 0.92,
                   threshold: float = 0.55, blank_idx: int | None = None):
    """Decode overlapping window probabilities into a smoothed gloss sequence
    (label ids only, duplicates collapsed, no timing). Unchanged behavior
    from the original implementation -- kept for any caller that only
    wants the flat id list."""
    probs = np.asarray(window_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return []
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    path = _viterbi_path(probs, stay_prob)

    result = []
    prev = None
    for t, c in enumerate(path):
        c = int(c)
        confidence = float(probs[t, c])
        if confidence < threshold or (blank_idx is not None and c == blank_idx):
            continue
        if c != prev:
            result.append(c)
            prev = c
    return result


def viterbi_events(window_probs: np.ndarray, id2label: dict,
                   window_times: list, stay_prob: float = 0.92,
                   threshold: float = 0.55, blank_idx: int | None = None):
    """Like viterbi_decode(), but returns timestamped events instead of a
    flat id list -- this is what powers "long video" mode's ordered gloss
    sequence with start/end timestamps.

    `window_times[i]` must be the (start_seconds, end_seconds) span that
    window i covers in the original video. Consecutive windows assigned
    the same smoothed label are merged into a single event spanning from
    the first window's start to the last window's end, with `confidence`
    set to the mean of the per-window confidences in that run (a run's
    mean is a steadier signal than any single window's peak).

    Returns a list of dicts: {label, confidence, start_time, end_time},
    in chronological order. Windows below `threshold` are dropped (same
    semantics as viterbi_decode) -- they simply don't start/extend a run.
    """
    probs = np.asarray(window_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return []
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    path = _viterbi_path(probs, stay_prob)

    events = []
    run_label = None
    run_start = None
    run_end = None
    run_confidences = []

    def _flush():
        if run_label is not None and run_confidences:
            events.append({
                "label": run_label,
                "confidence": float(np.mean(run_confidences)),
                "start_time": round(float(run_start), 2),
                "end_time": round(float(run_end), 2),
            })

    for t, c in enumerate(path):
        c = int(c)
        confidence = float(probs[t, c])
        accepted = confidence >= threshold and not (blank_idx is not None and c == blank_idx)
        label = id2label.get(c, str(c)) if accepted else None

        if label != run_label:
            _flush()
            run_label = label
            run_start = window_times[t][0]
            run_confidences = []
        if accepted:
            run_confidences.append(confidence)
            run_end = window_times[t][1]

    _flush()
    return events
