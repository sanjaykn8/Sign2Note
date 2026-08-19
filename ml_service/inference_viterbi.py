import numpy as np


def viterbi_decode(window_probs: np.ndarray, stay_prob: float = 0.92,
                   threshold: float = 0.55, blank_idx: int | None = None):
    """Decode overlapping window probabilities into a smoothed gloss sequence."""
    probs = np.asarray(window_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return []
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

    result = []
    prev = None
    for t, c in enumerate(path):
        confidence = float(probs[t, c])
        if confidence < threshold or (blank_idx is not None and c == blank_idx):
            continue
        if c != prev:
            result.append(c)
            prev = c
    return result
