# inference_viterbi.py
import numpy as np

def viterbi_decode(frame_logits, transition_logprob=None, blank_idx=None, top_threshold=0.5):
    # frame_logits: (T, C) probabilities or log-probs (use log softmax)
    import math
    logp = np.log(frame_logits + 1e-12)
    T, C = logp.shape
    if transition_logprob is None:
        # prefer staying in same class; simple transition prior:
        transition_logprob = np.log(np.eye(C)*0.9 + (1.0/ C)*0.1)
    dp = np.full((T, C), -1e9)
    ptr = np.zeros((T, C), dtype=int)
    dp[0] = logp[0]
    for t in range(1, T):
        for c in range(C):
            scores = dp[t-1] + transition_logprob[:, c] + logp[t, c]
            ptr[t, c] = np.argmax(scores)
            dp[t, c] = scores[ptr[t, c]]
    seq = []
    last = np.argmax(dp[-1])
    for t in range(T-1, -1, -1):
        seq.append(last)
        last = ptr[t, last]
    seq = list(reversed(seq))
    # collapse repeats and remove low-prob frames
    out = []
    prev = None
    for t, c in enumerate(seq):
        prob = frame_logits[t, c]
        if prob < top_threshold: continue
        if c != prev:
            out.append(c)
            prev = c
    return out
