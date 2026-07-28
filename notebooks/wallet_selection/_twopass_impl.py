"""Two-pass sliding window scoring for leader-follower implied trades.

Pass 1: For each leader trade, find all followers in [lt-tw, lt) → leader scores.
Pass 2: For each follower trade, find all leaders in [ft-tw, ft) → follower scores.

Returns per-leader and per-follower aggregates without materializing pairs.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _pass1_leader_scores(fi_s, li_s, fw_ids, cp_sorted, tw_ns):
    """Sliding window: for each leader trade, sum follower cpnl in [lt-tw, lt).

    Returns per-leader-trade scores and distinct follower counts.
    """
    n_f = len(fi_s)
    n_l = len(li_s)
    scores = np.empty(n_l, dtype=np.float64)
    distinct = np.empty(n_l, dtype=np.int64)
    f_start = 0
    f_end = 0
    for li_idx in range(n_l):
        lt_time = li_s[li_idx]
        lt_lo = lt_time - tw_ns
        while f_start < n_f and fi_s[f_start] < lt_lo:
            f_start += 1
        while f_end < n_f and fi_s[f_end] < lt_time:
            f_end += 1
        total = 0.0
        for k in range(f_start, f_end):
            total += cp_sorted[k]
        scores[li_idx] = total
        nd = 0
        for k in range(f_start, f_end):
            w = fw_ids[k]
            found = False
            for m in range(f_start, k):
                if fw_ids[m] == w:
                    found = True
                    break
            if not found:
                nd += 1
        distinct[li_idx] = nd
    return scores, distinct


@njit(cache=True, nogil=True)
def _pass2_follower_scores(fi_s, li_s, lw_ids, lt_scores, tw_ns):
    """Sliding window: for each follower trade, sum leader scores in [ft-tw, ft).

    Returns per-follower-trade scores and distinct leader counts.
    """
    n_f = len(fi_s)
    n_l = len(li_s)
    lt_prefix = np.empty(n_l + 1, dtype=np.float64)
    lt_prefix[0] = 0.0
    for i in range(n_l):
        lt_prefix[i + 1] = lt_prefix[i] + lt_scores[i]
    scores = np.empty(n_f, dtype=np.float64)
    distinct = np.empty(n_f, dtype=np.int64)
    l_start = 0
    l_end = 0
    for fi_idx in range(n_f):
        ft_time = fi_s[fi_idx]
        ft_lo = ft_time - tw_ns
        while l_start < n_l and li_s[l_start] < ft_lo:
            l_start += 1
        while l_end < n_l and li_s[l_end] < ft_time:
            l_end += 1
        scores[fi_idx] = lt_prefix[l_end] - lt_prefix[l_start]
        nd = 0
        for k in range(l_start, l_end):
            w = lw_ids[k]
            found = False
            for m in range(l_start, k):
                if lw_ids[m] == w:
                    found = True
                    break
            if not found:
                nd += 1
        distinct[fi_idx] = nd
    return scores, distinct


def _warmup():
    fi = np.array([0, 100, 200], dtype=np.int64)
    li = np.array([50, 150], dtype=np.int64)
    fw = np.array([0, 1, 0], dtype=np.int32)
    cp = np.array([1.0, 2.0, 3.0])
    lw = np.array([0, 1], dtype=np.int32)
    _pass1_leader_scores(fi, li, fw, cp, 100000000)
    _pass2_follower_scores(fi, li, lw, np.array([1.0, 2.0]), 100000000)
