"""
Signal quality framework, combination, and strategy evaluation helpers
for the stage1 wallet-selection notebooks.

Implements the Grinold & Kahn (1999) *Active Portfolio Management* toolkit:
IC, IR, bootstrap CI, hit rate, overlap analysis, and Markowitz signal
combination with shrinkage.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rank correlation (numpy-only Spearman)
# ---------------------------------------------------------------------------


def _rankdata(v):
    """Fractional ranking (scipy.stats.rankdata, method='average')."""
    n = len(v)
    sorter = np.argsort(v, kind="mergesort")
    ordinal = np.empty(n, dtype=np.intp)
    ordinal[sorter] = np.arange(n)
    rank = ordinal + 1.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and v[sorter[j]] == v[sorter[i]]:
            j += 1
        if j > i + 1:
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                rank[sorter[k]] = avg_rank
        i = j
    return rank


def spearman_rho(x, y):
    """Spearman rank correlation (numpy-only)."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 10:
        return np.nan
    rx = _rankdata(x[mask].values if hasattr(x, 'values') else x[mask])
    ry = _rankdata(y[mask].values if hasattr(y, 'values') else y[mask])
    rx_m = rx.mean()
    ry_m = ry.mean()
    num = np.sum((rx - rx_m) * (ry - ry_m))
    den = np.sqrt(np.sum((rx - rx_m) ** 2) * np.sum((ry - ry_m) ** 2))
    return num / den if den != 0 else np.nan


def compute_event_ic(signal, forward_roi):
    """IC: Spearman rank correlation between signal and forward copyable ROI."""
    return spearman_rho(signal, forward_roi)


# ---------------------------------------------------------------------------
# Price-residualized ROI evaluation
# ---------------------------------------------------------------------------


def rank_scores(x):
    """Van der Waerden scores: Phi^-1((rank - 0.5) / n).

    Maps fractional ranks to standard-normal scores, robust to degenerate /
    outlier-heavy forward metrics (e.g. the mass of copyable_roi at -1.0).
    """
    r = _rankdata(np.asarray(x, dtype=float))
    n = len(r)
    p = (r - 0.5) / n
    p = np.clip(p, 1e-6, 1 - 1e-6)
    norm = NormalDist()
    return np.fromiter((norm.inv_cdf(float(v)) for v in p), dtype=float, count=len(p))


def fit_roi_residualizer(roi_train, price_train):
    """OLS of rank-normalized ROI on rank-normalized price (train only).

    Returns dict with slope/intercept (linear-in-ranks price adjustment).
    """
    y = rank_scores(roi_train)
    x = rank_scores(price_train)
    xm, ym = x.mean(), y.mean()
    beta = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    intercept = ym - beta * xm
    return {'beta': float(beta), 'intercept': float(intercept)}


def residualized_roi(roi, price, fit):
    """Rank-normalized ROI minus the price component (train-fit coefficients)."""
    return rank_scores(roi) - fit['beta'] * rank_scores(price) - fit['intercept']


# ---------------------------------------------------------------------------
# IC / IR / bootstrap / hit-rate
# ---------------------------------------------------------------------------


def compute_event_ir(signal, forward_roi, timestamps, freq="D"):
    """IR = mean(IC_chunk) / std(IC_chunk) across time chunks.

    Higher IR means predictive power is consistent (Grinold & Kahn Ch. 7).
    """
    chunks = pd.Series(index=pd.DatetimeIndex(timestamps), data=np.arange(len(signal))).groupby(
        pd.Grouper(freq=freq)
    )
    ics = []
    for _, idx in chunks:
        if len(idx) < 5:
            continue
        rho = compute_event_ic(signal.iloc[idx], forward_roi.iloc[idx])
        if not np.isnan(rho):
            ics.append(rho)
    if len(ics) < 3:
        return np.nan
    arr = np.array(ics)
    return float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else np.nan


def _bootstrap_ic_core(x, y, n_iter, alpha, seed):
    """Vectorized fixed-rank bootstrap for Spearman IC.

    Ranks are computed once on the full sample; each bootstrap draw resamples
    the rank-transformed pairs and takes their Pearson correlation (= Spearman
    on the resample, exact up to tie rounding on duplicates).  Much faster than
    a per-draw rank recomputation for the same bootstrap distribution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 10:
        return np.nan, np.nan, np.nan
    x, y = x[mask], y[mask]
    rx = _rankdata(x) - np.mean(_rankdata(x))
    ry = _rankdata(y) - np.mean(_rankdata(y))
    nx = np.sqrt(np.sum(rx ** 2))
    ny = np.sqrt(np.sum(ry ** 2))
    if nx == 0 or ny == 0:
        return np.nan, np.nan, np.nan
    rx /= nx
    ry /= ny
    rng = np.random.default_rng(seed)
    boot_ics = np.empty(n_iter)
    block = 64
    for b0 in range(0, n_iter, block):
        b1 = min(b0 + block, n_iter)
        idx = rng.integers(0, n, size=(b1 - b0, n))
        boot_ics[b0:b1] = np.einsum("ij,ij->i", rx[idx], ry[idx])
    mean_ic = float(np.mean(boot_ics))
    ci_lo = float(np.percentile(boot_ics, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_ics, 100 * (1 - alpha / 2)))
    return mean_ic, ci_lo, ci_hi


def bootstrap_ic(signal, forward_roi, n_iter=10_000, alpha=0.05, seed=42):
    """Bootstrap CI for IC (Efron & Tibshirani 1993), vectorized.

    Returns (mean_ic, ci_lower, ci_upper).
    """
    return _bootstrap_ic_core(signal, forward_roi, n_iter, alpha, seed)


def hit_rate(signal, forward_roi):
    """Fraction of *active* events where signal sign matches PnL sign.

    Events with no position (signal == 0) are excluded: they carry no bet, so
    they must not be scored as misses.
    """
    mask = (signal.notna() & forward_roi.notna()
            & (forward_roi != 0) & (signal != 0))
    if mask.sum() < 10:
        return np.nan
    sgn_sig = np.sign(signal[mask])
    sgn_pnl = np.sign(forward_roi[mask])
    return float((sgn_sig == sgn_pnl).mean())


def signal_quality_report(signals_df, signal_cols, roi_col="copyable_roi",
                           dt_col="dt", ir_freq="D", n_bootstrap=5_000,
                           bootstrap=False):
    """Compute IC, IR, hit rate, and (optionally) bootstrap CI per signal.

    Returns a DataFrame with one row per signal, sorted by |IC|.
    """
    print(f"Signal quality report: {len(signal_cols)} signals, {len(signals_df)} events")
    if not signal_cols:
        return pd.DataFrame(columns=[
            "signal", "IC", "IR", "hit_rate", "bootstrap_mean_ic",
            "bootstrap_ci_lo", "bootstrap_ci_hi", "n_events",
        ])
    rows = []
    for col in signal_cols:
        ic = compute_event_ic(signals_df[col], signals_df[roi_col])
        ir = compute_event_ir(signals_df[col], signals_df[roi_col],
                               signals_df[dt_col], freq=ir_freq)
        hr = hit_rate(signals_df[col], signals_df[roi_col])

        if bootstrap:
            m, clo, chi = bootstrap_ic(signals_df[col], signals_df[roi_col],
                                       n_iter=n_bootstrap)
        else:
            m, clo, chi = -1, -1, -1
        rows.append({
            "signal": col,
            "IC": ic,
            "IR": ir,
            "hit_rate": hr,
            "bootstrap_mean_ic": m,
            "bootstrap_ci_lo": clo,
            "bootstrap_ci_hi": chi,
            "n_events": int(signals_df[col].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("IC", ascending=False, key=abs)


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------


def coincidence_rate(s1, s2):
    """Jaccard-like coincidence: P(both non-zero | either non-zero)."""
    both = ((s1.notna() & (s1 != 0)) & (s2.notna() & (s2 != 0))).sum()
    either = ((s1.notna() & (s1 != 0)) | (s2.notna() & (s2 != 0))).sum()
    return both / either if either > 0 else 0.0


def ic_correlation_matrix(signals_df, signal_cols, roi_col="copyable_roi"):
    """Pairwise IC of signal values on overlapping events."""
    n = len(signal_cols)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            both = signals_df[signal_cols[i]].notna() & signals_df[signal_cols[j]].notna()
            if both.sum() < 10:
                continue
            mat[i, j] = compute_event_ic(
                signals_df.loc[both, signal_cols[i]],
                signals_df.loc[both, signal_cols[j]],
            )
    return pd.DataFrame(mat, index=signal_cols, columns=signal_cols)


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------


def compute_optimal_weights(signals_df, signal_cols, roi_col="copyable_roi",
                            shrinkage=0.5):
    """Markowitz-optimal signal weights with shrinkage (Grinold & Kahn Ch. 13).

    w = (1 - lambda) * inv(Sigma) * IC + lambda * (1/n)

    Parameters
    ----------
    shrinkage : float
        0 = full Markowitz, 1 = equal weight.
    """
    n = len(signal_cols)
    ic_vec = np.array([
        compute_event_ic(signals_df[c], signals_df[roi_col]) or 0.0
        for c in signal_cols
    ])

    valid = signals_df[signal_cols].notna().all(axis=1)
    if valid.sum() < 10 or n <= 1:
        return pd.Series(np.ones(n) / n, index=signal_cols)

    sig_vals = signals_df.loc[valid, signal_cols].values
    cov = np.cov(sig_vals, rowvar=False)
    avg_var = np.trace(cov) / n
    shrunk_cov = (1 - shrinkage) * cov + shrinkage * np.eye(n) * avg_var

    try:
        inv_cov = np.linalg.solve(shrunk_cov, np.eye(n))
        w = inv_cov @ ic_vec
        w_abs_sum = np.sum(np.abs(w))
        if w_abs_sum > 1e-12:
            w = w / w_abs_sum
        else:
            w = np.ones(n) / n
    except np.linalg.LinAlgError:
        w = np.ones(n) / n

    return pd.Series(w, index=signal_cols)


def apply_composite_score(signals_df, signal_cols, weights):
    """Composite signal = sum w_i * signal_i."""
    result = np.zeros(len(signals_df))
    for col in signal_cols:
        result += weights[col] * signals_df[col].fillna(0.0).values
    return pd.Series(result, index=signals_df.index)


def cs_rank(s, grouper=None):
    """Cross-sectional rank transform. Maps values to [-1, 1] within groups.

    If grouper is provided, ranks within each group independently.
    Standard Grinold & Kahn normalization.
    """
    if grouper is not None:
        result = s.groupby(grouper, sort=False).transform(
            lambda x: 2.0 * (_rankdata(x.values) - 1.0) / max(len(x) - 1, 1) - 1.0
        )
    else:
        n = len(s)
        r = _rankdata(s.values) if hasattr(s, 'values') else _rankdata(np.asarray(s))
        result = 2.0 * (r - 1.0) / max(n - 1, 1) - 1.0
    if hasattr(s, 'index'):
        return pd.Series(result, index=s.index)
    return result


def fit_rank_transformer(s):
    """Fit a train-only rank normalizer that can be applied to new splits.

    Unlike :func:`cs_rank`, this transformer does not use the target split's
    own distribution. That keeps validation/test scores independent of other
    rows in those splits.
    """
    arr = np.asarray(s.values if hasattr(s, "values") else s, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"sorted_values": np.array([], dtype=float)}
    return {"sorted_values": np.sort(arr, kind="mergesort")}


def apply_rank_transformer(s, fit):
    """Apply a train-fitted rank normalizer, mapping to the same [-1, 1] scale."""
    arr = np.asarray(s.values if hasattr(s, "values") else s, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    sorted_values = np.asarray(fit.get("sorted_values", ()), dtype=float)
    if len(sorted_values) == 0:
        return pd.Series(out, index=s.index if hasattr(s, "index") else None)

    mask = np.isfinite(arr)
    vals = arr[mask]
    left = np.searchsorted(sorted_values, vals, side="left")
    right = np.searchsorted(sorted_values, vals, side="right")
    rank = (left + right + 1.0) / 2.0
    if len(sorted_values) == 1:
        scaled = np.zeros(len(vals), dtype=float)
    else:
        scaled = 2.0 * (rank - 1.0) / (len(sorted_values) - 1.0) - 1.0
        scaled = np.clip(scaled, -1.0, 1.0)
    out[mask] = scaled
    return pd.Series(out, index=s.index if hasattr(s, "index") else None)


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


def evaluate_strategy(df, score_col, threshold, cost_bps=0.0):
    """Copy trades with composite_score >= threshold; return summary dict.

    ``cost_bps`` is applied to ``copyable_notional`` so gross and net results
    can be compared explicitly.
    """
    fired = df[df[score_col] >= threshold].copy()
    if fired.empty:
        return {
            'threshold': threshold, 'trades': 0,
            'copyable_pnl': 0.0, 'copyable_roi': 0.0,
            'copyable_pnl_net': 0.0, 'copyable_roi_net': 0.0,
            'total_pnl': 0.0, 'notional': 0.0,
            'copyable_notional': 0.0, 'firing_rate': 0.0,
            'cost_paid': 0.0,
        }
    cnot = fired['copyable_notional'].sum()
    gross_copyable_pnl = float(fired['copyable_pnl'].sum())
    cost_paid = float(cnot * cost_bps / 10_000.0)
    net_copyable_pnl = gross_copyable_pnl - cost_paid
    return {
        'threshold': threshold,
        'trades': len(fired),
        'copyable_pnl': gross_copyable_pnl,
        'copyable_roi': gross_copyable_pnl / cnot if cnot > 0 else 0.0,
        'copyable_pnl_net': net_copyable_pnl,
        'copyable_roi_net': net_copyable_pnl / cnot if cnot > 0 else 0.0,
        'total_pnl': float(fired['pnl'].sum()),
        'notional': float(fired['notional'].sum()),
        'copyable_notional': float(cnot),
        'firing_rate': len(fired) / len(df),
        'cost_paid': cost_paid,
    }


def tune_threshold(df_val, score_col, min_trades=20, step=0.05):
    """Grid-search the score threshold on validation (max copyable_pnl)."""
    res = [evaluate_strategy(df_val, score_col, t) for t in np.arange(0.0, 1.05, step)]
    rdf = pd.DataFrame(res)
    cand = rdf[rdf['trades'] >= min_trades]
    row = (cand if not cand.empty else rdf).sort_values('copyable_pnl', ascending=False).iloc[0]
    return float(row['threshold'])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class GroupSignal:
    """Named signal bound to a wallet mask."""

    def __init__(self, name, mask, col, side):
        self.name = name
        self.mask = mask
        self.col = col
        self.side = side


def ic_by_split(c_train, c_val, c_test, cols, roi="copyable_roi"):
    """IC per signal per split; returns (rows, pivoted df)."""
    rows = []
    for col in cols:
        row = {"signal": col}
        for lbl, df_c in [("train", c_train), ("val", c_val), ("test", c_test)]:
            if col in df_c.columns:
                row[lbl] = compute_event_ic(df_c[col], df_c[roi])
        rows.append(row)
    return pd.DataFrame(rows), None
