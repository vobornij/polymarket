"""
Capital-constrained sizing backtest for composite scores.

Copies a *scaled share quantity* of each candidate trade (score-proportional,
clipped to ``copyable_qty``) under a global capital budget.  Capital is locked
from the trade's ``dt`` until market resolution (``end_date_iso``), so a budget
forces trades to compete — the direct, realistic test of whether a composite's
raw PnL edge survives risk-adjusted sizing.

The raw ``copyable_pnl`` edge is heavily price-driven ("buy cheap = more upside
per share"), so the module is designed to compare a **price-exposed** composite
(fit on ``copyable_pnl``) against a **price-controlled** composite (fit on
``pnl_res``) under the same budget: whichever sizes better on risk-adjusted PnL
answers whether the price component is tradable alpha or a variance artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _per_share_pnl(df: pd.DataFrame) -> pd.Series:
    """Dollar PnL per copied share: ``copyable_pnl / copyable_qty``."""
    return df["copyable_pnl"] / df["copyable_qty"].replace(0, np.nan)


def capital_constrained_sim(
    trades: pd.DataFrame,
    score_col: str,
    budget: float,
    scale: float,
    cost_bps: float = 0.0,
    score_floor: float | None = None,
    alpha_col: str | None = None,
    cap_col: str | None = None,
) -> dict:
    """Simulate copying ``trades`` with a score-proportional share size under a budget.

    Parameters
    ----------
    trades : DataFrame
        Candidate trades with columns ``dt``, ``price``, ``copyable_qty``,
        ``copyable_pnl``, ``end_date_iso`` and ``score_col``.  If ``market_close``
        is present, capital is released at ``max(end_date_iso, market_close)``
        (the market's actual close); otherwise ``end_date_iso`` alone.
    score_col : str
        Composite score column (rank-normalized, ~[-1, 1]).
    budget : float
        Global capital budget in dollars.
    scale : float
        Global size coefficient: ``qty = clip(scale * max(0, score) * alpha * copyable_qty,
        0, cap)``.
    cost_bps : float
        Cost in basis points applied to the notional of taken trades.
    score_floor : float | None
        If set, only trades with ``score >= score_floor`` are considered; sizing
        above the floor stays score-proportional.  ``None`` fires every positive
        score (the natural proportional overlay).
    alpha_col : str | None
        Per-trade multiplier column (defaults to 1.0).  Trades with ``alpha <= 0``
        are not fired.  This is the per-wallet copy scale.
    cap_col : str | None
        Per-trade maximum quantity column (defaults to ``copyable_qty``).  For
        scale > 1 this should be the share-depth cap ``bucket_avail_copy_qty``.

    Returns a dict with taken-trade info, capital utilization and a daily PnL
    series realized at market resolution.
    """
    prep = _prep_events(trades, score_col, score_floor, alpha_col, cap_col)
    if prep is None:
        return {"trades": 0, "taken": pd.Series(dtype=bool), "daily_pnl": pd.Series(dtype=float)}
    return _sweep_events(prep, scale, budget, cost_bps)


def _prep_events(
    trades: pd.DataFrame,
    score_col: str,
    score_floor: float | None = None,
    alpha_col: str | None = None,
    cap_col: str | None = None,
) -> dict | None:
    """Filter + pre-sort the event stream once for a ``(score_col, floor)``.

    Returns None if no trade survives.  The pre-sorted event arrays are shared
    across all ``scale`` values so the budget sweep can be reused cheaply.
    """
    t = trades[trades["copyable_qty"] > 0].copy()
    if t.empty:
        return None
    score = t[score_col].fillna(0.0)
    if score_floor is not None:
        t = t[score >= score_floor].copy()
        if t.empty:
            return None
        score = t[score_col].fillna(0.0)
    weight = score.clip(lower=0.0).values
    alpha = np.ones(len(t), dtype=float)
    if alpha_col is not None:
        alpha = t[alpha_col].fillna(1.0).clip(lower=0.0).values
    keep = (weight > 0) & (alpha > 0)
    if not keep.any():
        return None
    t = t[keep].copy()
    weight = weight[keep]
    alpha = alpha[keep]

    copyable_qty = t["copyable_qty"].values
    cap = copyable_qty
    if cap_col is not None:
        cap = np.clip(t[cap_col].fillna(0.0).values, 0.0, None)
    price = t["price"].values
    per_share = _per_share_pnl(t).values

    dt_ns = pd.to_datetime(t["dt"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
    end_ns = pd.to_datetime(t["end_date_iso"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
    if "market_close" in t.columns:
        close = pd.to_datetime(t["market_close"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
        end_ns = np.maximum(end_ns, close)
    # A trade that is the market's very last trade would have end == dt, making
    # its release event collide with its open (release sorts first, is skipped,
    # and the capital would never be freed).  Nudge such releases a second later.
    end_ns = np.maximum(end_ns, dt_ns + 1_000_000_000)

    # Event sweep: open at dt (cost), release at end_date_iso (free capital + pnl).
    # Releases sort before opens at equal timestamps so same-day resolutions recycle.
    n = len(t)
    ts = np.concatenate([dt_ns, end_ns])
    kind = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    idx = np.concatenate([np.arange(n), np.arange(n)])
    order = np.lexsort((kind, ts))  # stable: ts primary, release(0) before open(1)
    ts = ts[order]
    kind = kind[order]
    idx = idx[order]

    return {
        "n": n,
        "ts": ts,
        "kind": kind,
        "idx": idx,
        "weight": weight,
        "alpha": alpha,
        "copyable_qty": copyable_qty,
        "cap": cap,
        "price": price,
        "per_share": per_share,
        "end_ns": end_ns,
        "t_index": t.index,
    }


def _sweep_events(
    prep: dict,
    scale: float,
    budget: float,
    cost_bps: float = 0.0,
) -> dict:
    """Run the sequential budget sweep for a specific ``scale`` on a prep'd frame."""
    weight = prep["weight"]
    alpha = prep["alpha"]
    copyable_qty = prep["copyable_qty"]
    cap = prep["cap"]
    price = prep["price"]
    per_share = prep["per_share"]
    qty = np.clip(scale * weight * alpha * copyable_qty, 0.0, cap)
    cost = price * qty
    pnl = np.nan_to_num(per_share * qty, nan=0.0)

    n = prep["n"]
    ts = prep["ts"]
    kind = prep["kind"]
    idx = prep["idx"]
    budget = float(budget)
    tol = 1e-9
    taken = np.zeros(n, dtype=bool)
    used = 0.0
    peak = 0.0
    sum_used_dt = 0.0
    total_cost_taken = 0.0
    total_pnl_taken = 0.0
    prev = ts[0]
    for k in range(len(ts)):
        tsk = ts[k]
        if tsk != prev:
            sum_used_dt += used * (tsk - prev)
            prev = tsk
        i = idx[k]
        if kind[k] == 0:  # release
            if taken[i]:
                used -= cost[i]
        else:  # open
            c = cost[i]
            if used + c <= budget + tol:
                taken[i] = True
                used += c
                total_cost_taken += c
                total_pnl_taken += pnl[i]
        if used > peak:
            peak = used

    span = ts[-1] - ts[0]
    mean_used = float(sum_used_dt / span) if span > 0 else 0.0

    gross_pnl = total_pnl_taken
    cnot = total_cost_taken
    cost_paid = float(cnot * cost_bps / 10_000.0)
    net_pnl = gross_pnl - cost_paid

    taken_end = prep["end_ns"][taken]
    daily = pd.Series(
        pnl[taken],
        index=pd.DatetimeIndex(pd.to_datetime(taken_end, utc=True), name="date"),
    ).groupby(level=0).sum()
    daily.index = pd.to_datetime(daily.index, utc=True)

    return {
        "trades": int(taken.sum()),
        "taken": pd.Series(prep["t_index"][taken]),
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "cost_paid": cost_paid,
        "notional": cnot,
        "peak_used": peak,
        "mean_used": mean_used,
        "daily_pnl": daily,
    }


def sizing_sharpe(daily_pnl: pd.Series, periods_per_year: float = 365.0) -> float:
    """Annualized mean/std of a daily PnL series (0.0 if degenerate)."""
    s = daily_pnl.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(periods_per_year))


def score_floor_for_fraction(
    frame: pd.DataFrame,
    score_col: str,
    fraction: float,
) -> float:
    """Score floor that fires the top ``fraction`` of scored candidates."""
    s = frame[score_col].dropna()
    if s.empty:
        return 0.0
    return float(s.quantile(1.0 - fraction))


def block_bootstrap_sharpe(
    daily_pnl: pd.Series,
    block_size: int = 7,
    n_iter: int = 500,
    seed: int = 42,
    periods_per_year: float = 365.0,
) -> tuple[float, float, float]:
    """Block-bootstrap distribution of the annualized Sharpe of ``daily_pnl``.

    Resamples contiguous blocks (of ``block_size`` calendar days) with
    replacement to the original length, returning ``(point, lo, hi)`` where the
    interval is the 2.5-97.5 percentiles.  ``point`` is the plain sample Sharpe.
    """
    s = daily_pnl.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0, 0.0, 0.0
    point = float(s.mean() / s.std() * np.sqrt(periods_per_year))
    idx = s.index
    start = pd.Timestamp("2000-01-01", tz="UTC")
    day_ids = (idx - start).days.to_numpy()
    days = np.arange(int(day_ids.min()), int(day_ids.max()) + 1)
    vals = np.zeros(len(days))
    for d, v in s.items():
        vals[int((d - start).days) - int(day_ids.min())] = v
    non_nan = np.isfinite(vals)
    vals = np.where(non_nan, vals, 0.0)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(days) / block_size))
    sharps = np.empty(n_iter)
    for i in range(n_iter):
        blocks = rng.integers(0, max(len(days) - block_size + 1, 1), size=n_blocks)
        bs = np.concatenate([vals[b:b + block_size] for b in blocks])
        bs = bs[:len(days)]
        sd = bs.std()
        sharps[i] = bs.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    lo, hi = np.percentile(sharps, [2.5, 97.5])
    return point, float(lo), float(hi)


def select_scale(
    val_trades: pd.DataFrame,
    score_col: str,
    budget: float,
    scale_grid: np.ndarray,
    cost_bps: float = 0.0,
    primary: str = "sharpe_daily",
    score_floor: float | None = None,
) -> tuple[float, pd.DataFrame]:
    """Grid-search ``scale`` on validation by a Sharpe-like objective."""
    prep = _prep_events(val_trades, score_col, score_floor)
    rows = []
    for scale in scale_grid:
        if prep is None:
            continue
        res = _sweep_events(prep, float(scale), budget, cost_bps)
        if res["trades"] == 0:
            continue
        daily = res["daily_pnl"]
        weekly = daily.resample("W").sum()
        rows.append({
            "scale": float(scale),
            "trades": res["trades"],
            "net_pnl": res["net_pnl"],
            "peak_used": res["peak_used"],
            "mean_used": res["mean_used"],
            "pnl_per_peak": res["net_pnl"] / res["peak_used"] if res["peak_used"] > 0 else 0.0,
            "sharpe_daily": sizing_sharpe(daily, 365.0),
            "sharpe_weekly": sizing_sharpe(weekly, 52.0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return float(scale_grid[0]), out
    best = out.sort_values(primary, ascending=False).iloc[0]
    return float(best["scale"]), out
