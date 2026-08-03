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
        Global size coefficient: ``qty = clip(scale * max(0, score) * copyable_qty,
        0, copyable_qty)``.
    cost_bps : float
        Cost in basis points applied to the notional of taken trades.

    Returns a dict with taken-trade info, capital utilization and a daily PnL
    series realized at market resolution.
    """
    t = trades[trades["copyable_qty"] > 0].copy()
    if t.empty:
        return {"trades": 0, "taken": pd.Series(dtype=bool), "daily_pnl": pd.Series(dtype=float)}

    score = t[score_col].fillna(0.0)
    weight = score.clip(lower=0.0).values
    copyable_qty = t["copyable_qty"].values
    qty = np.clip(scale * weight * copyable_qty, 0.0, copyable_qty)
    cost = t["price"].values * qty
    pnl = _per_share_pnl(t).values * qty
    pnl = np.nan_to_num(pnl, nan=0.0)

    dt = pd.to_datetime(t["dt"], utc=True).values.astype("datetime64[ns]")
    end = pd.to_datetime(t["end_date_iso"], utc=True).values.astype("datetime64[ns]")
    if "market_close" in t.columns:
        close = pd.to_datetime(t["market_close"], utc=True).values.astype("datetime64[ns]")
        end = np.maximum(end, close)
    # A trade that is the market's very last trade would have end == dt, making
    # its release event collide with its open (release sorts first, is skipped,
    # and the capital would never be freed).  Nudge such releases a second later.
    end = np.maximum(end, dt + np.timedelta64(1, "s"))

    # Event sweep: open at dt (cost), release at end_date_iso (free capital + pnl).
    # Releases sort before opens at equal timestamps so same-day resolutions recycle.
    events = []
    for i in range(len(t)):
        events.append((dt[i], 1, i))  # 1 = open
        events.append((end[i], 0, i))  # 0 = release
    events.sort(key=lambda e: (e[0], e[1]))

    n = len(t)
    taken = np.zeros(n, dtype=bool)
    used = 0.0
    timeline_ts = [events[0][0]]
    timeline_used = [0.0]
    for ts, kind, i in events:
        if kind == 0:  # release
            if taken[i]:
                used -= cost[i]
        else:  # open
            if used + cost[i] <= budget + 1e-9:
                taken[i] = True
                used += cost[i]
        timeline_ts.append(ts)
        timeline_used.append(used)

    gross_pnl = float(pnl[taken].sum())
    cnot = float(cost[taken].sum())
    cost_paid = float(cnot * cost_bps / 10_000.0)
    net_pnl = gross_pnl - cost_paid

    used_arr = np.maximum(timeline_used, 0.0)
    peak_used = float(used_arr.max()) if len(used_arr) else 0.0
    if len(timeline_ts) > 1:
        dts = np.diff(np.asarray(timeline_ts, dtype="datetime64[ns]").astype("int64")) / 1e9
        span = float(dts.sum()) if dts.sum() > 0 else 1e-9
        mean_used = float(np.dot(used_arr[:-1], dts) / span) if dts.sum() > 0 else 0.0
    else:
        span = 1e-9
        mean_used = 0.0

    daily = pd.Series(
        pnl[taken], index=pd.DatetimeIndex(end[taken], name="date")
    ).groupby(level=0).sum()
    daily.index = pd.to_datetime(daily.index, utc=True)

    taken_ids = pd.Series(t.index[taken])

    return {
        "trades": int(taken.sum()),
        "taken": taken_ids,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "cost_paid": cost_paid,
        "notional": cnot,
        "peak_used": peak_used,
        "mean_used": mean_used,
        "daily_pnl": daily,
    }


def sizing_sharpe(daily_pnl: pd.Series, periods_per_year: float = 365.0) -> float:
    """Annualized mean/std of a daily PnL series (0.0 if degenerate)."""
    s = daily_pnl.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(periods_per_year))


def select_scale(
    val_trades: pd.DataFrame,
    score_col: str,
    budget: float,
    scale_grid: np.ndarray,
    cost_bps: float = 0.0,
    primary: str = "sharpe_daily",
) -> tuple[float, pd.DataFrame]:
    """Grid-search ``scale`` on validation by a Sharpe-like objective."""
    rows = []
    for scale in scale_grid:
        res = capital_constrained_sim(val_trades, score_col, budget, float(scale), cost_bps)
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
