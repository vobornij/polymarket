"""Deployed-capital exposure carve thresholds for wallet copies.

A carve is a bankroll-discipline filter: per wallet, keep only the fills whose
deployed capital (``position * price``, the post-trade USDC value of the book
on that condition/outcome) is above a threshold. Thresholds are fitted on the
training period only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wallet_capture_thresholds(
    frame: pd.DataFrame,
    fracs: tuple[float, ...],
    pnl_col: str = "copyable_pnl",
    exp_col: str = "deployed_capital",
) -> pd.DataFrame:
    """Per-wallet ``T_{w,f}``: smallest deployed capital whose fills capture >= ``f`` of
    that wallet's train PnL (fills sorted by deployed capital descending).

    For wallet ``w`` with total train PnL ``total``, sort its fills by
    ``exp_col`` descending and take the exposure of the first fill whose
    cumulative ``pnl_col`` reaches ``f * total``. ``f == 1.0`` yields the
    wallet's minimum exposure (copy everything). Wallets with ``total <= 0``
    get no threshold; callers treat them as keep-everything (``fillna(0.0)``).

    Returns a DataFrame with columns ``wallet``, ``frac``, ``threshold``.
    """
    rows: list[dict] = []
    for wallet, g in frame.groupby("wallet", sort=False):
        total = float(g[pnl_col].sum())
        if total <= 0:
            continue
        g = g.sort_values(exp_col, ascending=False)
        cum = g[pnl_col].cumsum().to_numpy(dtype=float)
        exps = g[exp_col].to_numpy(dtype=float)
        for f in sorted(set(fracs)):
            mask = cum >= f * total
            thr = float(exps[int(np.argmax(mask))]) if np.any(mask) else 0.0
            rows.append({"wallet": wallet, "frac": f, "threshold": thr})
    return pd.DataFrame(rows, columns=["wallet", "frac", "threshold"])


def carve_threshold_map(
    thresholds: pd.DataFrame, frac: float
) -> pd.Series:
    """``wallet -> T_{w,frac}`` Series from a ``wallet_capture_thresholds`` frame.

    Wallets absent from ``thresholds`` (no positive train PnL) are NaN and are
    kept whole by callers via ``fillna(0.0)``.
    """
    return thresholds[thresholds["frac"] == frac].set_index("wallet")["threshold"]