"""Copy-wallet exposure signal family.

For each candidate copy BUY, ``exposure = position * price`` measures the
copied wallet's post-trade position value on that (condition, outcome) in
USDC — both inputs are precomputed on the trade data.

The hypothesis (see ``ideas/exposure_threshold_capture.md``): a wallet only
builds a large book when it has edge, so high-exposure candidate trades carry
higher copyable PnL.

Methodology note: cross-sectional exposure is dominated by wallet size (a whale
bankroll vs a retail bankroll), so the primary axis is *wallet-relative*: where
does this trade sit in the wallet's own exposure distribution?  All per-wallet
transforms are fit on the train split only (no val/test leakage) and applied to
every split.

Signals attached:

- ``sig_exp_pos`` — position after the trade.
- ``sig_exp_usdc`` — exposure = ``position * price``.
- ``sig_exp_log`` — ``log1p(exposure)`` (heavy-tail tolerant).
- ``sig_exp_wt_rank`` — within-wallet exposure rank in [0, 1] (train-fit):
  0 = smallest book this wallet ever builds, 1 = its biggest.
- ``sig_exp_wt_minmax`` — within-wallet min-max scale in [0, 1] (train-fit;
  diagnostic — sensitive to a single outlier max exposure).
- ``sig_exp_marg`` — marginal exposure = ``quantity * price`` (the book added
  by this trade; contrasts with the post-trade level and separates conviction
  bets from averaging-down).
- ``sig_exp_marg_wt`` — within-wallet [0, 1] rank of the marginal exposure.
- ``sig_exp_thr_{f}`` — binary ``exposure >= T_{w,f}``: the largest exposure of
  wallet ``w`` whose train trades with ``exposure >= T_{w,f}`` capture a
  fraction ``f`` of the wallet's total train copyable PnL (f in ``fracs``).
- ``sig_exp_top_{t}`` — binary ``exposure >= P_{w,t}``: copy only the top ``t``
  of the wallet's train exposure distribution (threshold = train exposure
  quantile ``1 - t``; t in ``fracs``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.filters import COPY_DEFAULT, WalletFilter
from signal_lab.strategies.base import DeclarativeStrategy

BASE_COLS = (
    "sig_exp_pos",
    "sig_exp_usdc",
    "sig_exp_log",
    "sig_exp_wt_rank",
    "sig_exp_wt_minmax",
    "sig_exp_marg",
    "sig_exp_marg_wt",
)

DEFAULT_FRACS = (1.0, 0.8, 0.5, 0.2)


def threshold_col(frac: float) -> str:
    return f"sig_exp_thr_{frac:g}"


def top_col(top_frac: float) -> str:
    return f"sig_exp_top_{top_frac:g}"


def wallet_capture_thresholds(
    frame: pd.DataFrame,
    fracs: tuple[float, ...],
    pnl_col: str = "copyable_pnl",
    exp_col: str = "exposure",
) -> pd.DataFrame:
    """Per-wallet ``T_{w,f}``: largest exposure capturing >= ``f`` of train PnL.

    For wallet ``w`` with total train copyable PnL ``total``, sort its trades by
    exposure descending and take the exposure of the first trade where the
    cumulative PnL reaches ``f * total``.  Wallets with ``total <= 0`` are
    skipped (they contribute no positive-PnL capture).  ``f = 1.0`` yields the
    wallet's minimum exposure (copy everything).
    """
    rows: list[dict] = []
    fracs = sorted(set(fracs))
    for wallet, g in frame.groupby("wallet", sort=False):
        total = float(g[pnl_col].sum())
        if total <= 0:
            continue
        g = g.sort_values(exp_col, ascending=False)
        cum = g[pnl_col].cumsum().to_numpy(dtype=float)
        exposures = g[exp_col].to_numpy(dtype=float)
        for f in fracs:
            mask = cum >= f * total
            thr = float(exposures[int(np.argmax(mask))]) if np.any(mask) else 0.0
            rows.append({"wallet": wallet, "frac": f, "threshold": thr})
    return pd.DataFrame(rows, columns=["wallet", "frac", "threshold"])


def wallet_exposure_quantile_thresholds(
    frame: pd.DataFrame,
    keep_top_fracs: tuple[float, ...],
    exp_col: str = "exposure",
) -> pd.DataFrame:
    """Per-wallet ``P_{w,t}``: train exposure quantile ``1 - t``.

    Keeps the top ``t`` of the wallet's own exposure distribution, e.g. ``t =
    0.2`` copies only the biggest 20% of the trades this wallet ever makes.
    """
    rows: list[dict] = []
    for wallet, g in frame.groupby("wallet", sort=False):
        vals = g[exp_col].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        for t in keep_top_fracs:
            thr = float(np.quantile(vals, 1.0 - t))
            rows.append({"wallet": wallet, "top_frac": t, "threshold": thr})
    return pd.DataFrame(rows, columns=["wallet", "top_frac", "threshold"])


def fit_wallet_exposure_grids(
    frame: pd.DataFrame,
    exp_col: str = "exposure",
) -> dict[str, np.ndarray]:
    """Per-wallet sorted exposure grid, fit on train only."""
    return {
        wallet: np.sort(g[exp_col].to_numpy(dtype=float))
        for wallet, g in frame.groupby("wallet", sort=False)
    }


def apply_wallet_rank01(
    values: np.ndarray,
    wallet: str,
    grids: dict[str, np.ndarray],
) -> np.ndarray:
    """Map ``values`` to [0, 1] within the wallet's train exposure grid."""
    grid = grids.get(wallet)
    out = np.full(len(values), np.nan, dtype=float)
    if grid is None or len(grid) == 0:
        return out
    left = np.searchsorted(grid, values, side="left")
    right = np.searchsorted(grid, values, side="right")
    rank = (left + right + 1.0) / 2.0
    out[:] = (rank - 1.0) / (len(grid) - 1.0)
    return out


def apply_wallet_minmax(
    values: np.ndarray,
    wallet: str,
    grids: dict[str, np.ndarray],
) -> np.ndarray:
    """Map ``values`` to [0, 1] via the wallet's train min/max exposure."""
    grid = grids.get(wallet)
    out = np.full(len(values), np.nan, dtype=float)
    if grid is None or len(grid) == 0:
        return out
    lo, hi = float(grid[0]), float(grid[-1])
    if not np.isfinite(hi - lo) or hi - lo <= 0.0:
        out[:] = 0.0
        return out
    out[:] = (values - lo) / (hi - lo)
    return out


class ExposureSignals(DeclarativeStrategy):
    copy_mask: WalletFilter = COPY_DEFAULT

    def __init__(self, fracs: tuple[float, ...] = DEFAULT_FRACS):
        keep = sorted(f for f in fracs if f < 1.0)
        self.fracs = tuple(keep)
        self.tops = tuple(sorted(keep, reverse=True))

    @property
    def name(self) -> str:
        return "ExposureSignals"

    def get_signal_columns(self) -> list[str]:
        return (
            list(BASE_COLS)
            + [threshold_col(f) for f in self.fracs]
            + [top_col(t) for t in self.tops]
        )

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        frames = {name: frame.copy(deep=True) for name, frame in splits.items()}
        has_qty = "quantity" in frames["train"].columns
        for frame in frames.values():
            frame["exposure"] = frame["position"] * frame["price"]
            qty = frame["quantity"] if has_qty else frame["copyable_qty_5m_100"]
            frame["marg_exp"] = qty * frame["price"]
            frame["sig_exp_pos"] = frame["position"]
            frame["sig_exp_usdc"] = frame["exposure"]
            frame["sig_exp_log"] = np.log1p(frame["exposure"].clip(lower=0.0))
            frame["sig_exp_marg"] = frame["marg_exp"]

        train = frames["train"]
        grids = fit_wallet_exposure_grids(train, "exposure")
        mgrids = fit_wallet_exposure_grids(train, "marg_exp")
        for name, frame in frames.items():
            wals = frame["wallet"].to_numpy()
            for col, fn, gs in (
                ("sig_exp_wt_rank", apply_wallet_rank01, grids),
                ("sig_exp_wt_minmax", apply_wallet_minmax, grids),
                ("sig_exp_marg_wt", apply_wallet_rank01, mgrids),
            ):
                frame[col] = np.nan
                for wallet in np.unique(wals):
                    m = wals == wallet
                    frame.loc[m, col] = fn(
                        frame.loc[m, ("exposure" if "marg" not in col else "marg_exp")].to_numpy(dtype=float),
                        wallet,
                        gs,
                    )
                frame[col] = frame[col].fillna(0.0)

        th = wallet_capture_thresholds(train, self.fracs + (1.0,))
        for f in self.fracs:
            col = threshold_col(f)
            tmap = pd.Series(
                th[th["frac"] == f]["threshold"].to_numpy(),
                index=th[th["frac"] == f]["wallet"],
            )
            for name, frame in frames.items():
                T = frame["wallet"].map(tmap)
                frame[col] = np.where(T.isna(), 0.0, (frame["exposure"] >= T).astype(float))

        thq = wallet_exposure_quantile_thresholds(train, self.tops)
        for t in self.tops:
            col = top_col(t)
            tmap = pd.Series(
                thq[thq["top_frac"] == t]["threshold"].to_numpy(),
                index=thq[thq["top_frac"] == t]["wallet"],
            )
            for name, frame in frames.items():
                T = frame["wallet"].map(tmap)
                frame[col] = np.where(T.isna(), 0.0, (frame["exposure"] >= T).astype(float))
        return frames