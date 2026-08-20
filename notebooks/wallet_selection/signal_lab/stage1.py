"""Functional stage-1 copy-trade signal exploration.

There is no workspace object.  Plain functions take explicit data:

- :func:`load_stage1_data` loads trades and computes the train-period wallet /
  hold metrics (returns a plain tuple).
- :func:`candidate_splits_for` builds the BUY-trade candidate universe for a
  set of wallets, split chronologically with train-fitted ``roi_res``.
- :func:`restrict_trades` cuts ``df_full`` down to the candidate conditions
  (the input for the position-checkpoint index).
- :func:`attach_position_signal_panel` attaches a strategy's declared signal
  families to candidate splits.
- :func:`run_strategy` runs a declarative strategy end-to-end.

This module keeps the existing stage1 methodology but fixes split-level signal
normalization leakage by fitting rank transforms on train only.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from lib import (
    DEFAULT_TAGS,
    compute_copyable_notional,
    compute_opening_metrics,
    load_trades,
    split_data,
    split_data_at_dates,
)
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

try:
    from .filters import WalletFilter
    from .signal_engines import (
        POS_OPP,
        POS_OWN,
        VAL_OPP,
        VAL_OWN,
        PositionSignalEngine,
        SignalKind,
        attach_position_signals,
        compute_hold_time_metrics,
        signal_col_name,
    )
    from .signal_lib import (
        apply_composite_score,
        apply_rank_transformer,
        bootstrap_ic,
        compute_event_ic,
        compute_optimal_weights,
        evaluate_strategy,
        fit_rank_transformer,
        fit_roi_residualizer,
        residualized_roi,
    )
except ImportError:
    from filters import WalletFilter  # type: ignore
    from signal_engines import (  # type: ignore
        POS_OPP,
        POS_OWN,
        VAL_OPP,
        VAL_OWN,
        PositionSignalEngine,
        SignalKind,
        attach_position_signals,
        compute_hold_time_metrics,
        signal_col_name,
    )
    from signal_lib import (  # type: ignore
        apply_composite_score,
        apply_rank_transformer,
        bootstrap_ic,
        compute_event_ic,
        compute_optimal_weights,
        evaluate_strategy,
        fit_rank_transformer,
        fit_roi_residualizer,
        residualized_roi,
    )


DEFAULT_SIGNAL_KINDS = [POS_OWN, POS_OPP, VAL_OWN, VAL_OPP]

_ENGINE_COLS = [
    "wallet",
    "condition_id",
    "outcome",
    "dt",
    "side",
    "position",
    "quantity",
    "price",
]


class StrategyProtocol(Protocol):
    """Minimal structural interface :func:`run_strategy` needs from a strategy.

    The canonical protocol lives in ``signal_lab.strategies.base``; this local
    protocol exists so :mod:`signal_lab.stage1` does not import the strategies
    package (which imports back into this module).
    """

    copy_mask: WalletFilter

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]: ...

    def get_signal_columns(self) -> list[str]: ...


def _reference_frame(
    splits: dict[str, pd.DataFrame],
    which: str,
) -> pd.DataFrame:
    if which == "train":
        return splits["train"]
    if which == "val":
        return splits["val"]
    if which == "train_val":
        return pd.concat([splits["train"], splits["val"]], ignore_index=True)
    raise ValueError(f"Unknown split reference {which!r}")


def _attach_copy_wallet_metrics(df_train: pd.DataFrame) -> pd.DataFrame:
    wallet_metrics, _ = compute_wallet_metrics(df_train)
    wallet_metrics["copyable_pnl_factor"] = np.clip(
        wallet_metrics["copyable_pnl"]
        / wallet_metrics["total_pnl"].replace(0, np.nan),
        0,
        1.0,
    ).fillna(0.0)
    wallet_metrics["copyable_roi"] = (
        wallet_metrics["average_roi"] * wallet_metrics["copyable_pnl_factor"]
    )
    opening_metrics = compute_opening_metrics(df_train)
    wallet_metrics = wallet_metrics.merge(opening_metrics, on="wallet", how="left")
    for col in [
        "opening_roi",
        "opening_pnl",
        "opening_copyable_roi",
        "opening_copyable_pnl",
    ]:
        wallet_metrics[col] = wallet_metrics[col].fillna(0.0)
    return wallet_metrics


def load_stage1_data(
    *,
    tags: set[str] | None = DEFAULT_TAGS,
    max_shards: int | None = None,
    train_end: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
    max_lead_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load trades and compute the train-period metrics.

    Returns a plain tuple
    ``(df_full, df_train, df_val, df_test, wallet_metrics, hold_metrics)``.

    Pass ``train_end``/``val_end``/``test_start`` (ISO dates, UTC) to use
    :func:`signal_lab.lib.split_data_at_dates`; otherwise the default
    chronological 40/30/30 split is used.

    Pass ``max_lead_days`` to keep only trades within that many days of
    contract resolution (``last_condition_trade_ts - dt <= max_lead_days``);
    ``None`` keeps all trades.  Applied before splitting so splits and
    train-period wallet metrics are computed on the same filtered universe.
    """
    df_full = compute_copyable_notional(load_trades(tags=tags, max_shards=max_shards))
    if max_lead_days is not None:
        lead = (
            pd.to_datetime(df_full["last_condition_trade_ts"], utc=True, errors="coerce")
            - df_full["dt"]
        )
        df_full = df_full[lead <= pd.Timedelta(days=max_lead_days)].copy()
        print(
            f"Lead filter (<= {max_lead_days}d before resolution): "
            f"{len(df_full):,} trades"
        )
    df_full = df_full[df_full["side"] == "BUY"].copy().reset_index(drop=True)
    if train_end is not None or val_end is not None or test_start is not None:
        df_train, df_val, df_test = split_data_at_dates(
            df_full, train_end=train_end, val_end=val_end, test_start=test_start,
            date_col="last_condition_trade_ts",
        )
    else:
        df_train, df_val, df_test = split_data(df_full, method="chronological")
    wallet_metrics = _attach_copy_wallet_metrics(df_train)
    hold_metrics = compute_hold_time_metrics(df_train)
    return df_full, df_train, df_val, df_test, wallet_metrics, hold_metrics


def candidate_splits_for(
    df_full: pd.DataFrame,
    wallets: Iterable[str],
    *,
    train_end: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
) -> dict[str, pd.DataFrame]:
    """BUY trades of ``wallets`` split chronologically with train-fitted ``roi_res``.

    Adds a ``market_close`` column: the last observed trade timestamp per
    market (any wallet, any side).  ``end_date_iso`` is only a nominal midnight
    date, while markets actually keep trading into the next day, so capital in
    a sizing backtest must be released at ``market_close``, not ``end_date_iso``.

    Pass ``train_end``/``val_end``/``test_start`` (ISO dates, UTC) to use
    :func:`signal_lab.lib.split_data_at_dates`; otherwise the default
    chronological split is used.
    """
    candidate_trades = df_full[
        df_full["wallet"].isin(wallets) & (df_full["side"] == "BUY")
    ].copy()
    if "copyable_pnl_20m_100" not in candidate_trades.columns and "copyable_qty_20m_100" in candidate_trades.columns:
        candidate_trades["copyable_pnl_20m_100"] = (
            (candidate_trades["final_price"] - candidate_trades["price"])
            * candidate_trades["copyable_qty_20m_100"]
        )
    market_close = df_full.groupby("condition_id")["dt"].max()
    if train_end is not None or val_end is not None or test_start is not None:
        c_train, c_val, c_test = split_data_at_dates(
            candidate_trades,
            train_end=train_end, val_end=val_end, test_start=test_start,
            date_col="last_condition_trade_ts",
        )
    else:
        c_train, c_val, c_test = split_data(candidate_trades, method="chronological")
    residual_fit = fit_roi_residualizer(c_train["copyable_roi"], c_train["price"])
    splits: dict[str, pd.DataFrame] = {}
    for label, frame in (("train", c_train), ("val", c_val), ("test", c_test)):
        frame["roi_res"] = residualized_roi(
            frame["copyable_roi"], frame["price"], residual_fit
        )
        frame["market_close"] = frame["condition_id"].map(market_close)
        splits[label] = frame
    return splits


def restrict_trades(df_full: pd.DataFrame, conditions: Iterable[str]) -> pd.DataFrame:
    """Trades restricted to ``conditions`` — the checkpoint-index input."""
    return df_full[df_full["condition_id"].isin(conditions)][_ENGINE_COLS].copy()


def attach_position_signal_panel(
    trades: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    filters: Iterable[WalletFilter],
    *,
    kinds: list[SignalKind] | None = None,
    fresh_kinds: list[SignalKind] | None = None,
    taus_h: list[float] | None = None,
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Attach position-signal families to candidate splits.

    For each wallet filter, attaches the base ``kinds`` (default pos/val x
    own/opp) and, for each tau hour in ``taus_h``, the fresh counterpart of
    every kind in ``fresh_kinds`` (defaults to ``kinds``) with the tau baked
    into the column name (``sig_fval_opp_6h_flipper``).      The checkpoint index
    is built once over ``trades``.

    The engine is cached on the ``trades`` object (``_position_signal_engine``)
    so repeated calls over the same frame reuse it.  The cache is implicitly
    invalidated by construction: :func:`restrict_trades` always returns a fresh
    ``.copy()``, which drops the attribute, so any new pipeline run rebuilds the
    engine.  Invariant: do not mutate ``trades`` in place after the engine is
    built (the engine holds a reference to it); no pipeline code does.
    """
    frames = {name: frame.copy(deep=True) for name, frame in splits.items()}
    engine = getattr(trades, "_position_signal_engine", None)
    if engine is None:
        engine = PositionSignalEngine(trades)
        object.__setattr__(trades, "_position_signal_engine", engine)
    all_kinds = list(kinds or DEFAULT_SIGNAL_KINDS)
    fresh = list(fresh_kinds if fresh_kinds is not None else all_kinds)
    taus = [float(t) for t in (taus_h or ())]

    signal_cols: list[str] = []
    for flt in filters:
        print(f"       - attaching base signals for set: {flt.name}", flush=True)
        wallets = set(flt(wallet_metrics, hold_metrics))
        A, B = engine.build_set(wallets)
        for frame in frames.values():
            attach_position_signals(frame, flt.name, A, B)
        for kind in all_kinds:
            col = signal_col_name(kind, flt.name)
            if col in frames["train"].columns:
                signal_cols.append(col)
        for tau_h in taus:
            print(f"       - attaching fresh signals (tau={tau_h}h) for set: {flt.name}", flush=True)
            A, B = engine.build_set(
                wallets, fresh_tau_ns=tau_h * 60 * 60 * 1_000_000_000
            )
            for frame in frames.values():
                attach_position_signals(frame, flt.name, A, B, fresh_tau_h=tau_h)
            for kind in fresh:
                col = signal_col_name(kind.fresh(), flt.name, tau_h=tau_h)
                if col in frames["train"].columns:
                    signal_cols.append(col)
    return frames, signal_cols


def run_strategy(
    df_full: pd.DataFrame,
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
    strategy: StrategyProtocol,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Run a declarative strategy end-to-end.

    Rebuilds the candidate universe from the strategy's copy-wallet filter,
    re-splits it chronologically, re-residualizes ROI on that universe's
    training split, restricts the trade frame to the candidate conditions, and
    returns ``(splits, signal_cols)`` ready for :func:`evaluate_signal_panel`.
    """
    copy_wallets = set(strategy.copy_mask(wallet_metrics, hold_metrics))
    splits = candidate_splits_for(df_full, copy_wallets)
    conditions: set[str] = set()
    for frame in splits.values():
        conditions.update(frame["condition_id"].unique())
    trades = restrict_trades(df_full, conditions)
    splits = strategy.calculate_signals(
        splits,
        trades=trades,
        wallet_metrics=wallet_metrics,
        hold_metrics=hold_metrics,
    )
    return splits, strategy.get_signal_columns()


def run_strategies(
    df_full: pd.DataFrame,
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
    strategies: list[StrategyProtocol],
    copy_mask: WalletFilter | None = None,
    *,
    train_end: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Run multiple declarative strategies end-to-end on a shared universe.

    If copy_mask is None, it uses the first strategy's copy_mask.
    Pass ``train_end``/``val_end``/``test_start`` to share the same
    date-based split as :func:`load_stage1_data` /
    :func:`candidate_splits_for`.
    """
    if not strategies:
        raise ValueError("Must provide at least one strategy")

    mask = copy_mask if copy_mask is not None else strategies[0].copy_mask
    copy_wallets = set(mask(wallet_metrics, hold_metrics))
    splits = candidate_splits_for(
        df_full, copy_wallets,
        train_end=train_end, val_end=val_end, test_start=test_start,
    )
    conditions: set[str] = set()
    for frame in splits.values():
        conditions.update(frame["condition_id"].unique())
    trades = restrict_trades(df_full, conditions)

    all_cols = []
    for i, strategy in enumerate(strategies):
        print(f"  -> [{i+1}/{len(strategies)}] Calculating signals for {strategy.name}...", flush=True)
        splits = strategy.calculate_signals(
            splits,
            trades=trades,
            wallet_metrics=wallet_metrics,
            hold_metrics=hold_metrics,
        )
        all_cols.extend(strategy.get_signal_columns())

    # Remove duplicates while preserving order
    all_cols = list(dict.fromkeys(all_cols))
    return splits, all_cols


def evaluate_signal_panel(
    splits: dict[str, pd.DataFrame],
    signal_cols: list[str],
    *,
    roi_col: str = "roi_res",
    selection_splits: tuple[str, ...] = ("train", "val"),
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int = 42,
    min_ic: float = 0.005,
    presence_min: float = 0.005,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate arbitrary signal columns with the stage-1 IC protocol."""
    pooled = pd.concat(
        [splits[name][signal_cols + [roi_col]] for name in selection_splits],
        ignore_index=True,
    )
    rows = []
    selected = []
    for col in signal_cols:
        split_ics = {
            split: compute_event_ic(splits[split][col].fillna(0.0), splits[split][roi_col])
            for split in splits
        }
        mean_ic, ci_lo, ci_hi = bootstrap_ic(
            pooled[col].fillna(0.0),
            pooled[roi_col],
            n_iter=n_boot,
            alpha=alpha,
            seed=seed,
        )
        significant = np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)
        consistency = all(
            np.isfinite(split_ics[name])
            for name in selection_splits
        ) and all(
            np.sign(split_ics[selection_splits[0]]) * np.sign(split_ics[name]) > 0
            for name in selection_splits[1:]
        )
        presence = float((splits["train"][col] > 0).mean())
        row = {
            "signal": col,
            "presence_train": presence,
            "boot_mean_ic": mean_ic,
            "boot_ci_lo": ci_lo,
            "boot_ci_hi": ci_hi,
            "significant": significant,
        }
        for split, ic in split_ics.items():
            row[f"IC_{split}"] = ic
        rows.append(row)
        if (
            consistency
            and significant
            and presence >= presence_min
            and all(abs(split_ics[name]) >= min_ic for name in selection_splits)
        ):
            selected.append(col)
    report = pd.DataFrame(rows)
    if not report.empty:
        report["|IC_train|"] = report["IC_train"].abs()
        report = report.sort_values("|IC_train|", ascending=False).reset_index(drop=True)
    return report, selected


def rank_normalize_splits(
    splits: dict[str, pd.DataFrame],
    signal_cols: list[str],
    *,
    prefix: str = "rank_",
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
    """Train-fit signal normalization for combination and thresholding."""
    out = {name: frame.copy(deep=True) for name, frame in splits.items()}
    fits = {}
    for col in signal_cols:
        fit = fit_rank_transformer(out["train"][col].fillna(0.0))
        fits[col] = fit
        for split, frame in out.items():
            frame[f"{prefix}{col}"] = apply_rank_transformer(
                frame[col].fillna(0.0),
                fit,
            )
    return out, fits


def build_composite_scores(
    splits: dict[str, pd.DataFrame],
    signal_cols: list[str],
    *,
    roi_col: str = "roi_res",
    weight_split: str = "train",
    shrinkage: float = 0.5,
    prefix: str = "rank_",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, dict[str, np.ndarray]]]:
    """Build equal, IC-weighted, and shrinkage-Markowitz composites.

    Signal weights are fit on ``weight_split`` and then applied unchanged to all
    splits. Defaulting to ``train`` avoids reusing validation for both fitting
    and threshold selection.
    """
    normalized, rank_fits = rank_normalize_splits(splits, signal_cols, prefix=prefix)
    rank_cols = [f"{prefix}{col}" for col in signal_cols]
    ref = _reference_frame(normalized, weight_split)

    ic_vals = {
        col: compute_event_ic(ref[f"{prefix}{col}"], ref[roi_col])
        for col in signal_cols
    }
    ic_signed = {col: (ic if np.isfinite(ic) else 0.0) for col, ic in ic_vals.items()}
    n = len(signal_cols)
    if n == 0:
        return normalized, {}, rank_fits

    weights_equal = pd.Series(
        {f"{prefix}{col}": np.sign(ic_signed[col]) / n for col in signal_cols}
    )
    ic_sum = sum(abs(v) for v in ic_signed.values())
    weights_ic = pd.Series(
        {
            f"{prefix}{col}": (
                ic_signed[col] / ic_sum if ic_sum > 0 else np.sign(ic_signed[col]) / n
            )
            for col in signal_cols
        }
    )
    weights_shrink = compute_optimal_weights(
        ref,
        rank_cols,
        roi_col=roi_col,
        shrinkage=shrinkage,
    )
    schemes = {
        "equal": weights_equal,
        "ic_weighted": weights_ic,
        "shrinkage_markowitz": weights_shrink,
    }
    for frame in normalized.values():
        for name, weights in schemes.items():
            frame[f"composite_{name}"] = apply_composite_score(frame, rank_cols, weights)
    return normalized, schemes, rank_fits


def evaluate_threshold_grid(
    df: pd.DataFrame,
    score_col: str,
    *,
    thresholds: np.ndarray | None = None,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Evaluate a score threshold grid on one split."""
    if thresholds is None:
        thresholds = np.arange(-1.0, 1.01, 0.05)
    rows = [
        evaluate_strategy(df, score_col, float(threshold), cost_bps=cost_bps)
        for threshold in thresholds
    ]
    out = pd.DataFrame(rows)
    out["pnl_per_trade_net"] = out["copyable_pnl_net"] / out["trades"].clip(lower=1)
    return out
