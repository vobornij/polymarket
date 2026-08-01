"""Reusable workspace for rapid stage-1 copy-trade signal exploration.

The goal is to make the current ``stage1_experimental`` process notebook-light:

1. build a candidate-trade universe from public trades,
2. attach one or more signal families,
3. evaluate signals on train/val with price-residualized ROI,
4. combine, threshold, and report on a held-out test split.

This module keeps the existing stage1 methodology but makes the slow parts
reusable, and fixes split-level signal normalization leakage by fitting rank
transforms on train only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

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
)
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

try:
    from .signal_engines import PositionSignalEngine, archetype_sets, compute_hold_time_metrics
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
    from signal_engines import PositionSignalEngine, archetype_sets, compute_hold_time_metrics  # type: ignore
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


DEFAULT_COPY_RULES = {
    "min_buy_roi": 0.02,
    "min_buckets": 20,
    "min_markets": 15,
    "min_trade_count": 100,
    "max_drawdown_to_pnl": 0.6,
    "min_copyable_roi": 0.05,
}

DEFAULT_SIGNAL_KINDS = [
    ("pos", "own"),
    ("pos", "opp"),
    ("val", "own"),
    ("val", "opp"),
]

DEFAULT_FRESH_SIGNAL_KINDS = [
    ("pos", "own"),
    ("pos", "opp"),
    ("pos", "total"),
    ("val", "own"),
    ("val", "opp"),
    ("val", "total"),
    ("avgc", "own"),
    ("avgc", "opp"),
    ("uwl", "own"),
    ("uwl", "opp"),
    ("fpos", "own"),
    ("fpos", "opp"),
    ("fval", "own"),
    ("fval", "opp"),
    ("favgc", "own"),
    ("favgc", "opp"),
    ("fuwl", "own"),
    ("fuwl", "opp"),
]

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

_CACHE_DIR = Path("/tmp/pos_explore_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Stage1Workspace:
    df_full: pd.DataFrame
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    wallet_metrics: pd.DataFrame
    hold_metrics: pd.DataFrame
    copy_wallets: set[str]
    candidate_trades: pd.DataFrame
    candidate_splits: dict[str, pd.DataFrame]
    conditions: set[str]
    signal_sets: dict[str, set[str]]
    engine: PositionSignalEngine
    residual_fit: dict[str, float]
    set_tables_cache: dict[tuple[str, int | None], tuple[pd.DataFrame, pd.DataFrame]]

    def clone_candidate_splits(self) -> dict[str, pd.DataFrame]:
        return {
            split: frame.copy(deep=True)
            for split, frame in self.candidate_splits.items()
        }


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


def select_copy_wallets(
    wallet_metrics: pd.DataFrame,
    copy_rules: dict[str, float] | None = None,
) -> set[str]:
    copy_rules = {**DEFAULT_COPY_RULES, **(copy_rules or {})}
    mask = (
        (wallet_metrics["buy_roi"] >= copy_rules["min_buy_roi"])
        & (wallet_metrics["num_buckets"] >= copy_rules["min_buckets"])
        & (wallet_metrics["num_markets"] >= copy_rules["min_markets"])
        & (wallet_metrics["trade_count"] >= copy_rules["min_trade_count"])
        & (
            wallet_metrics["max_drawdown_to_pnl"].fillna(1.0)
            <= copy_rules["max_drawdown_to_pnl"]
        )
        & (wallet_metrics["copyable_roi"].fillna(0.0) >= copy_rules["min_copyable_roi"])
    )
    return set(wallet_metrics.loc[mask, "wallet"])


def build_stage1_workspace(
    *,
    tags: set[str] | None = DEFAULT_TAGS,
    copy_rules: dict[str, float] | None = None,
    archetype_min_trade_count: int = 100,
) -> Stage1Workspace:
    """Build the reusable stage-1 research workspace from raw public trades."""
    df_full = compute_copyable_notional(load_trades(tags=tags))
    df_train, df_val, df_test = split_data(df_full, method="chronological")

    wallet_metrics = _attach_copy_wallet_metrics(df_train)
    hold_metrics = compute_hold_time_metrics(df_train)
    copy_wallets = select_copy_wallets(wallet_metrics, copy_rules)

    candidate_trades = df_full[
        df_full["wallet"].isin(copy_wallets) & (df_full["side"] == "BUY")
    ].copy()
    c_train, c_val, c_test = split_data(candidate_trades, method="chronological")
    candidate_splits = {"train": c_train, "val": c_val, "test": c_test}

    residual_fit = fit_roi_residualizer(c_train["copyable_roi"], c_train["price"])
    for frame in candidate_splits.values():
        frame["roi_res"] = residualized_roi(
            frame["copyable_roi"],
            frame["price"],
            residual_fit,
        )

    signal_sets = {
        name: set(sel["wallet"])
        for name, sel in archetype_sets(
            wallet_metrics,
            hold_metrics,
            min_trade_count=archetype_min_trade_count,
        ).items()
    }
    conditions = set(candidate_trades["condition_id"].unique())
    restricted = df_full[df_full["condition_id"].isin(conditions)][_ENGINE_COLS].copy()
    engine = PositionSignalEngine(restricted)

    return Stage1Workspace(
        df_full=df_full,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        wallet_metrics=wallet_metrics,
        hold_metrics=hold_metrics,
        copy_wallets=copy_wallets,
        candidate_trades=candidate_trades,
        candidate_splits=candidate_splits,
        conditions=conditions,
        signal_sets=signal_sets,
        engine=engine,
        residual_fit=residual_fit,
        set_tables_cache={},
    )


def build_stage1_workspace_cached(
    *,
    tags: set[str] | None = DEFAULT_TAGS,
    copy_rules: dict[str, float] | None = None,
    archetype_min_trade_count: int = 100,
    cache_dir: Path = _CACHE_DIR,
    force: bool = False,
) -> Stage1Workspace:
    """Build the stage-1 workspace with simple parquet/pickle caching.

    The first run is still expensive because it builds the full training
    workspace. Later runs reuse the persisted frames and rebuild only the
    lightweight engine object.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "df_full": cache_dir / "signal_lab_df_full.parquet",
        "df_train": cache_dir / "signal_lab_df_train.parquet",
        "df_val": cache_dir / "signal_lab_df_val.parquet",
        "df_test": cache_dir / "signal_lab_df_test.parquet",
        "candidate_trades": cache_dir / "signal_lab_candidate_trades.parquet",
        "c_train": cache_dir / "signal_lab_c_train.parquet",
        "c_val": cache_dir / "signal_lab_c_val.parquet",
        "c_test": cache_dir / "signal_lab_c_test.parquet",
        "wallet_metrics": cache_dir / "signal_lab_wallet_metrics.parquet",
        "hold_metrics": cache_dir / "signal_lab_hold_metrics.parquet",
        "restricted": cache_dir / "signal_lab_df_restricted.parquet",
        "copy_wallets": cache_dir / "signal_lab_copy_wallets.pkl",
        "conditions": cache_dir / "signal_lab_conditions.pkl",
        "signal_sets": cache_dir / "signal_lab_signal_sets.pkl",
        "residual_fit": cache_dir / "signal_lab_residual_fit.pkl",
    }

    if not force and all(path.exists() for path in paths.values()):
        import pickle

        df_full = pd.read_parquet(paths["df_full"])
        df_train = pd.read_parquet(paths["df_train"])
        df_val = pd.read_parquet(paths["df_val"])
        df_test = pd.read_parquet(paths["df_test"])
        candidate_trades = pd.read_parquet(paths["candidate_trades"])
        c_train = pd.read_parquet(paths["c_train"])
        c_val = pd.read_parquet(paths["c_val"])
        c_test = pd.read_parquet(paths["c_test"])
        wallet_metrics = pd.read_parquet(paths["wallet_metrics"])
        hold_metrics = pd.read_parquet(paths["hold_metrics"])
        restricted = pd.read_parquet(paths["restricted"])
        with open(paths["copy_wallets"], "rb") as fh:
            copy_wallets = pickle.load(fh)
        with open(paths["conditions"], "rb") as fh:
            conditions = pickle.load(fh)
        with open(paths["signal_sets"], "rb") as fh:
            signal_sets = pickle.load(fh)
        with open(paths["residual_fit"], "rb") as fh:
            residual_fit = pickle.load(fh)
        engine = PositionSignalEngine(restricted)
        return Stage1Workspace(
            df_full=df_full,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            wallet_metrics=wallet_metrics,
            hold_metrics=hold_metrics,
            copy_wallets=set(copy_wallets),
            candidate_trades=candidate_trades,
            candidate_splits={"train": c_train, "val": c_val, "test": c_test},
            conditions=set(conditions),
            signal_sets={k: set(v) for k, v in signal_sets.items()},
            engine=engine,
            residual_fit=residual_fit,
            set_tables_cache={},
        )

    ws = build_stage1_workspace(
        tags=tags,
        copy_rules=copy_rules,
        archetype_min_trade_count=archetype_min_trade_count,
    )
    import pickle

    ws.df_full.to_parquet(paths["df_full"])
    ws.df_train.to_parquet(paths["df_train"])
    ws.df_val.to_parquet(paths["df_val"])
    ws.df_test.to_parquet(paths["df_test"])
    ws.candidate_trades.to_parquet(paths["candidate_trades"])
    ws.candidate_splits["train"].to_parquet(paths["c_train"])
    ws.candidate_splits["val"].to_parquet(paths["c_val"])
    ws.candidate_splits["test"].to_parquet(paths["c_test"])
    ws.wallet_metrics.to_parquet(paths["wallet_metrics"])
    ws.hold_metrics.to_parquet(paths["hold_metrics"])
    ws.df_full[ws.df_full["condition_id"].isin(ws.conditions)][_ENGINE_COLS].to_parquet(paths["restricted"])
    with open(paths["copy_wallets"], "wb") as fh:
        pickle.dump(sorted(ws.copy_wallets), fh)
    with open(paths["conditions"], "wb") as fh:
        pickle.dump(sorted(ws.conditions), fh)
    with open(paths["signal_sets"], "wb") as fh:
        pickle.dump({k: sorted(v) for k, v in ws.signal_sets.items()}, fh)
    with open(paths["residual_fit"], "wb") as fh:
        pickle.dump(ws.residual_fit, fh)
    return ws


def attach_position_signal_panel(
    workspace: Stage1Workspace,
    signal_sets: dict[str, Iterable[str]] | None = None,
    *,
    candidate_splits: dict[str, pd.DataFrame] | None = None,
    kinds: list[tuple[str, str]] | None = None,
    fresh_tau_ns: int | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Attach one or more position-signal families to candidate splits."""
    frames = (
        workspace.clone_candidate_splits()
        if candidate_splits is None
        else {name: frame.copy(deep=True) for name, frame in candidate_splits.items()}
    )
    chosen_sets = signal_sets or workspace.signal_sets
    all_kinds = kinds or (
        DEFAULT_FRESH_SIGNAL_KINDS if fresh_tau_ns is not None else DEFAULT_SIGNAL_KINDS
    )

    signal_cols: list[str] = []
    for set_name, wallets in chosen_sets.items():
        cache_key = (set_name, fresh_tau_ns)
        cached_tables = workspace.set_tables_cache.get(cache_key)
        if cached_tables is None:
            A, B = workspace.engine.build_set(
                set(wallets),
                conditions=workspace.conditions,
                fresh_tau_ns=fresh_tau_ns,
            )
            workspace.set_tables_cache[cache_key] = (A, B)
        else:
            A, B = cached_tables
        for frame in frames.values():
            workspace.engine.attach_position_signals(frame, set_name, A, B)
        for kind, var in all_kinds:
            col = f"sig_{kind}_{var}_{set_name}"
            if col in frames["train"].columns:
                signal_cols.append(col)
    return frames, signal_cols


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


def summarize_workspace(workspace: Stage1Workspace) -> pd.DataFrame:
    """Compact workspace summary for notebook display."""
    return pd.DataFrame(
        [
            {
                "copy_wallets": len(workspace.copy_wallets),
                "candidate_trades": len(workspace.candidate_trades),
                "candidate_conditions": len(workspace.conditions),
                "train_candidates": len(workspace.candidate_splits["train"]),
                "val_candidates": len(workspace.candidate_splits["val"]),
                "test_candidates": len(workspace.candidate_splits["test"]),
                "signal_sets": len(workspace.signal_sets),
            }
        ]
    )
