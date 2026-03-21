"""
Continuous signal scoring and threshold selection.

Uses training-period ``open_buy`` events to build calibration tables that
translate raw features (price bucket, consensus) into smooth 0–1 scores.
These tables are then applied to score new signal events in any period.

Main entry-points:

* :func:`build_calibration_tables` — fit price-bucket and consensus score tables
* :func:`apply_signal_score`       — apply fitted tables to any signal DataFrame
* :func:`select_signal_threshold`  — pick the best score threshold from a grid
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_lookup(series: pd.Series) -> pd.Series:
    """Min-max scale a Series to [0, 1], handling empty / constant inputs."""
    s = series.copy()
    if s.empty:
        return s
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Calibration table construction
# ---------------------------------------------------------------------------

def build_calibration_tables(
    calibration_signals: pd.DataFrame,
    price_prior: int = 50,
    consensus_prior: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fit price-bucket and consensus score tables from calibration signals.

    Parameters
    ----------
    calibration_signals:
        ``open_buy`` signal events (train-b period).
        Required columns: ``wallet``, ``price_bucket``, ``prob_edge``,
        ``prior_same_24h``, ``prior_opp_24h``.
    price_prior:
        Laplace-smoothing prior count for the price table.
    consensus_prior:
        Laplace-smoothing prior count for the consensus table.

    Returns
    -------
    (price_table, consensus_table, global_edge)
        *price_table* — DataFrame indexed by ``price_bucket`` with columns
        ``n``, ``sum_edge``, ``smoothed_edge``, ``price_score``

        *consensus_table* — DataFrame indexed by ``(same_bucket, opp_bucket)``
        with columns ``n``, ``sum_edge``, ``smoothed_edge``, ``consensus_score``

        *global_edge* — population-average ``prob_edge`` used as the prior mean
    """
    global_edge: float = float(calibration_signals["prob_edge"].mean())

    # --- price table ---
    price_table = calibration_signals.groupby("price_bucket").agg(
        n=("wallet", "size"),
        sum_edge=("prob_edge", "sum"),
    )
    price_table["smoothed_edge"] = (
        price_table["sum_edge"] + price_prior * global_edge
    ) / (price_table["n"] + price_prior)
    price_table["price_score"] = _scale_lookup(price_table["smoothed_edge"])

    # --- consensus table ---
    consensus_df = calibration_signals.copy()
    consensus_df["same_bucket"] = consensus_df["prior_same_24h"].clip(upper=3)
    consensus_df["opp_bucket"] = consensus_df["prior_opp_24h"].clip(upper=2)

    consensus_table = consensus_df.groupby(["same_bucket", "opp_bucket"]).agg(
        n=("wallet", "size"),
        sum_edge=("prob_edge", "sum"),
    )
    consensus_table["smoothed_edge"] = (
        consensus_table["sum_edge"] + consensus_prior * global_edge
    ) / (consensus_table["n"] + consensus_prior)
    consensus_table["consensus_score"] = _scale_lookup(
        consensus_table["smoothed_edge"]
    )

    return price_table.reset_index(), consensus_table.reset_index(), global_edge


# ---------------------------------------------------------------------------
# Score application
# ---------------------------------------------------------------------------

def apply_signal_score(
    signals: pd.DataFrame,
    price_table: pd.DataFrame,
    consensus_table: pd.DataFrame,
    wallet_weight: float = 0.45,
    conviction_weight: float = 0.15,
    price_weight: float = 0.20,
    consensus_weight: float = 0.20,
) -> pd.DataFrame:
    """Apply calibrated score tables to a signal DataFrame.

    Computes four component scores then combines them into a single
    ``signal_score`` in [0, 1]:

    * **wallet_component**    — ``wallet_quality`` (already 0–1)
    * **conviction_component** — log-scaled conviction ratio, clipped to [0, 1]
    * **price_component**     — lookup from *price_table*
    * **consensus_component** — lookup from *consensus_table*

    Parameters
    ----------
    signals:
        Signal events DataFrame.  Required columns: ``wallet_quality``,
        ``conviction_ratio``, ``price_bucket``, ``prior_same_24h``,
        ``prior_opp_24h``.
    price_table, consensus_table:
        Output of :func:`build_calibration_tables`.
    wallet_weight, conviction_weight, price_weight, consensus_weight:
        Component weights (should sum to 1.0).

    Returns
    -------
    Copy of *signals* with added columns:
        ``wallet_component``, ``conviction_component``,
        ``price_component``, ``consensus_component``, ``signal_score``.
    """
    if signals.empty:
        return signals.copy()

    price_map = price_table.set_index("price_bucket")["price_score"]
    consensus_map = consensus_table.set_index(["same_bucket", "opp_bucket"])[
        "consensus_score"
    ]

    scored = signals.copy()
    scored["same_bucket"] = scored["prior_same_24h"].clip(upper=3)
    scored["opp_bucket"] = scored["prior_opp_24h"].clip(upper=2)

    scored["wallet_component"] = scored["wallet_quality"].fillna(0.0)
    scored["conviction_component"] = np.clip(
        np.log1p(scored["conviction_ratio"]) / np.log(3.0), 0.0, 1.5
    ).clip(0.0, 1.0)
    scored["price_component"] = scored["price_bucket"].map(price_map).fillna(
        price_map.mean()
    )

    consensus_key = list(zip(scored["same_bucket"], scored["opp_bucket"]))
    scored["consensus_component"] = pd.Series(
        [consensus_map.get(k, np.nan) for k in consensus_key], index=scored.index
    ).fillna(consensus_map.mean())

    scored["signal_score"] = (
        wallet_weight * scored["wallet_component"]
        + conviction_weight * scored["conviction_component"]
        + price_weight * scored["price_component"]
        + consensus_weight * scored["consensus_component"]
    )
    return scored


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_signal_threshold(
    scored_signals: pd.DataFrame,
    threshold_grid: np.ndarray | None = None,
    min_signals: int = 30,
) -> float:
    """Pick the score threshold that maximises ``avg_copy_roi_capped`` on *scored_signals*.

    Parameters
    ----------
    scored_signals:
        Output of :func:`apply_signal_score` on the calibration period.
    threshold_grid:
        Array of candidate thresholds (default: ``np.arange(0.50, 0.96, 0.05)``).
    min_signals:
        Minimum number of signals that must pass the threshold for a candidate
        to be considered.  Falls back to the best unconstrained candidate when
        no candidate meets the threshold.

    Returns
    -------
    float
        The selected signal threshold.
    """
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.50, 0.96, 0.05), 2)

    rows = []
    for threshold in threshold_grid:
        subset = scored_signals[scored_signals["signal_score"] >= threshold]
        rows.append(
            {
                "threshold": threshold,
                "signals": len(subset),
                "avg_copy_roi_capped": (
                    subset["copy_roi_capped"].mean() if len(subset) else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    candidates = summary[summary["signals"] >= min_signals].copy()

    if candidates.empty:
        # fall back to best unconstrained row
        best_row = summary.sort_values("avg_copy_roi_capped", ascending=False).iloc[0]
    else:
        best_row = candidates.sort_values("avg_copy_roi_capped", ascending=False).iloc[0]

    return float(best_row["threshold"])
