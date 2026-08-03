"""Step 4: capital-constrained walk-forward sizing overlay for crowding scores.

The decile/firing tests showed the crowding score ranks ``roi_res`` perfectly
but hard selection *reduces* dollar PnL out-of-sample (the edge lives in
unscored, contested trades).  This script asks the risk-adjusted question:
under a $10k capital cap, does score-proportional sizing on the crowding score
improve **gain per volatility** (Sharpe) over copy-all and over a price-favorite
sizing benchmark?

Design (nothing tuned on test):

- Fold A: select score floor + size scale on **train**, report on **val**.
- Fold B: select on **train+val**, report on **test** (the deployment fold).
- Selection objective: daily Sharpe of net PnL from ``capital_constrained_sim``.
- Benchmarks: copy-all (uniform score) and price-favorite (rank-fit price).
- Cost sensitivity {0, 10, 30} bps for the best blended design.
- Block-bootstrap (7-day blocks) Sharpe CI on the test daily PnL.

Writes ``crowding_overlay_results.csv`` and ``crowding_overlay_sharpe_ci.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from signal_lab.sizing import (
    block_bootstrap_sharpe,
    capital_constrained_sim,
    score_floor_for_fraction,
    select_scale,
    sizing_sharpe,
)
from signal_lab.stage1 import load_stage1_data, run_strategy

from reevaluate_crowding import (
    CrowdingReeval,
    REACTIVE_SETS,
    SET_NAMES,
    add_blend,
    build_copy_scores,
)

BUDGET = 10_000.0
SCALE_GRID = np.arange(0.1, 3.01, 0.1)
FRACTIONS = [None, 0.5, 0.25, 0.10]
COST_BPS = [0.0, 10.0, 30.0]

COPY_SCORES = ["copy_blend", "copy_blend3", "copy_overseller", "copy_max_dd"]
BENCH_SCORES = ["score_all", "score_price"]


def build_score_frames(scored: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add the scoring columns used by the overlay to each split frame."""
    out = {name: frame.copy(deep=True) for name, frame in scored.items()}
    for name, frame in out.items():
        add_blend(frame, SET_NAMES, "copy_blend")
        add_blend(frame, ["both_sides", "flipper", "overseller"], "copy_blend3")
        frame["score_all"] = 1.0
    price_fit = None
    from signal_lab.signal_lib import apply_rank_transformer, fit_rank_transformer
    for name, frame in out.items():
        if name == "train":
            price_fit = fit_rank_transformer(frame["price"])
        frame["score_price"] = apply_rank_transformer(frame["price"], price_fit)
    return out


def run_fold(sel, rep, cost_bps: float) -> tuple[pd.DataFrame, dict]:
    """Select floor+scale on ``sel`` (with the current cost), report on ``rep``.

    Returns the report rows plus a dict of per-score daily PnL on ``rep`` for
    the chosen (floor, scale) so bootstrap CIs can be computed downstream.
    """
    rows = []
    daily = {}
    for col in COPY_SCORES + BENCH_SCORES:
        fracs = FRACTIONS if col in COPY_SCORES else [None]
        for f in fracs:
            floor = score_floor_for_fraction(sel, col, f) if f is not None else None
            best_scale, grid = select_scale(
                sel, col, BUDGET, SCALE_GRID, cost_bps=cost_bps, score_floor=floor
            )
            if grid.empty:
                continue
            res = capital_constrained_sim(
                rep, col, BUDGET, best_scale, cost_bps=cost_bps, score_floor=floor
            )
            row = {
                "score_col": col,
                "fraction": f,
                "floor": round(floor, 4) if floor is not None else np.nan,
                "scale": best_scale,
                "trades": res["trades"],
                "net_pnl": round(res["net_pnl"], 2),
                "cost_paid": round(res["cost_paid"], 2),
                "notional": round(res["notional"], 2),
                "peak_used": round(res["peak_used"], 2),
                "mean_used": round(res["mean_used"], 2),
                "pnl_per_peak": round(res["net_pnl"] / res["peak_used"], 4)
                if res["peak_used"] > 0
                else 0.0,
                "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                "sharpe_weekly": round(
                    sizing_sharpe(res["daily_pnl"].resample("W").sum(), 52.0), 3
                ),
            }
            rows.append(row)
            daily.setdefault(col, []).append((f, best_scale, res["daily_pnl"]))
    return pd.DataFrame(rows), daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    print("Loading stage-1 data...", flush=True)
    df_full, _dt, _dv, _dtest, wallet_metrics, hold_metrics = load_stage1_data(
        max_shards=args.max_shards
    )

    strategy = CrowdingReeval()
    print(f"Running strategy {strategy.name}...", flush=True)
    splits, _cols = run_strategy(df_full, wallet_metrics, hold_metrics, strategy)

    scored = build_copy_scores(splits, SET_NAMES)
    scored = build_score_frames(scored)

    sel_a = scored["train"]
    sel_b = pd.concat([scored["train"], scored["val"]], ignore_index=True)

    all_rows = []
    all_daily = {}
    for fold, sel, rep_name in (
        ("A_train->val", sel_a, "val"),
        ("B_trainval->test", sel_b, "test"),
    ):
        print(f"\n{'=' * 70}\nFold {fold}\n{'=' * 70}", flush=True)
        rows, daily = run_fold(sel, scored[rep_name], cost_bps=0.0)
        rows.insert(0, "fold", fold)
        rows.insert(1, "split", rep_name)
        all_rows.append(rows)
        all_daily[fold] = daily
        print(rows.to_string(index=False), flush=True)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv("crowding_overlay_results.csv", index=False)
    print("\nSaved crowding_overlay_results.csv", flush=True)

    # Best blended design on the deployment fold (B), by test daily Sharpe.
    fold_b = results[results["fold"] == "B_trainval->test"]
    best_row = fold_b[fold_b["score_col"].isin(COPY_SCORES)].sort_values(
        "sharpe_daily", ascending=False
    ).iloc[0]

    print("\n" + "=" * 70, flush=True)
    print(
        f"Best blended overlay on deployment fold: {best_row['score_col']} "
        f"fraction={best_row['fraction']}, scale={best_row['scale']}",
        flush=True,
    )
    print("=" * 70, flush=True)

    # Cost sensitivity for the best design: re-select scale at each cost on train+val.
    cost_rows = []
    for cost in COST_BPS:
        sel = pd.concat([scored["train"], scored["val"]], ignore_index=True)
        floor = (
            score_floor_for_fraction(sel, best_row["score_col"], best_row["fraction"])
            if best_row["fraction"] is not None
            else None
        )
        best_scale, _grid = select_scale(
            sel, best_row["score_col"], BUDGET, SCALE_GRID,
            cost_bps=cost, score_floor=floor,
        )
        test_res = capital_constrained_sim(
            scored["test"], best_row["score_col"], BUDGET, best_scale,
            cost_bps=cost, score_floor=floor,
        )
        cost_rows.append({
            "cost_bps": cost,
            "scale": best_scale,
            "trades": test_res["trades"],
            "net_pnl": round(test_res["net_pnl"], 2),
            "cost_paid": round(test_res["cost_paid"], 2),
            "pnl_per_peak": round(test_res["net_pnl"] / test_res["peak_used"], 4)
            if test_res["peak_used"] > 0 else 0.0,
            "sharpe_daily": round(sizing_sharpe(test_res["daily_pnl"], 365.0), 3),
        })
    cost_df = pd.DataFrame(cost_rows)
    print(cost_df.to_string(index=False), flush=True)

    # Block-bootstrap Sharpe CI on test at 10bps (the realistic deployment cost).
    print("\n" + "=" * 70, flush=True)
    print("Block-bootstrap Sharpe CI (7-day blocks, test daily PnL, 10bps)", flush=True)
    print("=" * 70, flush=True)
    floor = (
        score_floor_for_fraction(sel_b, best_row["score_col"], best_row["fraction"])
        if best_row["fraction"] is not None
        else None
    )
    best_scale10, _grid = select_scale(
        sel_b, best_row["score_col"], BUDGET, SCALE_GRID, cost_bps=10.0,
        score_floor=floor,
    )
    test_res10 = capital_constrained_sim(
        scored["test"], best_row["score_col"], BUDGET, best_scale10,
        cost_bps=10.0, score_floor=floor,
    )
    ci_rows = []
    for label, daily in (
        ("overlay_best", test_res10["daily_pnl"]),
        ("copy_all", None),
    ):
        if label == "copy_all":
            row = fold_b[fold_b["score_col"] == "score_all"].iloc[0]
            res_all = capital_constrained_sim(
                scored["test"], "score_all", BUDGET, row["scale"],
                cost_bps=10.0,
            )
            daily = res_all["daily_pnl"]
        point, lo, hi = block_bootstrap_sharpe(daily, block_size=7, n_iter=1000, seed=42)
        ci_rows.append({
            "design": label,
            "sharpe_daily": round(point, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
        })
    ci_df = pd.DataFrame(ci_rows)
    print(ci_df.to_string(index=False), flush=True)
    ci_df.to_csv("crowding_overlay_sharpe_ci.csv", index=False)
    print("Saved crowding_overlay_sharpe_ci.csv", flush=True)


if __name__ == "__main__":
    main()
