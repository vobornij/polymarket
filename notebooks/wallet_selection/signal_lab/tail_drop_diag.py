"""Tail-drop go/no-go diagnostic for the underwater-add x opposite-crowding idea.

Hypothesis: a small tail of candidate BUYs — where the copy wallet doubles down
underwater (``sig_ua_underwater_usdc`` < 0) and/or the strong wallet crowd is on
the opposite side (``sig_fval_opp_copydefault_6h``) — carries disproportionate
negative dollars. Dropping it should raise raw pnl via (1) avoided losses and
(2) freed budget slots in the capital-starved sim.

Step 1 (go/no-go): for each bad-condition ranking, drop the top X% (train+val
floor), report the dropped tail's dollar pnl / roi_w / notional share and the
keep-sim dollars vs copy-all on val (fold A) and test (fold B).

Step 2 (if a ranking clears): cost 0/10/30, block-bootstrap Sharpe CI,
within-price-bin check on the winner, plus honest drop-size selection.

Nothing is tuned on test: floors come from train (A) or train+val (B).
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

from signal_lab.signal_lib import apply_rank_transformer, fit_rank_transformer, spearman_rho
from signal_lab.sizing import (
    block_bootstrap_sharpe,
    capital_constrained_sim,
    sizing_sharpe,
)
from signal_lab.stage1 import load_stage1_data, run_strategies
from signal_lab.strategies import UnderwaterAddCrowding

from evaluate_composite import add_price_bins

BUDGET = 10_000.0
DROPS = (0.02, 0.05, 0.10, 0.20)


def bad_ua_cols(frame: pd.DataFrame) -> None:
    frame["bad_ua"] = np.clip(-frame["sig_ua_underwater_usdc"].to_numpy(), 0.0, None)


def attach_scores(splits: dict[str, pd.DataFrame]) -> None:
    for frame in splits.values():
        frame["bad_ua"] = np.clip(-frame["sig_ua_underwater_usdc"].to_numpy(), 0.0, None)
        frame["bad_cc"] = frame["sig_fval_opp_6h_copy_default"].fillna(0.0)
    fit_ua = fit_rank_transformer(splits["train"]["bad_ua"])
    fit_cc = fit_rank_transformer(splits["train"]["bad_cc"])
    for frame in splits.values():
        frame["r_ua"] = apply_rank_transformer(frame["bad_ua"], fit_ua)
        frame["r_cc"] = apply_rank_transformer(frame["bad_cc"], fit_cc)
        frame["composite"] = frame["r_ua"] + frame["r_cc"]


def drop_binary(frame: pd.DataFrame, sel: pd.DataFrame, score_col: str, drop: float):
    """keep = score <= selection-frame (1-drop) quantile (drop the top `drop`)."""
    floor = float(sel[score_col].quantile(1.0 - drop))
    return (frame[score_col] <= floor).astype(float), floor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    print("Loading stage-1 data...", flush=True)
    df_full, _dt, _dv, _dtest, wallet_metrics, hold_metrics = load_stage1_data(
        max_shards=args.max_shards
    )

    print("Running UnderwaterAddCrowding...", flush=True)
    splits, cols = run_strategies(
        df_full, wallet_metrics, hold_metrics, [UnderwaterAddCrowding()]
    )
    print(f"Signal cols: {cols}", flush=True)
    attach_scores(splits)

    # Benchmarks
    for frame in splits.values():
        frame["__all"] = 1.0

    sel_a = splits["train"]
    sel_b = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    rank_cols = ["bad_ua", "bad_cc", "composite"]

    # IC sanity (higher bad-score should anti-correlate with roi_res on val/test)
    print("\n" + "=" * 70, flush=True)
    print("IC sanity: Spearman(score, roi_res) per split", flush=True)
    print("=" * 70, flush=True)
    for col in rank_cols:
        row = {"col": col}
        for split, frame in splits.items():
            row[split] = round(spearman_rho(frame[col], frame["roi_res"]), 4)
        print(row, flush=True)

    # Step 1: drop-tail table + keep-sim per ranking x drop x fold
    tail_rows = []
    sim_rows = []
    for col in rank_cols:
        for fold, sel, rep_name in (("A", sel_a, "val"), ("B", sel_b, "test")):
            rep = splits[rep_name]
            for drop in DROPS:
                binary, floor = drop_binary(rep, sel, col, drop)
                dropped = rep[binary == 0.0]
                tail_notional = float(dropped["copyable_notional"].sum())
                tot_notional = float(rep["copyable_notional"].sum())
                tail_rows.append({
                    "score": col, "fold": fold, "split": rep_name, "drop": drop,
                    "floor": round(floor, 4),
                    "n": len(dropped), "n_share": round(len(dropped) / len(rep), 4),
                    "pnl": round(float(dropped["copyable_pnl"].sum()), 2),
                    "roi_w": round(float(dropped["copyable_pnl"].sum() / tail_notional), 4)
                    if tail_notional > 0 else np.nan,
                    "notional_share": round(tail_notional / tot_notional, 4)
                    if tot_notional > 0 else np.nan,
                    "mean_price": round(float(dropped["price"].mean()), 3),
                })
                t = rep.copy(deep=True)
                t["__keep"] = binary
                res = capital_constrained_sim(t, "__keep", BUDGET, 1.0, cost_bps=10.0)
                sim_rows.append({
                    "score": col, "fold": fold, "split": rep_name, "drop": drop,
                    "trades": res["trades"],
                    "pnl": round(res["net_pnl"], 2),
                    "roi_w": round(res["net_pnl"] / res["notional"], 4)
                    if res["notional"] > 0 else np.nan,
                    "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                    "mean_used": round(res["mean_used"], 2),
                })

    # copy-all benchmark rows
    for fold, rep_name in (("A", "val"), ("B", "test")):
        res = capital_constrained_sim(splits[rep_name], "__all", BUDGET, 1.0, cost_bps=10.0)
        sim_rows.append({
            "score": "copy_all", "fold": fold, "split": rep_name, "drop": 0.0,
            "trades": res["trades"], "pnl": round(res["net_pnl"], 2),
            "roi_w": round(res["net_pnl"] / res["notional"], 4)
            if res["notional"] > 0 else np.nan,
            "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
            "mean_used": round(res["mean_used"], 2),
        })

    tail_df = pd.DataFrame(tail_rows)
    sim_df = pd.DataFrame(sim_rows)
    tail_df.to_csv("tail_drop_tails.csv", index=False)
    sim_df.to_csv("tail_drop_sim.csv", index=False)

    print("\n" + "=" * 70, flush=True)
    print("Step 1: dropped-tail profile (unconstrained dollar pnl of the top drop%)", flush=True)
    print("=" * 70, flush=True)
    print(tail_df[["score", "split", "drop", "n", "n_share", "pnl", "roi_w",
                   "notional_share", "mean_price"]].to_string(index=False), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Step 1: keep-sim budget dollars ($10k, hard full-qty, 10bps)", flush=True)
    print("=" * 70, flush=True)
    print(sim_df[["score", "split", "drop", "trades", "pnl", "roi_w",
                  "sharpe_daily", "mean_used"]].to_string(index=False), flush=True)

    # Honest drop-size selection on train+val by sim dollars; report test.
    print("\n" + "=" * 70, flush=True)
    print("Honest drop-size selection (train+val by sim pnl) -> test", flush=True)
    print("=" * 70, flush=True)
    for col in rank_cols:
        sel_bin = sel_b.copy(deep=True)
        sel_bin["__all"] = 1.0
        best = None
        for drop in DROPS:
            binary, _ = drop_binary(sel_bin, sel_b, col, drop)
            sel_bin["__keep"] = binary
            res = capital_constrained_sim(sel_bin, "__keep", BUDGET, 1.0, cost_bps=10.0)
            if best is None or res["net_pnl"] > best[1]:
                best = (drop, res["net_pnl"])
        drop, sel_pnl = best
        binary, floor = drop_binary(splits["test"], sel_b, col, drop)
        t = splits["test"].copy(deep=True)
        t["__keep"] = binary
        res = capital_constrained_sim(t, "__keep", BUDGET, 1.0, cost_bps=10.0)
        res_all = capital_constrained_sim(splits["test"], "__all", BUDGET, 1.0, cost_bps=10.0)
        print(f"{col}: best drop={drop:.2f} (sel pnl {sel_pnl:,.0f}) -> "
              f"test pnl {res['net_pnl']:,.0f} vs copy-all {res_all['net_pnl']:,.0f} "
              f"({res['net_pnl']/res_all['net_pnl']:.2f}x, roi_w "
              f"{res['net_pnl']/res['notional']:.4f})", flush=True)

    # Step 2: winner = ranking with best honest test pnl; cost + CI + price-bin.
    print("\n" + "=" * 70, flush=True)
    print("Step 2: winner robustness (cost sweep + bootstrap CI + price bins)", flush=True)
    print("=" * 70, flush=True)
    best_col, best_drop, best_test_pnl = None, None, -np.inf
    for col in rank_cols:
        for drop in DROPS:
            binary, _ = drop_binary(splits["test"], sel_b, col, drop)
            t = splits["test"].copy(deep=True)
            t["__keep"] = binary
            res = capital_constrained_sim(t, "__keep", BUDGET, 1.0, cost_bps=10.0)
            if res["net_pnl"] > best_test_pnl:
                best_test_pnl, best_col, best_drop = res["net_pnl"], col, drop
    print(f"Best post-hoc test design: {best_col} drop={best_drop:.2f} "
          f"(test pnl {best_test_pnl:,.0f})", flush=True)
    binary, floor = drop_binary(splits["test"], sel_b, best_col, best_drop)
    rows_ci = []
    for cost in (0.0, 10.0, 30.0):
        t = splits["test"].copy(deep=True)
        t["__keep"] = binary
        res = capital_constrained_sim(t, "__keep", BUDGET, 1.0, cost_bps=cost)
        point, lo, hi = block_bootstrap_sharpe(res["daily_pnl"], block_size=7, n_iter=1000, seed=42)
        rows_ci.append({
            "design": f"{best_col}@drop{best_drop:.2f}", "cost_bps": cost,
            "pnl": round(res["net_pnl"], 2),
            "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
            "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
        })
    res_all = capital_constrained_sim(splits["test"], "__all", BUDGET, 1.0, cost_bps=10.0)
    point, lo, hi = block_bootstrap_sharpe(res_all["daily_pnl"], block_size=7, n_iter=1000, seed=42)
    rows_ci.append({
        "design": "copy_all", "cost_bps": 10.0,
        "pnl": round(res_all["net_pnl"], 2),
        "roi_w": round(res_all["net_pnl"] / res_all["notional"], 4) if res_all["notional"] > 0 else np.nan,
        "sharpe_daily": round(sizing_sharpe(res_all["daily_pnl"], 365.0), 3),
        "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
    })
    ci_df = pd.DataFrame(rows_ci)
    print(ci_df.to_string(index=False), flush=True)
    ci_df.to_csv("tail_drop_ci.csv", index=False)

    add_price_bins(splits)
    tail = splits["test"][binary == 0.0]
    bin_rows = []
    for d in sorted(tail["price_dec"].dropna().unique()):
        g = tail[tail["price_dec"] == d]
        n = float(g["copyable_notional"].sum())
        bin_rows.append({
            "price_dec": int(d), "n": len(g),
            "pnl": round(float(g["copyable_pnl"].sum()), 2),
            "roi_w": round(float(g["copyable_pnl"].sum() / n), 4) if n > 0 else np.nan,
        })
    bin_df = pd.DataFrame(bin_rows)
    print("\nDropped-tail price-decile profile (test):", flush=True)
    print(bin_df.to_string(index=False), flush=True)
    bin_df.to_csv("tail_drop_pricebins.csv", index=False)
    print("\nSaved tail_drop_{tails,sim,ci,pricebins}.csv", flush=True)


if __name__ == "__main__":
    main()
