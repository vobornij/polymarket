"""Verify the CopyWalletQualitySignals lead against a raw-pnl objective.

Hypothesis: copying only the candidate BUYs of the most active train-period copy
wallets (``sig_wal_trade_count``) produces more dollars per capital than
copy-all, under a $10k budget, without test tuning.

Design (everything selected on train or train+val, reported on val/test):

- **Hard top-k rule**: fire only trades whose raw signal >= the selection
  frame's ``(1 - fraction)`` quantile, at full copyable qty (scale = 1.0).
  This is the right sizing rule for a hard copy-filter (the earlier
  score-proportional / Sharpe-scaled sizing gate was the wrong rule for it).
- **Honest fraction selection**: pick the fraction maximizing budget-sim pnl on
  the *selection* frame (train for fold A, train+val for fold B).
- Benchmarks under the same hard rule: copy-all and price-top-50% (deep
  favorites) — to check the edge is not just the price-composition confound.
- Within-price-bin check on test for the winner.
- Block-bootstrap (7-day) Sharpe CI on test daily PnL; cost sensitivity {0,10,30}.

Writes pnl_lead_verify_results.csv and pnl_lead_verify_ci.csv.
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
    sizing_sharpe,
)
from signal_lab.signal_lib import apply_rank_transformer, fit_rank_transformer
from signal_lab.stage1 import load_stage1_data, run_strategies
from signal_lab.strategies import CopyWalletQualitySignals

from evaluate_composite import add_price_bins

BUDGET = 10_000.0
FRACTIONS = np.arange(0.05, 0.51, 0.05)
CANDIDATES = ["sig_wal_trade_count", "sig_wal_copyable_pnl"]


def hard_binary(sel_col: pd.Series, rep_col: pd.Series, fraction: float):
    """Binary fire indicator from the selection frame's quantile."""
    floor = float(sel_col.quantile(1.0 - fraction))
    return (rep_col >= floor).astype(float), floor


def sim_pnl(frame: pd.DataFrame, score_col: str) -> float:
    res = capital_constrained_sim(frame, score_col, BUDGET, 1.0)
    return res["net_pnl"], res


def run_fold(sel: pd.DataFrame, rep: pd.DataFrame, col: str, fraction: float):
    binary, floor = hard_binary(sel[col], rep[col], fraction)
    rep2 = rep.copy(deep=True)
    rep2["__hard"] = binary
    return capital_constrained_sim(rep2, "__hard", BUDGET, 1.0), floor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    print("Loading stage-1 data...", flush=True)
    df_full, _dt, _dv, _dtest, wallet_metrics, hold_metrics = load_stage1_data(
        max_shards=args.max_shards
    )

    print("Running CopyWalletQualitySignals...", flush=True)
    splits, cols = run_strategies(
        df_full, wallet_metrics, hold_metrics, [CopyWalletQualitySignals()]
    )
    print(f"Signal cols: {cols}", flush=True)

    # Benchmarks: copy-all, price-top-50 (deep favorites). Rank-normalized price.
    price_fit = fit_rank_transformer(splits["train"]["price"])
    for name, frame in splits.items():
        frame["__price_rank"] = apply_rank_transformer(frame["price"], price_fit)
        frame["__all"] = 1.0
    price_floor = float(
        pd.concat([splits["train"], splits["val"]], ignore_index=True)["__price_rank"].quantile(0.5)
    )
    for name, frame in splits.items():
        frame["__price_half"] = (frame["__price_rank"] >= price_floor).astype(float)

    sel_a = splits["train"]
    sel_b = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    rows = []
    fraction_profiles = []
    for col in CANDIDATES:
        for fold, sel, rep_name in (
            ("A", sel_a, "val"),
            ("B", sel_b, "test"),
        ):
            rep = splits[rep_name]
            # Honest fraction selection on the selection frame by budget-sim pnl.
            pnl_by_frac = []
            for f in FRACTIONS:
                res, _ = run_fold(sel, sel, col, f)
                pnl_by_frac.append({
                    "fraction": f,
                    "sel_pnl": res["net_pnl"],
                    "sel_roi_w": res["net_pnl"] / res["notional"] if res["notional"] > 0 else np.nan,
                    "sel_trades": res["trades"],
                })
            profile = pd.DataFrame(pnl_by_frac)
            profile.insert(0, "col", col)
            profile.insert(1, "fold", fold)
            fraction_profiles.append(profile)
            best_f = profile.sort_values("sel_pnl", ascending=False).iloc[0]["fraction"]

            res, floor = run_fold(sel, rep, col, best_f)
            rows.append({
                "col": col, "fold": fold, "split": rep_name,
                "fraction": best_f, "floor": round(floor, 3),
                "trades": res["trades"],
                "pnl": round(res["net_pnl"], 2),
                "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
                "pnl_per_peak": round(res["net_pnl"] / res["peak_used"], 4) if res["peak_used"] > 0 else 0.0,
                "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                "mean_used": round(res["mean_used"], 2),
            })

    for label, score_col in (("copy_all", "__all"), ("price_top50", "__price_half")):
        for fold, sel, rep_name in (("A", sel_a, "val"), ("B", sel_b, "test")):
            res = capital_constrained_sim(splits[rep_name], score_col, BUDGET, 1.0)
            rows.append({
                "col": label, "fold": fold, "split": rep_name,
                "fraction": np.nan, "floor": np.nan,
                "trades": res["trades"],
                "pnl": round(res["net_pnl"], 2),
                "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
                "pnl_per_peak": round(res["net_pnl"] / res["peak_used"], 4) if res["peak_used"] > 0 else 0.0,
                "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                "mean_used": round(res["mean_used"], 2),
            })

    results = pd.DataFrame(rows)
    profiles = pd.concat(fraction_profiles, ignore_index=True)
    results.to_csv("pnl_lead_verify_results.csv", index=False)
    profiles.to_csv("pnl_lead_verify_fractions.csv", index=False)

    print("\n" + "=" * 70, flush=True)
    print("Hard top-k sizing, scale=1.0, $10k budget", flush=True)
    print("=" * 70, flush=True)
    print(results.to_string(index=False), flush=True)

    # Winner on the deployment fold (B), by test pnl among candidates.
    fb = results[results["fold"] == "B"]
    cand_b = fb[fb["col"].isin(CANDIDATES)]
    win = cand_b.sort_values("pnl", ascending=False).iloc[0]

    # Within-price-bin confound check for the winner on test.
    print("\n" + "=" * 70, flush=True)
    print(f"Within-price-bin check for {win['col']} (test, fold B params)", flush=True)
    print("=" * 70, flush=True)
    add_price_bins(splits)
    binary, floor = hard_binary(sel_b[win["col"]], splits["test"][win["col"]], win["fraction"])
    t = splits["test"].copy(deep=True)
    t["__sel"] = binary.astype(bool)
    bin_rows = []
    for d in sorted(t["price_dec"].dropna().unique()):
        g = t[t["price_dec"] == d]
        gs = g[g["__sel"]]
        cnot = g["copyable_notional"].sum()
        cnot_s = gs["copyable_notional"].sum()
        bin_rows.append({
            "price_dec": int(d), "n": len(g), "n_sel": len(gs),
            "pnl_all": round(float(g["copyable_pnl"].sum()), 1),
            "pnl_sel": round(float(gs["copyable_pnl"].sum()), 1),
            "roi_w_all": round(float(g["copyable_pnl"].sum() / cnot), 4) if cnot > 0 else np.nan,
            "roi_w_sel": round(float(gs["copyable_pnl"].sum() / cnot_s), 4) if cnot_s > 0 else np.nan,
        })
    bin_df = pd.DataFrame(bin_rows)
    print(bin_df.to_string(index=False), flush=True)
    bin_df.to_csv("pnl_lead_verify_pricebins.csv", index=False)

    # Bootstrap Sharpe CI + cost sensitivity for winner vs copy-all on test.
    print("\n" + "=" * 70, flush=True)
    print(f"Bootstrap Sharpe CI (7-day blocks) + cost sensitivity, winner={win['col']}", flush=True)
    print("=" * 70, flush=True)
    rows_ci = []
    # re-run winner with cost by rebuilding the binary frame
    for cost in (0.0, 10.0, 30.0):
        t2 = splits["test"].copy(deep=True)
        t2["__hard"] = binary
        res_c = capital_constrained_sim(t2, "__hard", BUDGET, 1.0, cost_bps=cost)
        point, lo, hi = block_bootstrap_sharpe(res_c["daily_pnl"], block_size=7, n_iter=1000, seed=42)
        rows_ci.append({
            "design": f"wallet@{win['fraction']:.2f}", "cost_bps": cost,
            "pnl": round(res_c["net_pnl"], 2),
            "roi_w": round(res_c["net_pnl"] / res_c["notional"], 4) if res_c["notional"] > 0 else np.nan,
            "sharpe_daily": round(sizing_sharpe(res_c["daily_pnl"], 365.0), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
        })
    for label, col in (("copy_all", "__all"), ("price_top50", "__price_half")):
        res_c = capital_constrained_sim(splits["test"], col, BUDGET, 1.0, cost_bps=10.0)
        point, lo, hi = block_bootstrap_sharpe(res_c["daily_pnl"], block_size=7, n_iter=1000, seed=42)
        rows_ci.append({
            "design": label, "cost_bps": 10.0,
            "pnl": round(res_c["net_pnl"], 2),
            "roi_w": round(res_c["net_pnl"] / res_c["notional"], 4) if res_c["notional"] > 0 else np.nan,
            "sharpe_daily": round(sizing_sharpe(res_c["daily_pnl"], 365.0), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
        })
    ci_df = pd.DataFrame(rows_ci)
    print(ci_df.to_string(index=False), flush=True)
    ci_df.to_csv("pnl_lead_verify_ci.csv", index=False)
    print("\nSaved pnl_lead_verify_{results,fractions,pricebins,ci}.csv", flush=True)


if __name__ == "__main__":
    main()
