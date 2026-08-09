"""Decomposition + confound diagnostics for the per-wallet scaling result.

Isolates the tier3@2-0 win into (a) the drop of the bottom tier, (b) the 2x
scale of the top tier, and checks the cheap-price / concentration confounds
that killed prior selectors.  Uses the cached candidate splits (fast re-run).
"""

from __future__ import annotations

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

from signal_lab.sizing import capital_constrained_sim, sizing_sharpe
from signal_lab.wallet_scaling import (
    alpha_tier as build_alpha_tier,
    wallet_daily_pnl,
    wallet_stats,
)

BUDGET = 10_000.0
CACHE = Path("/var/folders/j8/0dbnwk8n6m933m843h7hb88w0000gn/T/opencode/wallet_scaling_splits")


def main():
    splits = {s: pd.read_parquet(CACHE / f"{s}.parquet") for s in ("train", "val", "test")}
    for fr in splits.values():
        fr["score1"] = 1.0

    st = wallet_stats(wallet_daily_pnl(splits["train"]))
    n_tier = 3
    alpha_max = 2.0
    alpha_min = 0.0
    a_tier = build_alpha_tier(st, n_tier, alpha_max, alpha_min)
    # tier: 2 = top (a_max), 1 = middle (a_min-ish), 0 = dropped
    tier_map = pd.Series(np.where(a_tier > 0, np.where(a_tier >= 0.9 * a_tier.max(), 2, 1), 0),
                         index=a_tier.index)
    print("tier counts:", tier_map.value_counts().sort_index().to_dict())
    for name in splits:
        splits[name]["tier"] = splits[name]["wallet"].map(tier_map).fillna(1).astype(int)

    # Per-tier profile (test)
    print("\n=== per-tier profile (test, unconstrained) ===")
    rows = []
    for name in ("train", "val", "test"):
        fr = splits[name]
        for t in (2, 1, 0):
            g = fr[fr["tier"] == t]
            notl = float(g["copyable_notional"].sum())
            rows.append({
                "split": name, "tier": t, "wallets": g["wallet"].nunique(),
                "trades": len(g), "notional": notl,
                "pnl": round(float(g["copyable_pnl"].sum()), 0),
                "mean_price": round(float(g["price"].mean()), 3),
                "mean_per_trade_pnl": round(float(g["copyable_pnl"].sum()) / max(len(g), 1), 2),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    # Decomposition sims
    print("\n=== decomposition sims (10bps) ===")
    for name in ("val", "test"):
        fr = splits[name]
        base = fr.copy()
        alpha_skip = np.where(base["tier"] > 0, 1.0, 0.0)
        alpha_tier = np.where(base["tier"] == 2, 2.0, np.where(base["tier"] == 1, 1.0, 0.0))
        base["alpha_skip"] = alpha_skip
        base["alpha_tier"] = alpha_tier
        designs = [
            ("copy_all",        "score1", 1.0, "copyable_qty_5m_100"),
            ("drop_only",       "alpha_skip", 1.0, "copyable_qty_5m_100"),
            ("drop_only_depth", "alpha_skip", 1.0, "bucket_avail_copy_qty"),
            ("tier3@2-0",       "alpha_tier", 1.0, "bucket_avail_copy_qty"),
        ]
        print(f"-- {name} --")
        for label, col, scale, cap in designs:
            res = capital_constrained_sim(base, "score1", BUDGET, scale, cost_bps=10.0,
                                          alpha_col=col, cap_col=cap)
            print(f"  {label:>15s}  trades={res['trades']:>7,}  pnl={res['net_pnl']:>10,.0f}  "
                  f"roi_w={res['net_pnl']/res['notional']:.4f}  sharpe={sizing_sharpe(res['daily_pnl'],365):.3f}  "
                  f"mean_used={res['mean_used']:>8,.0f}  peak={res['peak_used']:>8,.0f}", flush=True)

    # Price-bin confound for top tier vs copy-all (unconstrained test pnl)
    print("\n=== price-decile test pnl: top-tier vs all (unconstrained) ===")
    test = splits["test"]
    dec = pd.qcut(test["price"], 10, labels=False, duplicates="drop")
    test = test.assign(pdec=dec)
    all_pnl = test.groupby("pdec", observed=True).agg(
        n=("copyable_pnl", "size"), pnl=("copyable_pnl", "sum"), mean_price=("price", "mean"))
    all_pnl["mean_per_trade"] = all_pnl["pnl"] / all_pnl["n"].clip(lower=1)
    top = test[test["tier"] == 2]
    top_pnl = top.groupby("pdec", observed=True).agg(
        n=("copyable_pnl", "size"), pnl=("copyable_pnl", "sum"), mean_price=("price", "mean"))
    top_pnl["mean_per_trade"] = top_pnl["pnl"] / top_pnl["n"].clip(lower=1)
    out = pd.DataFrame({
        "pdec": all_pnl.index,
        "all_n": all_pnl["n"], "all_pnl": all_pnl["pnl"].round(0),
        "all_mpt": all_pnl["mean_per_trade"].round(2), "all_price": all_pnl["mean_price"].round(3),
        "top_n": top_pnl["n"], "top_pnl": top_pnl["pnl"].round(0),
        "top_mpt": top_pnl["mean_per_trade"].round(2), "top_price": top_pnl["mean_price"].round(3),
    })
    print(out.to_string(index=False))

    # Concentration
    print("\n=== concentration (test) ===")
    for lbl, fr in (("top-tier", test[test["tier"] == 2]), ("all", test)):
        m = fr.groupby("condition_id")["copyable_pnl"].sum().sort_values(ascending=False)
        share5 = m.head(5).sum() / m.sum()
        share1 = m.head(1).sum() / m.sum()
        print(f"  {lbl:>9s}: markets={len(m):,}  top1 market share={share1:.3f}  top5 share={share5:.3f}  "
              f"total_pnl={m.sum():,.0f}", flush=True)

    # Why alpha_max is insensitive: cap-binding fraction on top-tier test trades
    print("\n=== cap-binding on top-tier (test) ===")
    top = test[test["tier"] == 2]
    q = top["bucket_avail_copy_qty"]
    cq = top["copyable_qty_5m_100"]
    cap_bound = (q <= 1.05 * cq)
    print(f"  top-tier trades: {len(top):,}  depth-capped (depth<=1.05*copyable): {cap_bound.mean()*100:.1f}%  "
          f"median depth/copyable={ (q/cq).median():.2f}")


if __name__ == "__main__":
    main()
