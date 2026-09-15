"""Verify the exposure-threshold exploration artefacts.

Checks the CSVs written by ``explore_exposure_thresholds.py``: threshold-table
schema, fraction coverage, train self-consistency (realized capture >= frac),
and presence/format of the IC / sizing outputs.  Exit code 0 on pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECTED_FRACS = {1.0, 0.8, 0.5, 0.2}

errors: list[str] = []


def check(pred: bool, msg: str) -> None:
    if pred:
        print(f"  ok  {msg}", flush=True)
    else:
        errors.append(msg)
        print(f"  FAIL {msg}", flush=True)


def load(name: str) -> pd.DataFrame:
    path = HERE / name
    if not path.exists():
        errors.append(f"missing artefact {name}")
        print(f"  FAIL missing artefact {name}", flush=True)
        return pd.DataFrame()
    return pd.read_csv(path)


print("== exposure_thresholds.csv ==", flush=True)
th = load("exposure_thresholds.csv")
if not th.empty:
    check(set(th["frac"].unique()) == EXPECTED_FRACS, "fracs == {1.0, 0.8, 0.5, 0.2}")
    check((th["threshold"] >= 0).all(), "all thresholds >= 0")
    check(th.duplicated(["wallet", "frac"]).sum() == 0, "unique (wallet, frac) rows")

print("\n== exposure_capture_curve.csv ==", flush=True)
cv = load("exposure_capture_curve.csv")
if not cv.empty:
    check({"wallet", "exposure_pctile", "exposure", "cum_cpnl_frac"} <= set(cv.columns),
          "schema has wallet/pctile/exposure/cum_cpnl_frac")
    check(cv["exposure_pctile"].between(0, 1).all(), "exposure_pctile within [0, 1]")

print("\n== exposure_capture_summary.csv ==", flush=True)
sm = load("exposure_capture_summary.csv")
if not sm.empty:
    check(set(sm["frac"].unique()) <= EXPECTED_FRACS, "frac subset of targets")
    check(sm["retention"].between(0, 1).all(), "retention within [0, 1]")
    train = sm[sm["split"] == "train"]
    for f in EXPECTED_FRACS:
        row = train[train["frac"] == f]
        if not row.empty:
            check(
                float(row["realized_capture"].iloc[0]) >= f - 0.01,
                f"train realized capture >= {f} (self-consistency)",
            )

print("\n== exposure_ic_report_{pnl,roi}.csv ==", flush=True)
for label in ("pnl", "roi"):
    rep = load(f"exposure_ic_report_{label}.csv")
    if not rep.empty:
        check({"signal", "IC_train", "IC_val", "IC_test", "spearman_price"} <= set(rep.columns),
              f"IC panel schema ({label})")
        has_wt = {"sig_exp_wt_rank", "sig_exp_wt_minmax", "sig_exp_marg", "sig_exp_marg_wt"}
        check(has_wt <= set(rep["signal"]), f"wallet-relative signals present ({label})")
        check(np.isfinite(rep["wb_ic_roi"]).all(), f"wb_ic_roi finite ({label})")

print("\n== exposure_sim.csv / exposure_ci.csv ==", flush=True)
sim = load("exposure_sim.csv")
if not sim.empty:
    check({"design", "split", "sharpe_daily", "pnl"} <= set(sim.columns), "sim schema")
    test = sim[sim["split"] == "test"]
    check({"test"} <= set(sim["split"].unique()), "test single-pass present")
    check({"exposure_top80", "exposure_top50", "exposure_top20"} <= set(test["design"]),
          "quantile-top designs on test")
ci = load("exposure_ci.csv")
if not ci.empty:
    check({"design", "cost_bps", "sharpe_daily", "ci_lo", "ci_hi"} <= set(ci.columns),
          "ci schema")
    check({"exposure_top80", "exposure_top50", "exposure_top20"} <= set(ci["design"]),
          "quantile-top designs in CI")

print("\n== exposure_sim_unrestricted.csv / exposure_ci_unrestricted.csv ==", flush=True)
simu = load("exposure_sim_unrestricted.csv")
if not simu.empty:
    check({"design", "split", "roi_w", "sharpe_daily", "pnl"} <= set(simu.columns),
          "unrestricted sim schema")
    check("mean_used" not in simu.columns, "unrestricted sim has no budget-race columns")
    check({"exposure_top80", "exposure_top50", "exposure_top20"} <= set(simu[simu["split"] == "test"]["design"]),
          "quantile-top designs unrestricted")
ciu = load("exposure_ci_unrestricted.csv")
if not ciu.empty:
    check({"design", "cost_bps", "roi_w", "sharpe_daily", "ci_lo", "ci_hi"} <= set(ciu.columns),
          "unrestricted ci schema")
    check({"exposure_top80", "exposure_top50", "exposure_top20"} <= set(ciu["design"]),
          "quantile-top designs in unrestricted CI")

print("\n== exposure_bins_{exposure,marg}.csv ==", flush=True)
for axis in ("exposure", "marg"):
    bn = load(f"exposure_bins_{axis}.csv")
    if not bn.empty:
        check({"split", "axis", "bin", "n", "mean_roi_res", "sharpe_copyable"} <= set(bn.columns),
              f"bin schema ({axis})")
        check(set(bn["split"].unique()) == {"train", "val", "test"}, f"all splits present ({axis})")
        check({"0-20", "80-100", "90-100"} <= set(bn["bin"]), f"edge + tail bins present ({axis})")

print("\n== exposure_bins_{exposure,marg}_price.csv ==", flush=True)
for axis in ("exposure", "marg"):
    bp = load(f"exposure_bins_{axis}_price.csv")
    if not bp.empty:
        check({"split", "axis", "price_q", "bin", "n", "mean_roi_res", "sharpe_copyable"} <= set(bp.columns),
              f"price-bin schema ({axis})")
        check(bp["price_q"].nunique() >= 3, f"multiple price quintiles ({axis})")

print("\n== exposure_wallet_bins.csv / exposure_wallet_summary.csv ==", flush=True)
wb = load("exposure_wallet_bins.csv")
if not wb.empty:
    check({"wallet", "sharpe_top", "sharpe_rest", "diff", "n_top", "n_rest"} <= set(wb.columns),
          "wallet-bin schema")
    check((wb["n_top"] >= 10).all() and (wb["n_rest"] >= 10).all(),
          "wallet trades >= 10 in both top and rest")
ws = load("exposure_wallet_summary.csv")
if not ws.empty:
    check({"split", "n_wallets", "mean_diff", "ci90_lo", "ci90_hi"} <= set(ws.columns),
          "wallet summary schema")
    check(ws["split"].isin(["train", "val", "test"]).all(), "wallet summary split labels")

print("\n" + ("PASS" if not errors else f"FAIL ({len(errors)} errors)"))
sys.exit(1 if errors else 0)