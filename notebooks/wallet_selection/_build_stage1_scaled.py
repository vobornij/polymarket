"""
Build the stage1_scaled notebook: per-wallet copy sizing (tier3@2-0).

Openers-style layout:
  title -> setup/imports -> load data -> copy universe -> depth cap ->
  train wallet stats -> weight schemes -> val grid search -> test pass ->
  robustness (cost sweep + bootstrap CI) -> per-wallet contributions -> save.

Writes notebooks/wallet_selection/stage1_scaled.ipynb.
"""
import ast
import json
import sys
import uuid
from pathlib import Path

NB_OUT = Path(__file__).resolve().parent / "stage1_scaled.ipynb"


def cid():
    return uuid.uuid4().hex[:12]


def md(source):
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": source,
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


CELL_SETUP = """\
# Setup: imports, paths, constants
# clear_cache()  # uncomment to force rebuild of data/bucket_depth.parquet
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

NB_DIR = Path.cwd() if "__file__" not in globals() else Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))
OUT_DIR = NB_DIR / "signal_lab"

import numpy as np
import pandas as pd

from lib import DEFAULT_TAGS
from signal_lab.depth_cap import build_lookup, clear_cache
from signal_lab.filters import COPY_DEFAULT
from signal_lab.signal_lib import spearman_rho
from signal_lab.sizing import (
    block_bootstrap_sharpe,
    capital_constrained_sim,
    sizing_sharpe,
)
from signal_lab.stage1 import candidate_splits_for, load_stage1_data
from signal_lab.wallet_scaling import (
    alpha_kelly,
    alpha_tier,
    attach_depth_cap,
    run_sim,
    sim_row,
    wallet_daily_pnl,
    wallet_stats,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

BUDGET = 10_000.0
COST_SEL = 10.0
ALPHA_MAX_GRID = (2.0, 4.0, 8.0)
TIER_GRID = [(nt, am, amin) for nt in (3, 4, 5) for am in ALPHA_MAX_GRID for amin in (0.0, 0.25)]
UNIFORM_K_GRID = (0.5, 1.0, 2.0, 4.0)
"""

CELL_LOAD_DATA = """\
df_full, df_train, df_val, df_test, wallet_metrics, hold_metrics = load_stage1_data()
print(f"df_full: {len(df_full):,}")
print(f"  train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")
"""

CELL_COPY_UNIVERSE = """\
wallets = set(COPY_DEFAULT(wallet_metrics, hold_metrics))
print(f"copy_default wallets: {len(wallets)}")
"""

CELL_DEPTH_CAP = """\
lookup = build_lookup()
print(f"depth lookup buckets: {len(lookup):,}")
lookup["bucket_avail_copy_qty"].describe(percentiles=[.5, .9, .99])
"""

CELL_BUILD_SPLITS = """\
splits = candidate_splits_for(df_full, wallets)
splits = attach_depth_cap(splits, wallets, lookup=lookup)
del df_full, df_train, df_val, df_test, lookup

for name in ("train", "val", "test"):
    fr = splits[name]
    capped = (fr["bucket_avail_copy_qty"] < fr["copyable_qty"]).mean()
    print(f"{name:5s}: {len(fr):,}  trades_capped_by_depth={capped:.3f}")
"""

CELL_TRAIN_STATS = """\
train_daily = wallet_daily_pnl(splits["train"])
st = wallet_stats(train_daily)
print(f"wallets with train daily series: {len(st)}")
st[["mu", "sigma", "n_days", "sharpe_proxy", "total_pnl"]].sort_values(
    "sharpe_proxy", ascending=False
).head(15)
"""

CELL_SCHEMES = """\
schemes = {}
for am in ALPHA_MAX_GRID:
    schemes[f"kelly@{am:g}"] = ("kelly", alpha_kelly(st, am), {"alpha_max": am})
for (nt, am, amin) in TIER_GRID:
    schemes[f"tier{nt}@{am:g}-{amin:g}"] = (
        "tier",
        alpha_tier(st, nt, am, amin),
        {"n_tiers": nt, "alpha_max": am, "alpha_min": amin},
    )
for k in UNIFORM_K_GRID:
    schemes[f"uniform@{k:g}"] = ("uniform", pd.Series(k, index=st.index), {"k": k})
schemes["copy_all"] = ("copy_all", pd.Series(1.0, index=st.index), {})

print(f"schemes: {len(schemes)}")
"""

CELL_VAL_GRID = """\
sim_rows = []
best_per_scheme = {}
for name, (scheme, alpha_map, params) in schemes.items():
    res = run_sim(splits["val"], alpha_map, COST_SEL)
    row = sim_row(scheme, name, "val", res)
    sim_rows.append(row)
    key = scheme if scheme != "kelly" else "kelly"
    if key not in best_per_scheme or row["sharpe_daily"] > best_per_scheme[key][2]:
        best_per_scheme[key] = (name, params, row["sharpe_daily"])

sim_df = pd.DataFrame(sim_rows)
sim_df[sim_df["split"] == "val"].sort_values("sharpe_daily", ascending=False).head(15)
"""

CELL_VAL_BEST = """\
print("Selected per scheme (by val Sharpe):")
for key, (name, params, val_sharpe) in best_per_scheme.items():
    print(f"  {key:10s} -> {name:>22s}  val_sharpe={val_sharpe:.3f}")

best_name = max(
    best_per_scheme.values(), key=lambda x: x[2]
)[0]
print(f"\\nBest val config overall: {best_name}")
"""

CELL_TEST = """\
for key, (name, params, _val_sharpe) in best_per_scheme.items():
    alpha_map = schemes[name][1]
    res = run_sim(splits["test"], alpha_map, COST_SEL)
    row = sim_row(schemes[name][0], name, "test", res)
    sim_rows.append(row)

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(OUT_DIR / "wallet_scaling_sim.csv", index=False)
sim_df[sim_df["split"] == "test"].sort_values("sharpe_daily", ascending=False)
"""

CELL_ROBUSTNESS = """\
ci_rows = []
for key, (name, params, _) in best_per_scheme.items():
    alpha_map = schemes[name][1]
    for cost in (0.0, 10.0, 30.0):
        res = run_sim(splits["test"], alpha_map, cost)
        point, lo, hi = block_bootstrap_sharpe(res["daily_pnl"], block_size=7, n_iter=1000, seed=42)
        ci_rows.append({
            "design": name, "cost_bps": cost,
            "pnl": round(res["net_pnl"], 2),
            "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
            "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
        })

res_all = capital_constrained_sim(splits["test"], "score1", BUDGET, 1.0, cost_bps=COST_SEL)
point, lo, hi = block_bootstrap_sharpe(res_all["daily_pnl"], block_size=7, n_iter=1000, seed=42)
ci_rows.append({
    "design": "copy_all", "cost_bps": COST_SEL,
    "pnl": round(res_all["net_pnl"], 2),
    "roi_w": round(res_all["net_pnl"] / res_all["notional"], 4) if res_all["notional"] > 0 else np.nan,
    "sharpe_daily": round(sizing_sharpe(res_all["daily_pnl"], 365.0), 3),
    "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
})

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(OUT_DIR / "wallet_scaling_ci.csv", index=False)
ci_df
"""

CELL_CONTRIB = """\
test_daily = wallet_daily_pnl(splits["test"])
test_st = test_daily.groupby("wallet")["copyable_pnl"].agg(
    test_pnl="sum", test_n_days="size"
)
test_sharpe = test_daily.groupby("wallet")["copyable_pnl"].apply(
    lambda s: (s.mean() / s.std() * np.sqrt(365.0)) if s.std() > 0 and len(s) >= 2 else np.nan
).rename("test_sharpe")

contrib = st.join(test_st, how="outer").join(test_sharpe, how="outer").fillna(0.0)
contrib = contrib[contrib["test_n_days"] > 0]
alpha_cont = schemes[best_per_scheme["kelly"][0]][1]
alpha_tier_cont = schemes[best_per_scheme["tier"][0]][1]
contrib["alpha_kelly"] = contrib.index.map(alpha_cont).fillna(1.0)
contrib["alpha_tier"] = contrib.index.map(alpha_tier_cont).fillna(1.0)
contrib = contrib.reset_index()
contrib["test_roi"] = contrib["test_pnl"] / contrib["total_pnl"].replace(0, np.nan)
contrib.to_csv(OUT_DIR / "wallet_scaling_contrib.csv", index=False)

a = contrib["alpha_kelly"].to_numpy()
ts = contrib["test_sharpe"].to_numpy()
valid = np.isfinite(ts)
rho = spearman_rho(pd.Series(a[valid]), pd.Series(ts[valid])) if valid.sum() > 2 else np.nan
print(f"Spearman(alpha_kelly, wallet test sharpe) = {rho:.4f}  (n={int(valid.sum())})")
contrib[["wallet", "alpha_kelly", "alpha_tier", "test_pnl", "test_sharpe", "test_roi"]].head(15)
"""

CELL_SAVE = """\
import json
from datetime import datetime, timezone

best_name = max(best_per_scheme.values(), key=lambda x: x[2])[0]
best_params = schemes[best_name][2]

wallet_cols = [
    "wallet", "mu", "sigma", "n_days", "total_pnl", "sharpe_proxy",
    "alpha_kelly", "alpha_tier", "test_pnl", "test_n_days", "test_sharpe", "test_roi",
]
wallet_records = contrib[[c for c in wallet_cols if c in contrib.columns]].to_dict(orient="records")


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


wallet_records = [{k: _convert(v) for k, v in w.items()} for w in wallet_records]

test_row = sim_df[(sim_df["split"] == "test") & (sim_df["config"] == best_name)].iloc[0]
copy_all_row = sim_df[(sim_df["split"] == "test") & (sim_df["config"] == "copy_all")].iloc[0]

metadata = {
    "type": "scaled_copy",
    "tags": sorted(DEFAULT_TAGS),
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
    "n_wallets_selected": int((contrib["alpha_tier"] > 0).sum()),
    "n_wallets_total": len(wallets),
    "split_sizes": {k: int(len(v)) for k, v in splits.items()},
}

payload = {
    "stage": 1,
    "best_params": {k: _convert(v) for k, v in best_params.items()},
    "best_val_sharpe": float(max(best_per_scheme.values(), key=lambda x: x[2])[2]),
    "test_performance": {
        "config": best_name,
        "trades": int(test_row["trades"]),
        "pnl": float(test_row["pnl"]),
        "roi_w": float(test_row["roi_w"]),
        "sharpe_daily": float(test_row["sharpe_daily"]),
        "copy_all": {
            "trades": int(copy_all_row["trades"]),
            "pnl": float(copy_all_row["pnl"]),
            "roi_w": float(copy_all_row["roi_w"]),
            "sharpe_daily": float(copy_all_row["sharpe_daily"]),
        },
    },
    "metadata": metadata,
    "wallets": wallet_records,
}

out_path = NB_DIR / "stage1_scaled_result.json"
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Saved stage 1 scaled result -> {out_path.resolve()}")
"""

cells = [
    md("""# Stage 1: Per-Wallet Copy Sizing (tier3@2-0)

Fit per-wallet copy weights ``alpha_w`` on **train** per-wallet daily pnl
(3 tiers by Sharpe proxy: top 2x, middle 1x, bottom dropped), pick
hyperparameters on **validation** by sim Sharpe, single **test** pass.
Copy qty is capped by the reconstructed share-depth ``bucket_avail_copy_qty``.

**Output:** `stage1_scaled_result.json` + `signal_lab/wallet_scaling_{sim,ci,contrib}.csv`
"""),
    code(CELL_SETUP),
    md("## Load data"),
    code(CELL_LOAD_DATA),
    md("## Copy universe\n\nCandidate wallets = `COPY_DEFAULT` (copy-default filter)."),
    code(CELL_COPY_UNIVERSE),
    md("## Reconstruct share-depth cap\n\nLoad the cached `bucket_avail_copy_qty` lookup per (tx_hash, wallet, side, token_id); rebuilt from the enriched shards on first run or after `clear_cache()`."),
    code(CELL_DEPTH_CAP),
    code(CELL_BUILD_SPLITS),
    md("## Train per-wallet stats\n\nPer-wallet daily pnl (copyable, alpha=1) with mean/std shrinkage -> Sharpe proxy."),
    code(CELL_TRAIN_STATS),
    md("## Weight schemes\n\nAll benchmarked vs copy-all: shrunk max-Sharpe (Kelly), tier, uniform-k."),
    code(CELL_SCHEMES),
    md("## Validation grid search\n\nObjective: annualized Sharpe of daily resolution-pnl, fixed $10k budget, 10bps."),
    code(CELL_VAL_GRID),
    code(CELL_VAL_BEST),
    md("## Test: single pass per chosen config\n\nOne honest test pass for each scheme's val-chosen config (10bps)."),
    code(CELL_TEST),
    md("## Robustness: cost sweep + bootstrap CI\n\nCost sweep (0/10/30bps) + 7-day block-bootstrap Sharpe CI on test."),
    code(CELL_ROBUSTNESS),
    md("## Per-wallet contributions\n\nTrain alphas vs forward (test) wallet stats."),
    code(CELL_CONTRIB),
    md("## Save stage 1 result"),
    code(CELL_SAVE),
]

kernelspec = {
    "display_name": "polymarket-analysis-BY1ldWyW-py3.13",
    "language": "python",
    "name": "python3",
}
language_info = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.13.7",
}

new_nb = {
    "cells": cells,
    "metadata": {"kernelspec": kernelspec, "language_info": language_info},
    "nbformat": 4,
    "nbformat_minor": 5,
}

errors = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        lines = [
            ln for ln in "".join(c["source"]).splitlines()
            if not ln.lstrip().startswith("%")
        ]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            errors.append((i, str(e)))

if errors:
    for i, e in errors:
        print(f"cell {i}: {e}")
    sys.exit(1)

with open(NB_OUT, "w") as f:
    json.dump(new_nb, f, indent=1, ensure_ascii=False)

print(f"Written: {len(new_nb['cells'])} cells total -> {NB_OUT}")
