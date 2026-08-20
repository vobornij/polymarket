"""Per-wallet copy-sizing (alpha_w) test: does scaling each wallet's copy orders
by a train-fitted weight (alpha_w, may be 0 to skip, may exceed 1 capped by
share depth) beat copy-all on Sharpe of the daily resolution-realized pnl?

Objective: annualized Sharpe of the capital-constrained sim's daily pnl, fixed
$10k budget.  Alphas are fit on train only; per-scheme hyperparameters
(alpha_max, tier grid, uniform k) are selected on val by sim Sharpe; test is a
single honest pass for each scheme's chosen config.

Schemes (all benchmarked against copy-all alpha=1):
- shrunk max-Sharpe (Kelly):  alpha_w = clip(mu_shrunk / var_shrunk, 0, alpha_max),
  mean-1 normalized over the candidate wallets (zeros allowed).
- tier: 3-5 tiers by train Sharpe proxy, alphas a_min..a_max per tier, mean-1.
- uniform-k: alpha_w = k for all (pure leverage, capped by depth).

Cap: qty = clip(alpha_w * copyable_qty_5m_100, 0, bucket_avail_copy_qty) — the
share-depth cap exported by stage0 as ``avail_copy_qty_5m_100``.

Outputs: wallet_scaling_sim.csv (val for all configs + test for chosen),
wallet_scaling_ci.csv (cost sweep + bootstrap CI for chosen designs),
wallet_scaling_contrib.csv (per-wallet alphas vs forward test stats).
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

from signal_lab.filters import COPY_DEFAULT
from signal_lab.sizing import (
    block_bootstrap_sharpe,
    capital_constrained_sim,
    sizing_sharpe,
)
from signal_lab.stage1 import candidate_splits_for, load_stage1_data

BUDGET = 10_000.0
COST_SEL = 10.0
LAMBDA_MU = 20.0
LAMBDA_VAR = 20.0
KEY_COLS = ["tx_hash", "wallet", "side", "token_id"]
ALPHA_MAX_GRID = (2.0, 4.0, 8.0)
TIER_GRID = [(nt, am, amin) for nt in (3, 4, 5) for am in ALPHA_MAX_GRID for amin in (0.0, 0.25)]
UNIFORM_K_GRID = (0.5, 1.0, 2.0, 4.0)


def attach_depth_cap(
    splits: dict[str, pd.DataFrame],
    avail_col: str = "avail_copy_qty_5m_100",
    qty_col: str = "copyable_qty_5m_100",
) -> dict[str, pd.DataFrame]:
    """Set the share-depth cap for each candidate split from stage0's
    per-bucket max copy quantity, aliased to ``bucket_avail_copy_qty``.
    """
    out = {}
    for name, fr in splits.items():
        if avail_col not in fr.columns:
            raise KeyError(
                f"{name} split is missing '{avail_col}'; stage0 must export "
                "the per-bucket max copy quantity."
            )
        fr = fr.copy()
        fr["bucket_avail_copy_qty"] = fr[avail_col].clip(lower=0.0).fillna(fr[qty_col])
        fr["score1"] = 1.0
        out[name] = fr
    return out


def wallet_daily_pnl(frame: pd.DataFrame, pnl_col: str = "copyable_pnl") -> pd.DataFrame:
    """Per-wallet daily pnl (copyable_pnl, alpha=1) at the sim's release date."""
    end_ns = pd.to_datetime(frame["end_date_iso"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
    close_ns = pd.to_datetime(frame["market_close"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
    rel = np.maximum(end_ns, close_ns).astype("datetime64[ns]")
    g = frame.assign(rel_date=pd.DatetimeIndex(rel, name="rel_date").tz_localize("UTC"))
    return g.groupby(["wallet", "rel_date"])[pnl_col].sum().reset_index()


def wallet_stats(train_daily: pd.DataFrame, pnl_col: str = "copyable_pnl") -> pd.DataFrame:
    """Per-wallet mean/std of daily pnl (mu, sigma), with count shrinkage inputs."""
    st = train_daily.groupby("wallet")[pnl_col].agg(
        mu="mean", sigma="std", n_days="size", total_pnl="sum"
    )
    st["sigma"] = st["sigma"].fillna(0.0)
    n_w = st["n_days"].sum()
    mu_cs = float((st["mu"] * st["n_days"]).sum() / n_w) if n_w > 0 else 0.0
    var_cs = float((st["sigma"] ** 2 * st["n_days"]).sum() / n_w) if n_w > 0 else 1.0
    lam = LAMBDA_MU
    st["mu_shrunk"] = (st["n_days"] * st["mu"] + lam * mu_cs) / (st["n_days"] + lam)
    st["var_shrunk"] = (st["n_days"] * st["sigma"] ** 2 + LAMBDA_VAR * var_cs) / (
        st["n_days"] + LAMBDA_VAR
    )
    st["sharpe_proxy"] = st["mu_shrunk"] / np.sqrt(st["var_shrunk"])
    st["mu_cs"], st["var_cs"] = mu_cs, var_cs
    return st


def price_scale_fill_sim(
    signals: pd.DataFrame,
    tape: pd.DataFrame,
    scales: tuple[float, ...] = (1.0, 0.98, 0.95, 0.90),
    window_minutes: float = 5.0,
) -> pd.DataFrame:
    """Simulate limit-price copy fills for candidate BUY signals.

    Each candidate copy-wallet BUY at price ``p`` is copied at limit price
    ``p * scale``.  A scaled order fills only if, within ``(dt, dt + window]``,
    any trade on the same ``(condition_id, token_id)`` prints at
    ``price <= limit`` with a **strictly greater** timestamp (``scale == 1.0``
    is the market-copy baseline: filled immediately at ``p``).

    Fill entry is the limit price, so the recomputed pnl is
    ``copyable_pnl + copyable_qty_5m_100 * (p - limit)`` (same formula and quantity as
    the original ``copyable_pnl``); unfilled scaled trades contribute 0.

    ``signals`` needs ``wallet, condition_id, token_id, dt, price,
    copyable_qty_5m_100, copyable_pnl``.  ``tape`` needs ``condition_id, token_id,
    dt, price`` and may contain both sides / all wallets.

    Returns one row per (signal, scale) with ``dt, price, copyable_qty_5m_100,
    copyable_pnl, limit_price, filled, pnl``.
    """
    sig = signals[signals["copyable_qty_5m_100"] > 0].copy()
    if sig.empty:
        return pd.DataFrame(columns=[
            "wallet", "condition_id", "token_id", "dt", "price", "copyable_qty_5m_100",
            "copyable_pnl", "scale", "limit_price", "filled", "pnl",
        ])
    sig["dt_ns"] = pd.to_datetime(sig["dt"], utc=True).astype(np.int64)
    tape_t = tape.copy()
    tape_t["dt_ns"] = pd.to_datetime(tape_t["dt"], utc=True).astype(np.int64)

    win_ns = int(window_minutes * 60 * 1_000_000_000)
    tape_t = tape_t.sort_values(["condition_id", "token_id", "dt_ns"])
    tape_groups = {
        key: (g["dt_ns"].to_numpy(), g["price"].to_numpy())
        for key, g in tape_t.groupby(["condition_id", "token_id"], sort=False)
    }

    rows = []
    for key, g in sig.groupby(["condition_id", "token_id"], sort=False):
        cid, tid = key
        dt_s = price_s = None
        if key in tape_groups:
            dt_s, price_s = tape_groups[key]
        tau = g["dt_ns"].to_numpy()
        price = g["price"].to_numpy()
        cpnl = g["copyable_pnl"].to_numpy()
        qty = g["copyable_qty_5m_100"].to_numpy()
        wallet = g["wallet"].to_numpy()
        if dt_s is not None:
            start = np.searchsorted(dt_s, tau, side="right")
            end = np.searchsorted(dt_s, tau + win_ns, side="right")
        for i in range(len(g)):
            win_min = np.inf
            if dt_s is not None and end[i] > start[i]:
                win_min = price_s[start[i]:end[i]].min()
            for scale in scales:
                limit = price[i] * scale
                if scale == 1.0:
                    filled = True
                    pnl = cpnl[i]
                else:
                    filled = win_min <= limit
                    pnl = cpnl[i] + qty[i] * (price[i] - limit) if filled else 0.0
                rows.append((
                    wallet[i], cid, tid, g["dt"].iloc[i], price[i], qty[i],
                    cpnl[i], scale, limit, filled, pnl,
                ))
    return pd.DataFrame(rows, columns=[
        "wallet", "condition_id", "token_id", "dt", "price", "copyable_qty_5m_100",
        "copyable_pnl", "scale", "limit_price", "filled", "pnl",
    ])


def normalize_mean1(alpha: pd.Series) -> pd.Series:
    m = alpha.mean()
    return alpha / m if m > 0 else alpha


def alpha_kelly(st: pd.DataFrame, alpha_max: float) -> pd.Series:
    alpha = np.clip(st["mu_shrunk"].clip(lower=0.0) / st["var_shrunk"], 0.0, alpha_max)
    return normalize_mean1(alpha)


def alpha_tier(st: pd.DataFrame, n_tiers: int, alpha_max: float, alpha_min: float) -> pd.Series:
    ranked = st["sharpe_proxy"].rank(method="first", ascending=False).astype(int) - 1
    size = max(int(np.ceil(len(st) / n_tiers)), 1)
    tier = np.minimum(ranked // size, n_tiers - 1).astype(int)
    if n_tiers == 1:
        vals = np.array([alpha_min])
    else:
        vals = alpha_min + (alpha_max - alpha_min) * (n_tiers - 1 - np.arange(n_tiers)) / (n_tiers - 1)
    alpha = pd.Series(vals[tier.to_numpy()], index=st.index)
    return normalize_mean1(alpha)


def run_sim(frame: pd.DataFrame, alpha_map: pd.Series,
            pnl_col: str = "copyable_pnl",
            qty_col: str = "copyable_qty_5m_100",
            group_col: str | None = None,
            group_budget: float | None = None) -> dict:
    t = frame.copy(deep=True)
    t["alpha_w"] = t["wallet"].map(alpha_map).fillna(1.0)
    # When group_budget is set, disable the global cap so exposure is per-group only
    effective_budget = float("inf") if group_budget is not None else BUDGET
    res = capital_constrained_sim(t, "score1", effective_budget, 1.0,
                                  alpha_col="alpha_w", cap_col="bucket_avail_copy_qty",
                                  pnl_col=pnl_col, qty_col=qty_col,
                                  group_col=group_col, group_budget=group_budget)
    return res


def sim_row(scheme: str, config, split: str, res: dict) -> dict:
    return {
        "scheme": scheme, "config": str(config), "split": split,
        "trades": res["trades"],
        "pnl": round(res["net_pnl"], 2),
        "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
        "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
        "mean_used": round(res["mean_used"], 2),
        "peak_used": round(res["peak_used"], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--cached-splits-dir",
        type=str,
        default=None,
        help="Cache/load prepared candidate splits (with depth cap) as parquet.",
    )
    args = parser.parse_args()
    cache_dir = Path(args.cached_splits_dir) if args.cached_splits_dir else None
    cache_files = [cache_dir / f"{s}.parquet" for s in ("train", "val", "test")] if cache_dir else []
    if cache_dir and all(f.exists() for f in cache_files):
        print("Loading cached candidate splits...", flush=True)
        splits = {s: pd.read_parquet(f) for s, f in zip(("train", "val", "test"), cache_files)}
        for fr in splits.values():
            fr["score1"] = 1.0
    else:
        print("Loading stage-1 data...", flush=True)
        df_full, _dt, _dv, _dtest, wallet_metrics, hold_metrics = load_stage1_data(
            max_shards=args.max_shards
        )
        wallets = set(COPY_DEFAULT(wallet_metrics, hold_metrics))
        print(f"copy_default wallets: {len(wallets)}", flush=True)

        print("Building candidate splits...", flush=True)
        splits = candidate_splits_for(df_full, wallets)
        splits = attach_depth_cap(splits)
        del df_full, _dt, _dv, _dtest
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for s, f in zip(("train", "val", "test"), cache_files):
                splits[s].to_parquet(f, index=False)
            print(f"Cached splits -> {cache_dir}", flush=True)

    print("Computing train per-wallet daily pnl...", flush=True)
    train_daily = wallet_daily_pnl(splits["train"])
    st = wallet_stats(train_daily)
    print(f"wallets with train daily series: {len(st)}", flush=True)

    schemes = {}
    for am in ALPHA_MAX_GRID:
        schemes[f"kelly@{am:g}"] = ("kelly", alpha_kelly(st, am), {"alpha_max": am})
    for (nt, am, amin) in TIER_GRID:
        schemes[f"tier{nt}@{am:g}-{amin:g}"] = ("tier", alpha_tier(st, nt, am, amin),
                                                {"n_tiers": nt, "alpha_max": am, "alpha_min": amin})
    for k in UNIFORM_K_GRID:
        schemes[f"uniform@{k:g}"] = ("uniform", pd.Series(k, index=st.index), {"k": k})
    schemes["copy_all"] = ("copy_all", pd.Series(1.0, index=st.index), {})

    print("\n" + "=" * 78, flush=True)
    print("Val selection (config grid, sim Sharpe, 10bps)", flush=True)
    print("=" * 78, flush=True)
    sim_rows = []
    best_per_scheme: dict[str, tuple[str, dict, float]] = {}
    for name, (scheme, alpha_map, params) in schemes.items():
        res = run_sim(splits["val"], alpha_map, COST_SEL)
        row = sim_row(scheme, name, "val", res)
        sim_rows.append(row)
        print(f"{name:>22s}  sharpe={row['sharpe_daily']:.3f}  pnl={row['pnl']:>10,.0f}  "
              f"roi_w={row['roi_w']:.4f}  trades={row['trades']:,}  mean_used={row['mean_used']:,.0f}",
              flush=True)
        key = scheme if scheme != "kelly" else "kelly"
        if key not in best_per_scheme or row["sharpe_daily"] > best_per_scheme[key][2]:
            best_per_scheme[key] = (name, params, row["sharpe_daily"])
    print(f"\nSelected per scheme (by val Sharpe): "
          f"{ {k: v[0] for k, v in best_per_scheme.items()} }", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("Test: single pass for each scheme's chosen config (10bps) + copy-all", flush=True)
    print("=" * 78, flush=True)
    for key, (name, params, _val_sharpe) in best_per_scheme.items():
        alpha_map = schemes[name][1]
        res = run_sim(splits["test"], alpha_map, COST_SEL)
        row = sim_row(schemes[name][0], name, "test", res)
        sim_rows.append(row)
        print(f"{name:>22s}  sharpe={row['sharpe_daily']:.3f}  pnl={row['pnl']:>10,.0f}  "
              f"roi_w={row['roi_w']:.4f}  trades={row['trades']:,}  mean_used={row['mean_used']:,.0f}",
              flush=True)
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv("wallet_scaling_sim.csv", index=False)

    print("\n" + "=" * 78, flush=True)
    print("Robustness: cost sweep + 7-day block-bootstrap Sharpe CI (test)", flush=True)
    print("=" * 78, flush=True)
    ci_rows = []
    for key, (name, params, _) in best_per_scheme.items():
        alpha_map = schemes[name][1]
        for cost in (0.0, 10.0, 30.0):
            res = run_sim(splits["test"], alpha_map, cost)
            point, lo, hi = block_bootstrap_sharpe(res["daily_pnl"], block_size=7, n_iter=1000, seed=42)
            ci_rows.append({"design": name, "cost_bps": cost,
                            "pnl": round(res["net_pnl"], 2),
                            "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
                            "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    res_all = capital_constrained_sim(splits["test"], "score1", BUDGET, 1.0, cost_bps=COST_SEL)
    point, lo, hi = block_bootstrap_sharpe(res_all["daily_pnl"], block_size=7, n_iter=1000, seed=42)
    ci_rows.append({"design": "copy_all", "cost_bps": COST_SEL,
                    "pnl": round(res_all["net_pnl"], 2),
                    "roi_w": round(res_all["net_pnl"] / res_all["notional"], 4) if res_all["notional"] > 0 else np.nan,
                    "sharpe_daily": round(sizing_sharpe(res_all["daily_pnl"], 365.0), 3),
                    "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    ci_df = pd.DataFrame(ci_rows)
    print(ci_df.to_string(index=False), flush=True)
    ci_df.to_csv("wallet_scaling_ci.csv", index=False)

    print("\n" + "=" * 78, flush=True)
    print("Per-wallet contributions (unconstrained, forward)", flush=True)
    print("=" * 78, flush=True)
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
    contrib.to_csv("wallet_scaling_contrib.csv", index=False)

    from signal_lab.signal_lib import spearman_rho
    a = contrib["alpha_kelly"].to_numpy()
    ts = contrib["test_sharpe"].to_numpy()
    valid = np.isfinite(ts)
    rho = spearman_rho(pd.Series(a[valid]), pd.Series(ts[valid])) if valid.sum() > 2 else np.nan
    print(f"\nSpearman(alpha_kelly, wallet test sharpe) = {rho:.4f}  (n={int(valid.sum())})", flush=True)
    med = contrib["alpha_kelly"].median()
    hi_g = contrib[contrib["alpha_kelly"] > med]
    lo_g = contrib[contrib["alpha_kelly"] <= med]
    for label, g in (("top-half alpha", hi_g), ("bottom-half alpha", lo_g)):
        print(f"{label}: wallets={len(g):,}  test_pnl={g['test_pnl'].sum():>10,.0f}  "
              f"mean test_sharpe={g['test_sharpe'].mean():.4f}  "
              f"median alpha={g['alpha_kelly'].median():.3f}", flush=True)
    top5 = contrib.sort_values("alpha_kelly", ascending=False).head(5)
    tot = contrib["test_pnl"].sum()
    print(f"top-5 alpha wallets test_pnl share: {top5['test_pnl'].sum() / tot:.3f} "
          f"(total test_pnl {tot:,.0f})", flush=True)
    print(f"\nSaved wallet_scaling_{{sim,ci,contrib}}.csv", flush=True)


if __name__ == "__main__":
    main()
