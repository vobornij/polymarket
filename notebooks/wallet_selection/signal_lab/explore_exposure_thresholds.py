"""Explore the copy-wallet exposure idea end-to-end.

``--ic``    Phase A: exposure signals -> IC vs ``copyable_pnl`` and ``roi_res``
            (standard signal panel + price-confound check).  Writes
            ``exposure_ic_report_{pnl,roi}.csv``.

``--capture`` Phase B: precompute per-wallet exposure thresholds that capture a
            target fraction (1.0/0.8/0.5/0.2) of each wallet's train copyable
            PnL, apply them to val/test, and compare capital-constrained
            sizing (Sharpe/PnL) against copy-all and the wallet-scaling
            schemes (kelly/tier/uniform) computed on the same universe.
            Writes ``exposure_thresholds.csv``, ``exposure_capture_curve.csv``,
            ``exposure_capture_summary.csv``, ``exposure_sim.csv``,
            ``exposure_ci.csv``.

Data is loaded once; both phases share the same copy universe.
Run with ``--max-shards N`` to iterate cheaply on a subset first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from lib import DEFAULT_TAGS
from signal_lab.filters import COPY_DEFAULT
from signal_lab.signal_lib import spearman_rho
from signal_lab.sizing import block_bootstrap_sharpe, capital_constrained_sim, sizing_sharpe
from signal_lab.stage1 import (
    candidate_splits_for,
    evaluate_signal_panel,
    load_stage1_data,
    run_strategy,
)
from signal_lab.strategies import ExposureSignals
from signal_lab.strategies.exposure_threshold import (
    apply_wallet_rank01,
    fit_wallet_exposure_grids,
    wallet_capture_thresholds,
    wallet_exposure_quantile_thresholds,
)
from signal_lab.wallet_scaling import (
    alpha_kelly,
    alpha_tier,
    attach_depth_cap,
    wallet_daily_pnl,
    wallet_stats,
)

BUDGET = 10_000.0
COST_SEL = 10.0
FRACS = (1.0, 0.8, 0.5, 0.2)
TOP_FRACS = (0.8, 0.5, 0.2)
PRICE_BINS = 5
RANK_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RANK_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
ALPHA_MAX_GRID = (2.0, 4.0, 8.0)
TIER_GRID = [(nt, am, amin) for nt in (3, 4, 5) for am in ALPHA_MAX_GRID for amin in (0.0, 0.25)]
UNIFORM_K_GRID = (0.5, 1.0, 2.0, 4.0)


def run_cap(
    frame: pd.DataFrame,
    alpha_map: pd.Series,
    cost_bps: float,
    budget: float = BUDGET,
) -> dict:
    """Capital-constrained sim with per-wallet alpha and the share-depth cap.

    ``budget=float("inf")`` runs the unrestricted sim: every offered trade is
    taken at ``alpha_w * copyable_qty`` (depth-capped), no skip-race.
    """
    t = frame.copy(deep=True)
    t["alpha_w"] = t["wallet"].map(alpha_map).fillna(1.0)
    return capital_constrained_sim(
        t, "score1", budget, 1.0,
        cost_bps=cost_bps,
        alpha_col="alpha_w",
        cap_col="bucket_avail_copy_qty",
    )


def sim_row(name: str, split: str, res: dict, config: str = "") -> dict:
    return {
        "design": name, "config": config, "split": split,
        "trades": res["trades"],
        "pnl": round(res["net_pnl"], 2),
        "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
        "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
        "mean_used": round(res["mean_used"], 2),
        "peak_used": round(res["peak_used"], 2),
    }


# ---------------------------------------------------------------------------
# Phase A: IC panel
# ---------------------------------------------------------------------------


def within_price_bin_ic(
    pooled: pd.DataFrame,
    cols: list[str],
    targets: dict[str, str],
    price_col: str = "price",
    n_bins: int = PRICE_BINS,
) -> pd.DataFrame:
    """Pooled IC of each signal vs ``target`` *within* price-quantile bins.

    Answers: does the signal still separate copyable PnL / roi_res once the
    price axis is held (roughly) constant inside each wallet-relative bin?  The
    weighted mean uses bin population weights; NaN (bin IC degenerate) counts as
    0 contribution without a weight.
    """
    cols = [c for c in cols if c in pooled.columns]
    cols = list(dict.fromkeys(cols))  # dedupe, keep order
    starts = pooled[price_col].rank()
    cuts = pd.qcut(starts, n_bins, labels=False, duplicates="drop")
    weights = cuts.value_counts(sort=False)
    rows = []
    for col in cols:
        row: dict = {"signal": col}
        for label, target in targets.items():
            weighted = 0.0
            num = 0
            for bin_no in weights.index:
                idx = np.flatnonzero(cuts.to_numpy() == bin_no)
                seg = pooled.iloc[idx]
                ic = spearman_rho(seg[col].fillna(0.0), seg[target])
                if np.isnan(ic):
                    continue
                w = int(weights.loc[bin_no])
                weighted += w * ic
                num += w
            row[f"wb_ic_{label}"] = weighted / num if num else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_ic(
    df_full: pd.DataFrame,
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
    tag_suffix: str = "",
) -> None:
    print("==" * 40, flush=True)
    print("Phase A: exposure signal IC panel", flush=True)
    print("==" * 40, flush=True)
    strategy = ExposureSignals(fracs=FRACS)
    splits, cols = run_strategy(df_full, wallet_metrics, hold_metrics, strategy)

    print(f"\ncandidate universes: "
          f"train={len(splits['train']):,}  val={len(splits['val']):,}  test={len(splits['test']):,}",
          flush=True)

    _tgt = ["copyable_pnl", "roi_res"]
    pooled = pd.concat(
        [splits["train"][cols + _tgt + ["price"]], splits["val"][cols + _tgt + ["price"]]],
        ignore_index=True,
    )
    wb = within_price_bin_ic(pooled, cols, {"pnl": "copyable_pnl", "roi": "roi_res"})
    wb_map = wb.set_index("signal")

    for roi_col, label in (("copyable_pnl", "pnl"), ("roi_res", "roi")):
        report, selected = evaluate_signal_panel(splits, cols, roi_col=roi_col)
        report["spearman_price"] = [
            spearman_rho(pooled[c].fillna(0.0), pooled["price"]) for c in report["signal"]
        ]
        report["wb_ic_pnl"] = report["signal"].map(wb_map["wb_ic_pnl"])
        report["wb_ic_roi"] = report["signal"].map(wb_map["wb_ic_roi"])
        report.to_csv(f"exposure_ic_report_{label}{tag_suffix}.csv", index=False)
        print(f"\nIC vs {roi_col} (saved exposure_ic_report_{label}{tag_suffix}.csv)", flush=True)
        print("selected:", selected, flush=True)
        print(report.to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
# Phase B: capture thresholds + sizing
# ---------------------------------------------------------------------------


def capture_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-wallet cumulative-PnL capture curve by exposure quantile."""
    rows = []
    for wallet, g in frame.groupby("wallet", sort=False):
        total = float(g["copyable_pnl"].sum())
        if total <= 0:
            continue
        g = g.sort_values("exposure", ascending=False)
        cum = g["copyable_pnl"].cumsum().to_numpy(dtype=float)
        n = len(g)
        step = max(int(n // 100), 1)
        seen = set()
        for i in range(0, n, step):
            seen.add(i)
            rows.append({
                "wallet": wallet, "n": n,
                "exposure_pctile": i / max(n - 1, 1),
                "exposure": float(g["exposure"].iloc[i]),
                "cum_cpnl_frac": float(cum[i] / total),
            })
        if n - 1 not in seen:
            rows.append({
                "wallet": wallet, "n": n, "exposure_pctile": 1.0,
                "exposure": float(g["exposure"].iloc[-1]),
                "cum_cpnl_frac": float(cum[-1] / total),
            })
    return pd.DataFrame(rows)


def capture_summary_rows(frame: pd.DataFrame, th: pd.DataFrame) -> list[dict]:
    rows = []
    wallets = th["wallet"].unique()
    base = frame[frame["wallet"].isin(wallets)].copy()
    if base.empty:
        return rows
    pnl_all = float(base["copyable_pnl"].sum())
    cnot_all = float(base["copyable_notional"].sum())
    for frac in FRACS:
        t = th[th["frac"] == frac][["wallet", "threshold"]]
        tmap = pd.Series(t["threshold"].to_numpy(), index=t["wallet"])
        T = base["wallet"].map(tmap)
        keep = (base["exposure"] >= T).fillna(False)
        kept = base[keep]
        pnl_kept = float(kept["copyable_pnl"].sum())
        cnot_kept = float(kept["copyable_notional"].sum())
        rows.append({
            "frac": frac,
            "trades_total": len(base),
            "trades_kept": len(kept),
            "qty_kept": int((kept["copyable_qty_5m_100"] > 0).sum()),
            "retention": len(kept) / len(base) if len(base) else np.nan,
            "cpnl_all": pnl_all,
            "cpnl_kept": pnl_kept,
            "realized_capture": pnl_kept / pnl_all if pnl_all != 0 else np.nan,
            "roi_all": pnl_all / cnot_all if cnot_all != 0 else np.nan,
            "roi_kept": pnl_kept / cnot_kept if cnot_kept != 0 else np.nan,
            "roi_res_mean_all": float(base["roi_res"].mean()),
            "roi_res_mean_kept": float(kept["roi_res"].mean()),
            "exp_med_kept": float(kept["exposure"].median()),
        })
    return rows


def filtered_split(
    splits: dict[str, pd.DataFrame],
    th: pd.DataFrame,
    name: str,
    frac: float,
) -> pd.DataFrame:
    t = th[th["frac"] == frac][["wallet", "threshold"]]
    tmap = pd.Series(t["threshold"].to_numpy(), index=t["wallet"])
    T = splits[name]["wallet"].map(tmap)
    return splits[name][(splits[name]["exposure"] >= T).fillna(False)].copy()


def filtered_split_q(
    splits: dict[str, pd.DataFrame],
    thq: pd.DataFrame,
    name: str,
    top_frac: float,
) -> pd.DataFrame:
    t = thq[thq["top_frac"] == top_frac][["wallet", "threshold"]]
    tmap = pd.Series(t["threshold"].to_numpy(), index=t["wallet"])
    T = splits[name]["wallet"].map(tmap)
    return splits[name][(splits[name]["exposure"] >= T).fillna(False)].copy()


def scaling_sim(
    splits: dict[str, pd.DataFrame],
    th: pd.DataFrame,
    thq: pd.DataFrame,
    tag_suffix: str = "",
    unrestricted: bool = False,
) -> None:
    budget = float("inf") if unrestricted else BUDGET
    mode = "UNRESTRICTED (budget=inf, proportional-to-wallet, no skip)" if unrestricted \
        else "CAPITAL-CONSTRAINED (10k, skip-race)"
    out_tag = "_unrestricted" if unrestricted else ""
    print("\n" + "=" * 78, flush=True)
    print(f"Sizing: exposure filters vs wallet scaling (10bps) — {mode}", flush=True)
    print("=" * 78, flush=True)
    train_daily = wallet_daily_pnl(splits["train"])
    st = wallet_stats(train_daily)
    print(f"wallets with train daily series: {len(st)}", flush=True)
    thq = wallet_exposure_quantile_thresholds(splits["train"], TOP_FRACS)

    schemes: dict[str, tuple[str, pd.Series, dict]] = {
        "copy_all": ("scaling", pd.Series(1.0, index=st.index), {})
    }
    for am in ALPHA_MAX_GRID:
        schemes[f"kelly@{am:g}"] = ("kelly", alpha_kelly(st, am), {"alpha_max": am})
    for (nt, am, amin) in TIER_GRID:
        schemes[f"tier{nt}@{am:g}-{amin:g}"] = (
            "tier", alpha_tier(st, nt, am, amin),
            {"n_tiers": nt, "alpha_max": am, "alpha_min": amin},
        )
    for k in UNIFORM_K_GRID:
        schemes[f"uniform@{k:g}"] = ("uniform", pd.Series(k, index=st.index), {"k": k})

    best_per_family: dict[str, tuple[str, float]] = {}
    sim_rows: list[dict] = []
    for name, (family, alpha_map, _params) in schemes.items():
        res = run_cap(splits["val"], alpha_map, COST_SEL, budget)
        row = sim_row(name, "val", res, config=name)
        sim_rows.append(row)
        if family not in best_per_family or row["sharpe_daily"] > best_per_family[family][1]:
            best_per_family[family] = (name, row["sharpe_daily"])
    for frac in FRACS:
        res = run_cap(filtered_split(splits, th, "val", frac), pd.Series(1.0, index=st.index), COST_SEL, budget)
        sim_rows.append(sim_row(f"exposure_{frac:g}", "val", res, config=f"frac={frac:g}"))
    for t in TOP_FRACS:
        res = run_cap(filtered_split_q(splits, thq, "val", t), pd.Series(1.0, index=st.index), COST_SEL, budget)
        sim_rows.append(sim_row(f"exposure_top{t*100:.0f}", "val", res, config=f"top={t:g}"))

    print("Selected per family (by val Sharpe):",
          {k: v[0] for k, v in best_per_family.items()}, flush=True)
    test_designs = [v[0] for v in best_per_family.values()]
    for name in test_designs:
        res = run_cap(splits["test"], schemes[name][1], COST_SEL, budget)
        sim_rows.append(sim_row(name, "test", res, config=name))
    for frac in FRACS:
        res = run_cap(filtered_split(splits, th, "test", frac), pd.Series(1.0, index=st.index), COST_SEL, budget)
        sim_rows.append(sim_row(f"exposure_{frac:g}", "test", res, config=f"frac={frac:g}"))
    for t in TOP_FRACS:
        res = run_cap(filtered_split_q(splits, thq, "test", t), pd.Series(1.0, index=st.index), COST_SEL, budget)
        sim_rows.append(sim_row(f"exposure_top{t*100:.0f}", "test", res, config=f"top={t:g}"))

    sim_df = pd.DataFrame(sim_rows)
    if unrestricted:
        sim_df = sim_df.drop(columns=["mean_used", "peak_used"], errors="ignore")
    sim_df.to_csv(f"exposure_sim{out_tag}{tag_suffix}.csv", index=False)
    print(f"\nVal selection + test single-pass (saved exposure_sim{out_tag}{tag_suffix}.csv):", flush=True)
    print(sim_df.to_string(index=False), flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"Robustness: cost sweep + 7-day block-bootstrap Sharpe CI (test) — {mode}", flush=True)
    print("=" * 78, flush=True)
    ci_rows: list[dict] = []
    ci_designs = (
        test_designs
        + [f"exposure_{f:g}" for f in FRACS]
        + [f"exposure_top{t*100:.0f}" for t in TOP_FRACS]
    )
    for design in ci_designs:
        if design.startswith("exposure_top"):
            frame = filtered_split_q(splits, thq, "test", float(design.split("top")[1]) / 100.0)
            alpha = pd.Series(1.0, index=st.index)
        elif design.startswith("exposure_"):
            frame = filtered_split(splits, th, "test", float(design.split("_")[1]))
            alpha = pd.Series(1.0, index=st.index)
        else:
            frame = splits["test"]
            alpha = schemes[design][1]
        for cost in (0.0, 10.0, 30.0):
            res = run_cap(frame, alpha, cost, budget)
            point, lo, hi = block_bootstrap_sharpe(res["daily_pnl"], block_size=7, n_iter=1000, seed=42)
            ci_rows.append({
                "design": design, "cost_bps": cost,
                "pnl": round(res["net_pnl"], 2),
                "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
                "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            })
    ci_df = pd.DataFrame(ci_rows)
    print(ci_df.to_string(index=False), flush=True)
    ci_df.to_csv(f"exposure_ci{out_tag}{tag_suffix}.csv", index=False)


def add_exposure_cols(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Attach level ('exposure') and marginal ('marg') axes on candidate splits."""
    for fr in splits.values():
        fr["exposure"] = fr["position"] * fr["price"]
        qty = fr["quantity"] if "quantity" in fr.columns else fr["copyable_qty_5m_100"]
        fr["marg"] = qty * fr["price"]
    return splits


def load_cached_splits(cached_splits_dir: str) -> dict[str, pd.DataFrame]:
    cache_dir = Path(cached_splits_dir)
    splits = {s: pd.read_parquet(cache_dir / f"{s}.parquet") for s in ("train", "val", "test")}
    return add_exposure_cols(splits)


def get_candidate_splits(
    df_full: pd.DataFrame,
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
    cached_splits_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Copy-default candidate BUY splits, depth-capped, cached as parquet."""
    wallets = set(COPY_DEFAULT(wallet_metrics, hold_metrics))
    print(f"copy_default wallets: {len(wallets)}", flush=True)
    cache_dir = Path(cached_splits_dir) if cached_splits_dir else None
    cache_files = [cache_dir / f"{s}.parquet" for s in ("train", "val", "test")] if cache_dir else []
    if cache_dir and all(f.exists() for f in cache_files):
        print("Loading cached candidate splits...", flush=True)
        splits = {s: pd.read_parquet(f) for s, f in zip(("train", "val", "test"), cache_files)}
    else:
        splits = candidate_splits_for(df_full, wallets)
        splits = attach_depth_cap(splits)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for s, f in zip(("train", "val", "test"), cache_files):
                splits[s].to_parquet(f, index=False)
            print(f"Cached depth-capped splits -> {cache_dir}", flush=True)
    return add_exposure_cols(splits)


def run_capture(
    splits: dict[str, pd.DataFrame],
    tag_suffix: str = "",
    unrestricted: bool = False,
) -> None:
    print("==" * 40, flush=True)
    print("Phase B: exposure capture thresholds + sizing vs wallet scaling", flush=True)
    print("==" * 40, flush=True)

    th = wallet_capture_thresholds(splits["train"], FRACS)
    thq = wallet_exposure_quantile_thresholds(splits["train"], TOP_FRACS)
    th.to_csv(f"exposure_thresholds{tag_suffix}.csv", index=False)
    print(f"\nSaved exposure_thresholds.csv: {len(th):,} rows "
          f"({th['wallet'].nunique()} wallets x {th['frac'].nunique()} fracs)", flush=True)
    for f, g in th.groupby("frac"):
        print(f"  frac={f:g}: wallets={len(g)}  thresh med={g['threshold'].median():9.2f} "
              f"p90={g['threshold'].quantile(.9):10.2f}  max={g['threshold'].max():,.0f}",
              flush=True)

    curve = capture_curve(splits["train"])
    curve.to_csv(f"exposure_capture_curve{tag_suffix}.csv", index=False)
    print(f"\nSaved exposure_capture_curve.csv: {len(curve):,} rows", flush=True)

    sum_rows: list[dict] = []
    for name, fr in (
        ("train", splits["train"]),
        ("val", splits["val"]),
        ("test", splits["test"]),
    ):
        print(f"\n=== {name} ({len(fr):,} candidate trades, cpnl={fr['copyable_pnl'].sum():,.0f}) ===",
              flush=True)
        for row in capture_summary_rows(fr, th):
            row["split"] = name
            sum_rows.append(row)
            print(
                f"  f={row['frac']:g}: kept={row['trades_kept']:>6}/{row['trades_total']:<6}"
                f"({row['retention']*100:4.1f}%)  capture={row['realized_capture']:.2f}  "
                f"roi={row['roi_kept']:.4f} (all {row['roi_all']:.4f})  "
                f"roi_res={row['roi_res_mean_kept']:+.4f} (all {row['roi_res_mean_all']:+.4f})  "
                f"exp_med={row['exp_med_kept']:,.0f}",
                flush=True,
            )
    pd.DataFrame(sum_rows).to_csv(f"exposure_capture_summary{tag_suffix}.csv", index=False)

    scaling_sim(splits, th, thq, tag_suffix, unrestricted=unrestricted)
    if not unrestricted:
        scaling_sim(splits, th, thq, tag_suffix, unrestricted=True)


# ---------------------------------------------------------------------------
# Phase C: conditional exposure-bin diagnostic (within-wallet rank)
# ---------------------------------------------------------------------------


def attach_wallet_rank(
    splits: dict[str, pd.DataFrame],
    axis: str = "exposure",
) -> pd.DataFrame:
    """Within-wallet rank01 of ``axis`` (train-fit), applied to all splits.

    Extrapolates outside train exposure bounds without clipping; new wallets
    on val/test (not in train grids) keep rank NaN.
    """
    train = splits["train"]
    grids = fit_wallet_exposure_grids(train, axis)
    col = f"rank_{axis}"
    for name, fr in splits.items():
        out = np.empty(len(fr), dtype=float)
        out.fill(np.nan)
        for w, vals in fr.groupby("wallet", sort=False)[axis]:
            if w in grids:
                r = apply_wallet_rank01(vals.to_numpy(dtype=float), w, grids)
                out[fr["wallet"].to_numpy() == w] = r
        fr[col] = out
    return splits


def _bin_stats(frame: pd.DataFrame, rank_col: str, label: str) -> dict:
    g = frame[frame[rank_col].notna() & frame["roi_res"].notna() & frame["copyable_roi"].notna()]
    g = g[(g[rank_col] >= RANK_BINS[0]) & (g[rank_col] <= RANK_BINS[-1])]
    roi = g["copyable_roi"].to_numpy(dtype=float)
    sd = roi.std(ddof=1)
    return {
        "n": int(len(g)),
        "median_exposure": float(g["exposure"].median()) if len(g) else np.nan,
        "mean_roi_res": float(g["roi_res"].mean()) if len(g) else np.nan,
        "med_roi_res": float(g["roi_res"].median()) if len(g) else np.nan,
        "mean_copyable_roi": float(roi.mean()) if len(g) else np.nan,
        "sd_copyable_roi": float(sd) if len(g) else np.nan,
        "sharpe_copyable": (float(roi.mean() / sd) if len(roi) >= 5 and sd > 0 else np.nan),
        "pnl": float(g["copyable_pnl"].sum()) if len(g) else np.nan,
    }


def bin_table_rows(splits: dict[str, pd.DataFrame], axis: str = "exposure") -> list[dict]:
    rank_col = f"rank_{axis}"
    rows = []
    for split, fr in splits.items():
        for label, (lo, hi) in zip(RANK_LABELS, zip(RANK_BINS[:-1], RANK_BINS[1:])):
            seg = fr[(fr[rank_col] >= lo) & (fr[rank_col] < hi)]
            row = _bin_stats(seg, rank_col, label)
            row.update({"split": split, "axis": axis, "bin": label})
            rows.append(row)
        seg = fr[fr[rank_col] >= 0.9]
        row = _bin_stats(seg, rank_col, "90-100")
        row.update({"split": split, "axis": axis, "bin": "90-100"})
        rows.append(row)
    return rows


def bin_price_rows(splits: dict[str, pd.DataFrame], axis: str = "exposure", n_bins: int = PRICE_BINS) -> list[dict]:
    rank_col = f"rank_{axis}"
    rows = []
    for split, fr in splits.items():
        fr = fr[fr["roi_res"].notna() & fr["copyable_roi"].notna()]
        q = pd.qcut(fr["price"].rank(method="first"), n_bins, labels=False, duplicates="drop")
        for pq, seg in fr.groupby(q, sort=False):
            plo, phi = float(seg["price"].min()), float(seg["price"].max())
            for label, (lo, hi) in zip(RANK_LABELS, zip(RANK_BINS[:-1], RANK_BINS[1:])):
                b = seg[(seg[rank_col] >= lo) & (seg[rank_col] < hi)]
                row = _bin_stats(b, rank_col, label)
                row.update({"split": split, "axis": axis, "price_q": int(pq),
                            "price_min": plo, "price_max": phi, "bin": label,
                            "mean_price": float(seg["price"].mean())})
                rows.append(row)
    return rows


def wallet_sharpe_rows(splits: dict[str, pd.DataFrame], axis: str = "exposure", seed: int = 42) -> dict:
    """Wallet-level copyable Sharpe top-20% vs rest (higher relative exposure)."""
    rank_col = f"rank_{axis}"
    out: dict[str, list[dict]] = {s: [] for s in splits}
    rng = np.random.default_rng(seed)
    for split, fr in splits.items():
        g = fr[fr[rank_col].notna() & fr["copyable_roi"].notna() & (fr["copyable_notional"] > 0)]
        for w, seg in g.groupby("wallet", sort=False):
            top = seg[seg[rank_col] >= 0.8]["copyable_roi"].to_numpy(dtype=float)
            rest = seg[seg[rank_col] < 0.8]["copyable_roi"].to_numpy(dtype=float)
            if len(top) < 10 or len(rest) < 10:
                continue
            st = top.mean() / top.std(ddof=1) if top.std(ddof=1) > 0 else np.nan
            sr = rest.mean() / rest.std(ddof=1) if rest.std(ddof=1) > 0 else np.nan
            if np.isnan(st) or np.isnan(sr):
                continue
            out[split].append({
                "wallet": w, "n_top": int(len(top)), "n_rest": int(len(rest)),
                "sharpe_top": float(st), "sharpe_rest": float(sr),
                "diff": float(st - sr),
            })
    summary: dict[str, dict] = {}
    for split, rows in out.items():
        diffs = np.array([np.clip(r["diff"], -10.0, 10.0) for r in rows], dtype=float)
        tops = np.array([np.clip(r["sharpe_top"], -10.0, 10.0) for r in rows], dtype=float)
        rests = np.array([np.clip(r["sharpe_rest"], -10.0, 10.0) for r in rows], dtype=float)
        n = len(diffs)
        if n == 0:
            summary[split] = {"n_wallets": 0}
            continue
        boot = np.array([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(2000)])
        summary[split] = {
            "n_wallets": n,
            "mean_diff": float(diffs.mean()),
            "median_diff": float(np.median(diffs)),
            "p_gt0": float((diffs > 0).mean()),
            "ci90_lo": float(np.percentile(boot, 5)),
            "ci90_hi": float(np.percentile(boot, 95)),
            "mean_sharpe_top": float(tops.mean()),
            "mean_sharpe_rest": float(rests.mean()),
        }
    return {"rows": out, "summary": summary}


def run_bins(splits: dict[str, pd.DataFrame], tag_suffix: str = "") -> None:
    print("\n" + "=" * 78, flush=True)
    print("Phase C: conditional within-wallet exposure bin diagnostic", flush=True)
    print("=" * 78, flush=True)

    for axis in ("exposure", "marg"):
        splits = attach_wallet_rank(splits, axis)
        rows = bin_table_rows(splits, axis)
        pd.DataFrame(rows).to_csv(f"exposure_bins_{axis}{tag_suffix}.csv", index=False)
        price_rows = bin_price_rows(splits, axis)
        pd.DataFrame(price_rows).to_csv(f"exposure_bins_{axis}_price{tag_suffix}.csv", index=False)

        print(f"\n--- within-wallet rank bin means by split (axis={axis}) ---", flush=True)
        tab = pd.DataFrame(rows)
        piv = tab.pivot_table(index="bin", columns="split", values="mean_roi_res")
        print(piv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)
        spiv = tab.pivot_table(index="bin", columns="split", values="sharpe_copyable")
        print("\nper-trade copyable Sharpe by bin:", flush=True)
        print(spiv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)

    w = wallet_sharpe_rows(splits, "exposure")
    pd.DataFrame([r for s in w["rows"].values() for r in s]).to_csv(
        f"exposure_wallet_bins{tag_suffix}.csv", index=False)
    print("\nwallet-level copyable Sharpe top-20% vs rest (train-fit rank):", flush=True)
    for split, s in w["summary"].items():
        if s.get("n_wallets", 0) == 0:
            print(f"  {split}: no wallets with >=10 top+rest trades", flush=True)
            continue
        print(
            f"  {split}: n={s['n_wallets']:3d}  mean diff={s['mean_diff']:+.4f}  "
            f"median diff={s['median_diff']:+.4f}  p(>0)={s['p_gt0']:.2f}  "
            f"boot90=[{s['ci90_lo']:+.4f},{s['ci90_hi']:+.4f}]  "
            f"sharpe top/rest={s['mean_sharpe_top']:+.3f}/{s['mean_sharpe_rest']:+.3f}",
            flush=True)
    pd.DataFrame(w["summary"]).T.reset_index().rename(columns={"index": "split"}).to_csv(
        f"exposure_wallet_summary{tag_suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--tags", nargs="*", default=None,
                        help="Data tags to load (default: Politics)")
    parser.add_argument("--cached-splits-dir", type=str, default=None,
                        help="load-or-prepare depth-capped candidate splits as parquet")
    parser.add_argument("--ic", action="store_true", default=False)
    parser.add_argument("--capture", action="store_true", default=False)
    parser.add_argument("--bins", action="store_true", default=False)
    parser.add_argument("--unrestricted", action="store_true", default=False,
                        help="Run sizing under the unrestricted sim (budget=inf, "
                             "proportional-to-wallet, no skip-race).")
    args = parser.parse_args()
    do_ic = args.ic or not (args.capture or args.bins)
    do_capture = args.capture or not (args.ic or args.bins)
    do_bins = args.bins or not (args.ic or args.capture)

    tags = set(args.tags) if args.tags else DEFAULT_TAGS
    tag_suffix = "" if tags == DEFAULT_TAGS else "-" + "-".join(sorted(tags))

    splits = None
    cache_ready = (
        args.cached_splits_dir is not None
        and all(
            (Path(args.cached_splits_dir) / f"{s}.parquet").exists()
            for s in ("train", "val", "test")
        )
    )
    if not (do_ic or do_capture) and cache_ready:
        print("Loading cached candidate splits (skipping raw data load)...", flush=True)
        splits = load_cached_splits(args.cached_splits_dir)
    else:
        print("Loading stage-1 data...", flush=True)
        df_full, _t, _v, _x, wallet_metrics, hold_metrics = load_stage1_data(
            tags=tags, max_shards=args.max_shards
        )
        if do_ic:
            run_ic(df_full, wallet_metrics, hold_metrics, tag_suffix)
        if do_capture or do_bins:
            splits = get_candidate_splits(df_full, wallet_metrics, hold_metrics, args.cached_splits_dir)
        if do_capture:
            run_capture(splits, tag_suffix, unrestricted=args.unrestricted)
    if do_bins and splits is not None:
        run_bins(splits, tag_suffix)


if __name__ == "__main__":
    main()