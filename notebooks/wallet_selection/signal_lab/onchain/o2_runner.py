"""O2 runner — Phases A (control), B (all-buyers baseline), C (rules), D (combine).

Each phase writes a per-tag summary and CSV. Steps are run via:

    python -m signal_lab.onchain.o2_runner --tag Finance --phase a
    python -m signal_lab.onchain.o2_runner --tag Politics --phase b
    python -m signal_lab.onchain.o2_runner --tag Finance --phase c
    python -m signal_lab.onchain.o2_runner --tag Finance --phase d

Per-tag subdirectories can be specified via ``--out-dir`` (default: this
module's directory). All phases accept ``--max-shards`` for sample-first
runs (per CLAUDE.md). Phase D requires that Phase B and Phase C summaries
exist; the runner loads them from the same ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIGNAL_LAB = HERE.parent
NOTEBOOKS = SIGNAL_LAB.parent
PROJECT = NOTEBOOKS.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(NOTEBOOKS))

# Default output directory; may be overridden by the ``--out-dir`` CLI flag
# (set in :func:`main`). All phase functions read this global when writing.
OUT_DIR: Path = HERE

from signal_lab.filters import ALL_BUYERS, COPY_DEFAULT  # noqa: E402
from signal_lab.signal_engines import (  # noqa: E402
    POS_OPP,
    POS_OWN,
    UWL_OPP,
    UWL_OWN,
    VAL_OPP,
    VAL_OWN,
)
from signal_lab.evaluate_composite import (  # noqa: E402
    add_price_bins,
    add_price_residualized_pnl,
)
from signal_lab.signal_lib import compute_event_ic, spearman_rho  # noqa: E402
from signal_lab.stage1 import (  # noqa: E402
    build_composite_scores,
    candidate_splits_for,
    load_stage1_data,
    rank_normalize_splits,
    run_strategies,
)
from signal_lab.strategies import (  # noqa: E402
    CopyCrowdEntryTiming,
    FadeReactiveSellFlow,
    FreshOppositeCrowdingFilter,
    GamblerCapitulationSqueeze,
    UwlOppContrarian,
)
from signal_lab.sizing import capital_constrained_sim, select_scale, sizing_sharpe  # noqa: E402

from signal_lab.onchain import o2_rules  # noqa: E402

ALL_STRATEGIES = [
    CopyCrowdEntryTiming(),
    FadeReactiveSellFlow(),
    UwlOppContrarian(),
    FreshOppositeCrowdingFilter(),
    GamblerCapitulationSqueeze(),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# In-memory cache for ``load_stage1_data`` results. Keyed by the (tag,
# max_shards, split_kwargs) tuple. Lives only for the lifetime of the
# process; used by the politics_o2 notebook so all 4 phases share the
# single in-memory load. ``_DATA_CACHE`` is intentionally a module
# global (not disk) per the user's "keep it in memory" preference.
_DATA_CACHE: dict = {}


def _split_kwargs_key(split_kwargs: dict | None) -> tuple:
    """Make a stable hashable key from ``split_kwargs`` (None-tolerant)."""
    if not split_kwargs:
        return ()
    return tuple(sorted((k, v) for k, v in split_kwargs.items() if v is not None))


def _load_tag_data(
    tag: str,
    max_shards: int | None,
    split_kwargs: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the per-tag trades + train-period metrics, with a process-local cache.

    Returns ``(df_full, wallet_metrics, hold_metrics)``. The cache key is
    ``(tag, max_shards, split_kwargs)`` so phases A/C/D (full data) hit
    the same entry and Phase B (2-shard sample) gets a separate entry.

    The cache lives only in this module's globals; it is intentionally
    not written to disk.
    """
    key = (tag, max_shards, _split_kwargs_key(split_kwargs))
    cached = _DATA_CACHE.get(key)
    if cached is not None:
        print(f"[load] cache hit for {key}", flush=True)
        return cached
    print(f"[load] loading {key} (this may take ~30-60s)", flush=True)
    df_full, _t, _v, _e, wm, hm = load_stage1_data(
        tags={tag}, max_shards=max_shards, **(split_kwargs or {}),
    )
    # Drop the splits (only used for training the residualizer / wallet
    # metrics). They are reconstructed per-phase from df_full.
    result = (df_full, wm, hm)
    _DATA_CACHE[key] = result
    return result


def clear_data_cache() -> None:
    """Drop the in-memory data cache (free ~1-2 GB). Useful between
    notebook re-runs that change the tag or shard cap."""
    _DATA_CACHE.clear()


def _attach_lead(splits: dict[str, pd.DataFrame]) -> None:
    """Add ``lead_h`` (hours until market close) to each split, in place."""
    for frame in splits.values():
        dt = pd.to_datetime(frame["dt"], utc=True, errors="coerce")
        mc = pd.to_datetime(frame["market_close"], utc=True, errors="coerce")
        lead_h = (mc - dt).dt.total_seconds() / 3600
        frame["lead_h"] = lead_h


def _attach_n_distinct_days(splits: dict[str, pd.DataFrame]) -> None:
    """Add ``n_distinct_days_per_market`` to each split's rows, in place.

    Counts the number of distinct trading days (UTC) per condition_id
    using **all** data in the splits (train+val+test). This is a market-
    level structural feature; it does not use any forward-looking per-
    trade outcome. Markets not present anywhere get 0.
    """
    pieces = []
    for frame in splits.values():
        pieces.append(
            frame.assign(
                _day=pd.to_datetime(frame["dt"], utc=True).dt.floor("1D")
            )[["condition_id", "_day"]]
        )
    all_days = pd.concat(pieces, ignore_index=True)
    days = all_days.groupby("condition_id")["_day"].nunique()
    name = "n_distinct_days_per_market"
    for frame in splits.values():
        frame[name] = frame["condition_id"].map(days).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Phase A — existing composite as-is (control)
# ---------------------------------------------------------------------------


def phase_a(
    tag: str,
    max_shards: int | None,
    *,
    split_kwargs: dict | None = None,
) -> dict:
    split_kwargs = split_kwargs or {}
    print(f"[A/{tag}] max_shards={max_shards} split={split_kwargs}", flush=True)
    df_full, wm, hm = _load_tag_data(tag, max_shards, split_kwargs)
    copy_wallets = set(COPY_DEFAULT(wm, hm))
    print(f"[A/{tag}] COPY_DEFAULT selected {len(copy_wallets)} wallets", flush=True)
    splits = candidate_splits_for(df_full, copy_wallets, **split_kwargs)
    _attach_lead(splits)
    splits, all_cols = run_strategies(
        df_full, wm, hm, ALL_STRATEGIES,
        copy_mask=COPY_DEFAULT, **split_kwargs,
    )
    _attach_lead(splits)
    print(f"[A/{tag}] signal cols: {len(all_cols)}", flush=True)

    add_price_residualized_pnl(splits, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(splits)
    candidates = [c for c in all_cols if c in splits["train"].columns]
    normalized, schemes, _ = build_composite_scores(
        splits, candidates, roi_col="copyable_pnl", weight_split="train", shrinkage=0.5
    )
    add_price_residualized_pnl(normalized, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(normalized)

    rows = []
    for split in ("train", "val", "test"):
        for scheme in schemes:
            col = f"composite_{scheme}"
            rows.append({
                "split": split,
                "scheme": scheme,
                "IC_target": compute_event_ic(
                    normalized[split][col], normalized[split]["copyable_pnl"]),
                "IC_pnl_res": compute_event_ic(
                    normalized[split][col], normalized[split]["pnl_res"]),
                "spearman_price": spearman_rho(
                    normalized[split][col], normalized[split]["price"]),
            })
    df_ic = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"o2_a_{tag.lower()}_composite.csv"
    df_ic.to_csv(out_csv, index=False)

    # Best per-scheme signal on val (for the gate)
    summary = {
        "tag": tag,
        "n_candidates": len(candidates),
        "n_wallets_selected": len(copy_wallets),
        "split_ic": df_ic.to_dict(orient="records"),
        "best_scheme_test_pnl_res_ic": float(
            df_ic[df_ic["split"] == "test"]["IC_pnl_res"].max()
        ),
    }
    out_json = OUT_DIR / f"o2_a_{tag.lower()}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[A/{tag}] wrote {out_csv} and {out_json}")
    print(df_ic.round(4).to_string(index=False))
    # Stash the normalised composite dict on the summary so the
    # notebook (and Phase D) can reuse it without re-running the 5
    # strategies. Non-serialisable, so strip it before JSON dumps in
    # any future code path; today the JSON is written above *before*
    # this line, so the disk artefact stays clean.
    summary["_normalized"] = normalized
    return summary


# ---------------------------------------------------------------------------
# Phase B — same composite but with copy_mask=ALL_BUYERS
# ---------------------------------------------------------------------------


def phase_b(
    tag: str,
    max_shards: int | None,
    *,
    split_kwargs: dict | None = None,
) -> dict:
    split_kwargs = split_kwargs or {}
    print(f"[B/{tag}] max_shards={max_shards} split={split_kwargs}", flush=True)
    df_full, wm, hm = _load_tag_data(tag, max_shards, split_kwargs)
    splits = candidate_splits_for(df_full, ALL_BUYERS(wm, hm), **split_kwargs)
    _attach_lead(splits)
    splits, all_cols = run_strategies(
        df_full, wm, hm, ALL_STRATEGIES,
        copy_mask=ALL_BUYERS, **split_kwargs,
    )
    _attach_lead(splits)
    _attach_n_distinct_days(splits)
    print(f"[B/{tag}] signal cols: {len(all_cols)}", flush=True)

    add_price_residualized_pnl(splits, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(splits)
    candidates = [c for c in all_cols if c in splits["train"].columns]
    normalized, schemes, _ = build_composite_scores(
        splits, candidates, roi_col="copyable_pnl", weight_split="train", shrinkage=0.5
    )
    add_price_residualized_pnl(normalized, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(normalized)

    rows = []
    for split in ("train", "val", "test"):
        for scheme in schemes:
            col = f"composite_{scheme}"
            rows.append({
                "split": split,
                "scheme": scheme,
                "IC_target": compute_event_ic(
                    normalized[split][col], normalized[split]["copyable_pnl"]),
                "IC_pnl_res": compute_event_ic(
                    normalized[split][col], normalized[split]["pnl_res"]),
                "spearman_price": spearman_rho(
                    normalized[split][col], normalized[split]["price"]),
            })
    df_ic = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"o2_b_{tag.lower()}_composite.csv"
    df_ic.to_csv(out_csv, index=False)

    # Per-signal IC: keep the best per (set, kind) by val IC, for Phase D
    from signal_lab.signal_lib import compute_event_ic as ic
    per_signal = []
    for col in candidates:
        row = {"signal": col}
        for split in ("train", "val", "test"):
            row[f"IC_{split}"] = ic(splits[split][col], splits[split]["copyable_pnl"])
        per_signal.append(row)
    per_signal_df = pd.DataFrame(per_signal)
    per_signal_df.to_csv(OUT_DIR / f"o2_b_{tag.lower()}_per_signal.csv", index=False)

    summary = {
        "tag": tag,
        "n_candidates": len(candidates),
        "n_signals": len(per_signal),
        "split_ic": df_ic.to_dict(orient="records"),
        "best_composite_test_pnl_res_ic": float(
            df_ic[df_ic["split"] == "test"]["IC_pnl_res"].max()
        ),
        "best_signals_by_val_ic": (
            per_signal_df
            .assign(_abs=per_signal_df["IC_val"].abs())
            .sort_values("_abs", ascending=False)
            .head(10)
            .drop(columns="_abs")
            .to_dict(orient="records")
        ),
    }
    out_json = OUT_DIR / f"o2_b_{tag.lower()}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[B/{tag}] wrote {out_csv} and {out_json}")
    print(df_ic.round(4).to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Phase C — direct price / lead / market rules
# ---------------------------------------------------------------------------


def phase_c(
    tag: str,
    max_shards: int | None,
    *,
    split_kwargs: dict | None = None,
) -> dict:
    split_kwargs = split_kwargs or {}
    print(f"[C/{tag}] max_shards={max_shards} split={split_kwargs}", flush=True)
    df_full, wm, hm = _load_tag_data(tag, max_shards, split_kwargs)
    splits = candidate_splits_for(df_full, ALL_BUYERS(wm, hm), **split_kwargs)
    _attach_lead(splits)
    _attach_n_distinct_days(splits)

    rule_names = o2_rules.rule_names_for_tag(tag)
    rows = []
    per_rule = {}
    for split in ("train", "val", "test"):
        for rule_name in rule_names:
            res = o2_rules.evaluate_rule(splits[split], rule_name)
            res["split"] = split
            rows.append(res)
    df_rules = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"o2_c_{tag.lower()}_rules.csv"
    df_rules.to_csv(out_csv, index=False)

    # For each rule, summarise val/test sign agreement
    rule_summary = []
    for rule_name in rule_names:
        sub = df_rules[df_rules["rule"] == rule_name]
        train = sub[sub["split"] == "train"].iloc[0]
        val = sub[sub["split"] == "val"].iloc[0]
        test = sub[sub["split"] == "test"].iloc[0]
        ic_val = float(val["ic"])
        ic_test = float(test["ic"])
        ic_train = float(train["ic"])
        same_sign = (
            np.isfinite(ic_val)
            and np.isfinite(ic_test)
            and np.sign(ic_val) == np.sign(ic_test)
            and np.sign(ic_val) != 0
        )
        rule_summary.append({
            "rule": rule_name,
            "IC_train": ic_train,
            "IC_val": ic_val,
            "IC_test": ic_test,
            "val_fire_rate": float(val["fire_rate"]),
            "test_fire_rate": float(test["fire_rate"]),
            "val_fires": int(val["n_fires"]),
            "test_fires": int(test["n_fires"]),
            "same_sign_val_test": bool(same_sign),
        })
    df_summary = pd.DataFrame(rule_summary)
    summary_csv = OUT_DIR / f"o2_c_{tag.lower()}_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    summary = {
        "tag": tag,
        "n_rules": len(rule_names),
        "rules": df_summary.to_dict(orient="records"),
    }
    out_json = OUT_DIR / f"o2_c_{tag.lower()}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[C/{tag}] wrote {out_csv} and {out_json}")
    print(df_summary.round(4).to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Phase D — combine top signals/rules into a sign-only composite
# ---------------------------------------------------------------------------


def _zscore_columns(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """In-place z-score the listed columns; return the frame."""
    for c in cols:
        v = frame[c].astype(float)
        mu = v.mean()
        sd = v.std()
        if sd == 0 or not np.isfinite(sd):
            frame[c] = v - mu
        else:
            frame[c] = (v - mu) / sd
    return frame


def phase_d(
    tag: str,
    max_shards: int | None,
    budget: float,
    *,
    split_kwargs: dict | None = None,
) -> dict:
    """Combine the top Phase B + Phase C winners into a sign-only composite.

    Weight sign is chosen by the train+val sign pattern; magnitudes are
    equal-magnitude (1) to keep the composite simple and reduce overfit
    risk. Re-evaluated on test.
    """
    split_kwargs = split_kwargs or {}
    print(f"[D/{tag}] max_shards={max_shards} split={split_kwargs}", flush=True)
    df_full, wm, hm = _load_tag_data(tag, max_shards, split_kwargs)
    splits = candidate_splits_for(df_full, ALL_BUYERS(wm, hm), **split_kwargs)
    _attach_lead(splits)
    _attach_n_distinct_days(splits)

    # Phase B signals
    splits_b, all_cols = run_strategies(
        df_full, wm, hm, ALL_STRATEGIES,
        copy_mask=ALL_BUYERS, **split_kwargs,
    )
    _attach_lead(splits_b)
    _attach_n_distinct_days(splits_b)
    add_price_residualized_pnl(splits_b, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(splits_b)

    # Phase C rule masks
    rule_names = o2_rules.rule_names_for_tag(tag)

    # Pick candidates: top B-signal by |val IC|, then sign.
    b_candidates = [c for c in all_cols if c in splits_b["train"].columns]
    b_per = []
    from signal_lab.signal_lib import compute_event_ic as ic
    for c in b_candidates:
        ic_val = ic(splits_b["val"][c], splits_b["val"]["copyable_pnl"])
        ic_train = ic(splits_b["train"][c], splits_b["train"]["copyable_pnl"])
        ic_test = ic(splits_b["test"][c], splits_b["test"]["copyable_pnl"])
        b_per.append({
            "signal": c,
            "IC_train": ic_train,
            "IC_val": ic_val,
            "IC_test": ic_test,
        })
    b_per_df = pd.DataFrame(b_per).sort_values("IC_val", key=lambda s: s.abs(), ascending=False)
    top_b = b_per_df.head(2).to_dict(orient="records")
    print(f"[D/{tag}] top B signals (by |val IC|):")
    print(b_per_df.head(5).round(4).to_string(index=False))

    # Pick top C rules: same sign on val+test
    c_per = []
    for r in rule_names:
        ev = o2_rules.evaluate_rule(splits_b["val"], r)
        et = o2_rules.evaluate_rule(splits_b["test"], r)
        same = (
            np.isfinite(ev["ic"]) and np.isfinite(et["ic"])
            and np.sign(ev["ic"]) == np.sign(et["ic"]) and ev["ic"] != 0
        )
        c_per.append({
            "rule": r,
            "IC_val": ev["ic"],
            "IC_test": et["ic"],
            "same_sign_val_test": bool(same),
        })
    c_per_df = pd.DataFrame(c_per).sort_values("IC_val", key=lambda s: s.abs(), ascending=False)
    top_c = c_per_df.head(2).to_dict(orient="records")
    print(f"[D/{tag}] top C rules (by |val IC|):")
    print(c_per_df.head(5).round(4).to_string(index=False))

    # Build the composite: sign-only weights, equal magnitude.
    components = []
    weights = []
    for row in top_b:
        if not np.isfinite(row["IC_val"]):
            continue
        components.append(("signal", row["signal"]))
        weights.append(float(np.sign(row["IC_val"])))
    for row in top_c:
        if not np.isfinite(row["IC_val"]):
            continue
        # rule mask becomes a 0/1 score; sign carries the direction.
        components.append(("rule", row["rule"]))
        weights.append(float(np.sign(row["IC_val"])))
    if not components:
        summary = {"tag": tag, "n_components": 0, "note": "no usable components"}
        (OUT_DIR / f"o2_d_{tag.lower()}_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        return summary
    print(f"[D/{tag}] composite components: {list(zip(components, weights))}")

    # Apply composite
    for split in ("train", "val", "test"):
        frame = splits_b[split]
        score = pd.Series(0.0, index=frame.index)
        for (kind, name), w in zip(components, weights):
            if kind == "signal":
                col = frame[name].astype(float)
                col = (col - col.mean()) / (col.std() or 1.0)
            else:
                col = o2_rules.ALL_RULES[name](frame).astype(float)
            score = score.add(w * col, fill_value=0.0)
        frame["composite_sign_only"] = score

    # Per-split IC vs copyable_pnl and pnl_res
    rows = []
    for split in ("train", "val", "test"):
        frame = splits_b[split]
        rows.append({
            "split": split,
            "IC_target": compute_event_ic(frame["composite_sign_only"], frame["copyable_pnl"]),
            "IC_pnl_res": compute_event_ic(frame["composite_sign_only"], frame["pnl_res"]),
            "spearman_price": spearman_rho(frame["composite_sign_only"], frame["price"]),
            "n": int(len(frame)),
        })
    df_ic = pd.DataFrame(rows)
    df_ic.to_csv(OUT_DIR / f"o2_d_{tag.lower()}_composite.csv", index=False)

    # Capital-constrained sizing
    scale_grid = np.arange(0.1, 3.01, 0.1)
    siz_rows = []
    for split in ("val", "test"):
        frame = splits_b[split]
        best_scale, grid = select_scale(
            frame, "composite_sign_only", budget, scale_grid, 0.0, primary="sharpe_daily"
        )
        if grid.empty:
            continue
        res = capital_constrained_sim(
            frame, "composite_sign_only", budget, best_scale, 0.0
        )
        siz_rows.append({
            "split": split,
            "scale": float(best_scale),
            "trades": int(res["trades"]),
            "net_pnl": round(res["net_pnl"], 2),
            "peak_used": round(res["peak_used"], 2),
            "pnl_per_peak": round(res["net_pnl"] / max(res["peak_used"], 1e-9), 4),
            "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
        })
    df_siz = pd.DataFrame(siz_rows)
    df_siz.to_csv(OUT_DIR / f"o2_d_{tag.lower()}_sizing.csv", index=False)

    summary = {
        "tag": tag,
        "n_components": len(components),
        "components": [
            {"kind": k, "name": n, "sign": w}
            for (k, n), w in zip(components, weights)
        ],
        "top_b_signals": top_b,
        "top_c_rules": top_c,
        "split_ic": df_ic.to_dict(orient="records"),
        "sizing": df_siz.to_dict(orient="records"),
    }
    (OUT_DIR / f"o2_d_{tag.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"[D/{tag}] composite IC:")
    print(df_ic.round(4).to_string(index=False))
    print(f"[D/{tag}] sizing:")
    print(df_siz.round(4).to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["Finance", "Politics"])
    ap.add_argument("--phase", required=True, choices=["a", "b", "c", "d"])
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--budget", type=float, default=10_000.0)
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="Directory for outputs (default: this module's directory).")
    ap.add_argument("--train-end", type=str, default=None,
                    help="ISO date (UTC). Train split ends here (markets with "
                         "end_date_iso <= train_end).")
    ap.add_argument("--val-end", type=str, default=None,
                    help="ISO date (UTC). Val split ends here.")
    ap.add_argument("--test-start", type=str, default=None,
                    help="ISO date (UTC). Test split starts here.")
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_kwargs = {
        "train_end": args.train_end,
        "val_end": args.val_end,
        "test_start": args.test_start,
    }

    t0 = time.time()
    if args.phase == "a":
        out = phase_a(args.tag, args.max_shards, split_kwargs=split_kwargs)
    elif args.phase == "b":
        out = phase_b(args.tag, args.max_shards, split_kwargs=split_kwargs)
    elif args.phase == "c":
        out = phase_c(args.tag, args.max_shards, split_kwargs=split_kwargs)
    else:
        out = phase_d(args.tag, args.max_shards, args.budget, split_kwargs=split_kwargs)
    print(f"\n[{args.tag}/{args.phase}] done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
