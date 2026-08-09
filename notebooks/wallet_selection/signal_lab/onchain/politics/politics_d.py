"""Politics Phase D — combine Phase A composite with the strongest Phase C rule.

This is a focused variant of :mod:`o2_runner` for Politics that avoids the
slow ``run_strategies`` call (which timed out for the all-buyers universe
on Politics in the original Phase D). It reuses the existing
``o2_a_politics_*`` artefacts and the per-split ``price_lt_0p1`` mask.

The composite is the sign-only sum of the Phase A shrinkage_markowitz
score (when available) and the ``price_lt_0p1`` mask. Capital-constrained
sizing at $10k.

Run:

    python -m signal_lab.onchain.politics.politics_d
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ONCHAIN = HERE.parent
SIGNAL_LAB = ONCHAIN.parent
NOTEBOOKS = SIGNAL_LAB.parent
PROJECT = NOTEBOOKS.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(NOTEBOOKS))

from signal_lab.evaluate_composite import (  # noqa: E402
    add_price_bins,
    add_price_residualized_pnl,
    build_composite_scores,
)
from signal_lab.filters import ALL_BUYERS, COPY_DEFAULT  # noqa: E402
from signal_lab.onchain import o2_rules  # noqa: E402
from signal_lab.onchain.o2_runner import _attach_lead, _load_tag_data  # noqa: E402
from signal_lab.sizing import capital_constrained_sim, select_scale, sizing_sharpe  # noqa: E402
from signal_lab.signal_lib import compute_event_ic, spearman_rho  # noqa: E402
from signal_lab.stage1 import candidate_splits_for  # noqa: E402
from signal_lab.strategies import (  # noqa: E402
    CopyCrowdEntryTiming,
    FadeReactiveSellFlow,
    FreshOppositeCrowdingFilter,
    GamblerCapitulationSqueeze,
    UwlOppContrarian,
)


ALL_STRATEGIES = [
    CopyCrowdEntryTiming(),
    FadeReactiveSellFlow(),
    UwlOppContrarian(),
    FreshOppositeCrowdingFilter(),
    GamblerCapitulationSqueeze(),
]

A_COMPOSITE = "composite_shrinkage_markowitz"
RULE_NAME = "price_lt_0p1"
BUDGET = 10_000.0

# Default split: train ends 2026-02-01, val ends 2026-05-31, test starts
# 2026-06-01. Override via the ``--train-end``/``--val-end``/``--test-start``
# CLI flags or by passing ``split_kwargs`` programmatically.
DEFAULT_SPLIT = {
    "train_end": "2026-02-01",
    "val_end": "2026-05-31",
    "test_start": "2026-06-01",
}


def _build_phase_a_composite(
    split_kwargs: dict,
) -> dict:
    """Build the Phase A ``shrinkage_markowitz`` composite on the
    COPY_DEFAULT subset. Returns ``{split: frame_with_A_COMPOSITE}``.

    The composite is the per-row value of the shrinkage-weighted
    combination of the 5 strategies' signal columns, computed on the
    train split and applied to train/val/test.

    Used when :func:`_attach_composite_and_rule` is called without a
    pre-built ``phase_a_norm`` (i.e. when run as a standalone CLI).
    """
    df_full, wm, hm = _load_tag_data("Politics", None, split_kwargs)
    splits_a = candidate_splits_for(
        df_full, COPY_DEFAULT(wm, hm), **split_kwargs,
    )

    from signal_lab.stage1 import run_strategies as run_strats
    splits_a, all_cols = run_strats(
        df_full, wm, hm, ALL_STRATEGIES,
        copy_mask=COPY_DEFAULT, **split_kwargs,
    )
    add_price_residualized_pnl(splits_a, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(splits_a)
    candidates = [c for c in all_cols if c in splits_a["train"].columns]
    norm_a, _schemes, _ = build_composite_scores(
        splits_a, candidates, roi_col="copyable_pnl",
        weight_split="train", shrinkage=0.5,
    )
    return norm_a


def _attach_composite_and_rule(
    splits: dict[str, pd.DataFrame],
    *,
    split_kwargs: dict | None = None,
    phase_a_norm: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Attach ``A_composite`` (Phase A shrinkage_markowitz) and ``rule_mask`` (RULE_NAME) to the all-buyers splits.

    Returns a dict ``{split: frame}`` with the new columns.

    If ``phase_a_norm`` is provided (a dict from Phase A with the
    ``composite_shrinkage_markowitz`` column already attached), the
    full Phase A strategy run is skipped — the composite is read
    directly from it. This is the fast path used by the
    ``politics_o2`` notebook when Phase A has already been run in the
    same process.
    """
    split_kwargs = {**DEFAULT_SPLIT, **(split_kwargs or {})}
    # 1. Phase A composite on COPY_DEFAULT: either reuse or rebuild.
    if phase_a_norm is None:
        norm_a = _build_phase_a_composite(split_kwargs)
    else:
        norm_a = phase_a_norm

    # 2. Build the all-buyers splits; attach lead, the rule mask, and the A
    # composite mapped from each COPY_DEFAULT trade to the all-buyers frame.
    df_full, wm, hm = _load_tag_data("Politics", None, split_kwargs)
    splits_b = candidate_splits_for(
        df_full, ALL_BUYERS(wm, hm), **split_kwargs,
    )
    _attach_lead(splits_b)
    add_price_residualized_pnl(splits_b, target_col="copyable_pnl", out_col="pnl_res")
    add_price_bins(splits_b)

    for split in ("train", "val", "test"):
        a_frame = norm_a[split][["wallet", "condition_id", "dt", A_COMPOSITE]].copy()
        a_frame["dt"] = pd.to_datetime(a_frame["dt"], utc=True)
        b_frame = splits_b[split]
        b_frame["dt"] = pd.to_datetime(b_frame["dt"], utc=True)
        # Vectorised join on (wallet, condition_id, dt)
        merged = b_frame.merge(
            a_frame, on=["wallet", "condition_id", "dt"], how="left"
        )
        merged[A_COMPOSITE] = merged[A_COMPOSITE].fillna(0.0)
        rule_mask = (
            o2_rules.ALL_RULES[RULE_NAME](merged).fillna(False).astype(int)
        )
        merged["rule_mask"] = rule_mask.astype(float)
        # Sign-only composite: +1 on the A score, +1 on the rule mask.
        a_z = (
            (merged[A_COMPOSITE] - merged[A_COMPOSITE].mean())
            / (merged[A_COMPOSITE].std() or 1.0)
        )
        merged["composite"] = a_z + merged["rule_mask"]
        splits_b[split] = merged
    return splits_b


def _sizing_summary(res: dict, scale: float, split: str, frame: pd.DataFrame) -> dict:
    daily = res["daily_pnl"]
    return {
        "split": split,
        "scale": float(scale),
        "trades": int(res["trades"]),
        "net_pnl": round(res["net_pnl"], 2),
        "peak_used": round(res["peak_used"], 2),
        "mean_used": round(res["mean_used"], 2),
        "pnl_per_peak": round(res["net_pnl"] / max(res["peak_used"], 1e-9), 4),
        "sharpe_daily": round(sizing_sharpe(daily, 365.0), 3),
        "n_candidates": int(len(frame)),
    }


def _save_daily_pnl(splits: dict[str, pd.DataFrame]) -> None:
    """Run sizing on val/test (composite + rule) and write daily-PnL timeseries to disk.

    Saves a wide DataFrame with one column per ``(split, score)`` pair, with
    flat column names ``{v,t}_{combo,rule}`` (test prefixed by ``t_``).
    """
    scale_grid = np.arange(0.1, 3.01, 0.1)
    out = {}
    for split in ("val", "test"):
        frame = splits[split]
        for col in ("composite", "rule_mask"):
            best_scale, _ = select_scale(
                frame, col, BUDGET, scale_grid, 0.0, primary="sharpe_daily"
            )
            res = capital_constrained_sim(
                frame, col, BUDGET, float(best_scale), 0.0
            )
            daily = res["daily_pnl"].sort_index()
            short_split = "v" if split == "val" else "t"
            short_col = "combo" if col == "composite" else "rule"
            out[f"{short_split}_{short_col}"] = daily
    df = pd.concat(out, axis=1)
    # Defensive: if the columns come back as a MultiIndex for any reason,
    # flatten with "_" join; otherwise just stringify the single level.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = ["_".join(map(str, c)) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    df.to_csv(HERE / "o2_d_politics_daily_pnl.csv")
    print(f"[D/Politics] wrote {HERE / 'o2_d_politics_daily_pnl.csv'} ({df.shape})")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", type=str, default=DEFAULT_SPLIT["train_end"])
    p.add_argument("--val-end", type=str, default=DEFAULT_SPLIT["val_end"])
    p.add_argument("--test-start", type=str, default=DEFAULT_SPLIT["test_start"])
    args = p.parse_args(argv)

    split_kwargs = {
        "train_end": args.train_end,
        "val_end": args.val_end,
        "test_start": args.test_start,
    }

    t0 = time.time()
    print(
        f"[D/Politics] building composite on all-buyers frame "
        f"(train_end={args.train_end} val_end={args.val_end} test_start={args.test_start})",
        flush=True,
    )
    # The notebook can pass ``phase_a_norm`` in via a process-global set
    # in the orchestrator cell before this CLI is invoked. Default (CLI
    # use): None → rebuild Phase A composite from scratch.
    phase_a_norm = globals().get("_PHASE_A_NORM_OVERRIDE")
    splits = _attach_composite_and_rule(
        {}, split_kwargs=split_kwargs, phase_a_norm=phase_a_norm,
    )

    rows = []
    for split in ("train", "val", "test"):
        frame = splits[split]
        rows.append({
            "split": split,
            "IC_target_combo": compute_event_ic(
                frame["composite"], frame["copyable_pnl"]),
            "IC_pnl_res_combo": compute_event_ic(
                frame["composite"], frame["pnl_res"]),
            "IC_roi_res_combo": compute_event_ic(
                frame["composite"], frame["roi_res"]),
            "IC_target_rule": compute_event_ic(
                frame["rule_mask"], frame["copyable_pnl"]),
            "IC_pnl_res_rule": compute_event_ic(
                frame["rule_mask"], frame["pnl_res"]),
            "IC_roi_res_rule": compute_event_ic(
                frame["rule_mask"], frame["roi_res"]),
            "spearman_price_combo": spearman_rho(
                frame["composite"], frame["price"]),
            "spearman_price_rule": spearman_rho(
                frame["rule_mask"], frame["price"]),
            "n": int(len(frame)),
        })
    df_ic = pd.DataFrame(rows)
    df_ic.to_csv(HERE / "o2_d_politics_composite.csv", index=False)

    # Sizing: pick scale on val by daily Sharpe, report val and test.
    scale_grid = np.arange(0.1, 3.01, 0.1)
    siz_rows = []
    for split in ("val", "test"):
        frame = splits[split]
        best_scale, _ = select_scale(
            frame, "composite", BUDGET, scale_grid, 0.0, primary="sharpe_daily"
        )
        res = capital_constrained_sim(frame, "composite", BUDGET, float(best_scale), 0.0)
        siz_rows.append(_sizing_summary(res, best_scale, split, frame))
        # Also: pure rule sizing (for reference / "Strategy P-1" comparison)
        best_rule_scale, _ = select_scale(
            frame, "rule_mask", BUDGET, scale_grid, 0.0, primary="sharpe_daily"
        )
        res_rule = capital_constrained_sim(
            frame, "rule_mask", BUDGET, float(best_rule_scale), 0.0
        )
        siz_rows.append({
            "split": split,
            "scale": float(best_rule_scale),
            "trades": int(res_rule["trades"]),
            "net_pnl": round(res_rule["net_pnl"], 2),
            "peak_used": round(res_rule["peak_used"], 2),
            "mean_used": round(res_rule["mean_used"], 2),
            "pnl_per_peak": round(res_rule["net_pnl"] / max(res_rule["peak_used"], 1e-9), 4),
            "sharpe_daily": round(sizing_sharpe(res_rule["daily_pnl"], 365.0), 3),
            "n_candidates": int(len(frame)),
            "score_col": "rule_mask",
        })
    df_siz = pd.DataFrame(siz_rows)
    df_siz.to_csv(HERE / "o2_d_politics_sizing.csv", index=False)

    summary = {
        "tag": "Politics",
        "rule": RULE_NAME,
        "a_composite_col": A_COMPOSITE,
        "split_ic": df_ic.to_dict(orient="records"),
        "sizing": df_siz.to_dict(orient="records"),
    }
    (HERE / "o2_d_politics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\n[D/Politics] IC table:")
    print(df_ic.round(4).to_string(index=False))
    print(f"\n[D/Politics] sizing:")
    print(df_siz.round(4).to_string(index=False))
    _save_daily_pnl(splits)
    print(f"\n[D/Politics] done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
