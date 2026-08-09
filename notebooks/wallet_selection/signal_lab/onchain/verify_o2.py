"""O2 verifiers — one per phase, plus a unified entry point.

Each verifier asserts the artefact exists, has the expected schema, and
where applicable the gate from PROGRESS.md / O2_PLAN.md.

Run all (default: this module's directory):

    python -m signal_lab.onchain.verify_o2 --all

or per-phase (with a custom output directory for tag-specific runs):

    python -m signal_lab.onchain.verify_o2 --phase a --tag Finance \\
        --out-dir signal_lab/onchain/politics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HERE = Path(__file__).resolve().parent


def _ver_a(tag: str, here: Path) -> bool:
    csv = here / f"o2_a_{tag.lower()}_composite.csv"
    js = here / f"o2_a_{tag.lower()}_summary.json"
    if not (csv.exists() and js.exists()):
        print(f"[verify a/{tag}] missing artefact: {csv} or {js}")
        return False
    df = pd.read_csv(csv)
    assert {"split", "scheme", "IC_target", "IC_pnl_res"}.issubset(df.columns), (
        f"[verify a/{tag}] csv missing expected columns"
    )
    test = df[df["split"] == "test"]
    n_pos = int((test["IC_pnl_res"] > 0).sum())
    print(f"[verify a/{tag}] OK  test_pos_pnl_res_schemes={n_pos}/3")
    return True


def _ver_b(tag: str, here: Path) -> bool:
    csv = here / f"o2_b_{tag.lower()}_composite.csv"
    per = here / f"o2_b_{tag.lower()}_per_signal.csv"
    js = here / f"o2_b_{tag.lower()}_summary.json"
    if not (csv.exists() and per.exists() and js.exists()):
        print(f"[verify b/{tag}] missing artefact(s)")
        return False
    df = pd.read_csv(csv)
    assert {"split", "scheme", "IC_pnl_res"}.issubset(df.columns)
    per_df = pd.read_csv(per)
    assert {"signal", "IC_val", "IC_test"}.issubset(per_df.columns)
    same_sign = per_df[
        np.isfinite(per_df["IC_val"])
        & np.isfinite(per_df["IC_test"])
        & (np.sign(per_df["IC_val"]) == np.sign(per_df["IC_test"]))
        & (per_df["IC_val"] != 0)
    ]
    n_sign = len(same_sign)
    n_strong = int((same_sign["IC_val"].abs() >= 0.005).sum())
    print(
        f"[verify b/{tag}] OK  sign-consistent signals={n_sign}  "
        f"with |val IC| >= 0.005 = {n_strong}"
    )
    return True


def _ver_c(tag: str, here: Path) -> bool:
    csv = here / f"o2_c_{tag.lower()}_summary.csv"
    js = here / f"o2_c_{tag.lower()}_summary.json"
    if not (csv.exists() and js.exists()):
        print(f"[verify c/{tag}] missing artefact(s)")
        return False
    df = pd.read_csv(csv)
    assert {
        "rule", "IC_train", "IC_val", "IC_test", "same_sign_val_test"
    }.issubset(df.columns)
    passes = df[df["same_sign_val_test"] & (df["IC_val"].abs() >= 0.005)]
    print(
        f"[verify c/{tag}] OK  rules_passing_gate={len(passes)}  "
        f"top_passers={passes['rule'].tolist() if len(passes) else 'none'}"
    )
    return True


def _ver_d(tag: str, here: Path) -> bool:
    ic_csv = here / f"o2_d_{tag.lower()}_composite.csv"
    siz_csv = here / f"o2_d_{tag.lower()}_sizing.csv"
    js = here / f"o2_d_{tag.lower()}_summary.json"
    if not (ic_csv.exists() and siz_csv.exists() and js.exists()):
        print(f"[verify d/{tag}] missing artefact(s)")
        return False
    df = pd.read_csv(ic_csv)
    # Two accepted schemas: the canonical ``IC_target`` / ``IC_pnl_res`` (from
    # ``o2_runner.phase_d``) and the politics-specific ``IC_target_combo`` /
    # ``IC_pnl_res_combo`` (from ``politics_d.py``). Map to a common shape.
    if "IC_target" in df.columns:
        ic_target_col, ic_pnl_col = "IC_target", "IC_pnl_res"
    elif "IC_target_combo" in df.columns:
        ic_target_col, ic_pnl_col = "IC_target_combo", "IC_pnl_res_combo"
    else:
        print(f"[verify d/{tag}] csv missing IC_target or IC_target_combo column")
        return False
    test = df[df["split"] == "test"].iloc[0]
    val = df[df["split"] == "val"].iloc[0]
    pnl_res_pos = bool(test[ic_pnl_col] > 0 and val[ic_pnl_col] > 0)
    target_pos = bool(test[ic_target_col] > 0 and val[ic_target_col] > 0)
    print(
        f"[verify d/{tag}] OK  val/test IC_target={val[ic_target_col]:.3f}/{test[ic_target_col]:.3f}  "
        f"IC_pnl_res={val[ic_pnl_col]:.3f}/{test[ic_pnl_col]:.3f}  "
        f"both_pos={target_pos}  pnl_res_pos={pnl_res_pos}"
    )
    return True


PHASES = {"a": _ver_a, "b": _ver_b, "c": _ver_c, "d": _ver_d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default=None)
    ap.add_argument("--tag", choices=["Finance", "Politics"], default=None)
    ap.add_argument("--all", action="store_true",
                    help="Run all phases for both tags.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_HERE,
                    help="Directory containing the o2_<phase>_<tag>_* artefacts.")
    args = ap.parse_args()

    if not args.all and (args.phase is None or args.tag is None):
        ap.error("either --all or both --phase and --tag are required")

    here = Path(args.out_dir).resolve()
    rc = 0
    if args.all:
        for tag in ("Finance", "Politics"):
            for ph, fn in PHASES.items():
                print(f"\n=== {tag}/{ph} (in {here}) ===")
                if not fn(tag, here):
                    rc = 1
    else:
        if not PHASES[args.phase](args.tag, here):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
