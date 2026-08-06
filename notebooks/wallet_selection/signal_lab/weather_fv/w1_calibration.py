"""W1 — market self-calibration baseline for weather markets.

For each BUY trade on a Weather condition, take the trade price as the
market's implied probability at that time. Bucket by:
  - lead_h bucket = (end_date_iso - dt) bucketed
  - price bin = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

Per bucket, compute:
  - n_trades
  - outcome_rate (mean of token_winner)
  - mean_price
  - brier (mean (price - outcome)^2)
  - reliability gap (outcome_rate - mean_price)

Output:
  weather_fv/w1_calibration.csv
  weather_fv/w1_summary.json

The point of W1 is to identify *where* the market is miscalibrated. If the
gap is small everywhere, Track W has limited room and the early-stop gate
fires.

Run modes:
    python w1_calibration.py --sample 10000
    python w1_calibration.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIGNAL_LAB = HERE.parent
NOTEBOOKS = SIGNAL_LAB.parent
PROJECT = NOTEBOOKS.parent.parent
sys.path.insert(0, str(PROJECT))

TRADES_DIR = PROJECT / "data" / "polygon_trades_processed"
PARSED = HERE / "w0_markets_parsed.parquet"
OUT_CSV = HERE / "w1_calibration.csv"
OUT_SUMMARY = HERE / "w1_summary.json"

PRICE_BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0001]
# lead_h = (last_condition_trade_ts - dt) in hours; can be negative.
LEAD_BINS_H = [-1000, 0, 6, 12, 24, 48, 72, 168, 720, 8760]
LEAD_LABELS = [
    "post-close", "[0,6h)", "[6,12h)", "[12,24h)", "[1,2d)",
    "[2,3d)", "[3,7d)", "[7,30d)", "[30d,1y)",
]


def load_trades(sample: int | None) -> pd.DataFrame:
    """Load BUY trades on Weather conditions, with token_winner as label.

    Streams through trade shards, optionally capped at ``sample`` total
    weather BUY trades for fast iteration.
    """
    parsed = pd.read_parquet(PARSED, columns=["condition_id", "date", "end_date_iso", "city", "unit"])
    parsed = parsed.dropna(subset=["end_date_iso", "date"])
    weather_ids = set(parsed["condition_id"].tolist())
    print(f"weather condition_ids: {len(weather_ids):,}")

    pieces = []
    total_kept = 0
    for f in sorted(TRADES_DIR.glob("*.parquet")):
        df = pd.read_parquet(
            f,
            columns=[
                "wallet", "condition_id", "dt", "side", "outcome",
                "avg_price", "token_winner", "final_price",
                "last_condition_trade_ts",
            ],
        )
        df = df[(df["side"] == "BUY") & (df["condition_id"].isin(weather_ids))]
        if sample is not None:
            remaining = sample - total_kept
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.sample(n=remaining, random_state=0)
        pieces.append(df)
        total_kept += len(df)
        print(f"  {f.name}: {len(df):>8,} (kept)  cum={total_kept:,}")
    trades = pd.concat(pieces, ignore_index=True)
    trades = trades.merge(parsed, on="condition_id", how="inner")
    print(f"BUY trades on weather: {len(trades):,}")
    return trades


def add_lead(trades: pd.DataFrame) -> pd.DataFrame:
    """lead_h = (last_condition_trade_ts - dt) in hours.

    The on-chain ``end_date_iso`` is the event date (00:00 UTC of the day
    in the question), not the resolution timestamp. The market typically
    closes 12-24h after that, captured by ``last_condition_trade_ts``.
    We use that as the lead anchor.
    """
    trades = trades.copy()
    trades["dt"] = pd.to_datetime(trades["dt"], utc=True, errors="coerce")
    trades["last_condition_trade_ts"] = pd.to_datetime(
        trades["last_condition_trade_ts"], utc=True, errors="coerce"
    )
    trades = trades.dropna(
        subset=["dt", "last_condition_trade_ts", "avg_price", "token_winner"]
    )
    trades["lead_h"] = (
        trades["last_condition_trade_ts"] - trades["dt"]
    ).dt.total_seconds() / 3600
    # Keep all valid lead times including small negatives (post-close
    # rebalancing trades; they are still informative for outcome).
    trades = trades[trades["lead_h"].abs() <= 8760]
    trades["price_bin"] = pd.cut(
        trades["avg_price"], bins=PRICE_BINS, right=False, labels=False
    )
    trades["lead_bin"] = pd.cut(
        trades["lead_h"], bins=LEAD_BINS_H, right=False, labels=False
    )
    return trades


def brier(group: pd.DataFrame) -> float:
    p = group["avg_price"].astype(float)
    y = group["token_winner"].astype(int)
    return float(((p - y) ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", type=int, default=200_000)
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    sample = None if args.full else args.sample

    trades = load_trades(sample)
    trades = add_lead(trades)
    print(f"after filtering: {len(trades):,}")

    # Global Brier (market price vs outcome)
    global_brier = brier(trades)
    summary = {
        "n_trades": int(len(trades)),
        "global_brier": global_brier,
        "global_brier_baseline_naive": 0.25,  # always predict 0.5
    }

    # Per-bucket stats
    rows = []
    for (lb, pb), g in trades.groupby(["lead_bin", "price_bin"], observed=True):
        if len(g) < 30:
            continue
        rows.append({
            "lead_bin": int(lb),
            "price_bin": int(pb),
            "n": int(len(g)),
            "mean_price": float(g["avg_price"].mean()),
            "outcome_rate": float(g["token_winner"].mean()),
            "brier": brier(g),
            "reliability_gap": float(g["token_winner"].mean() - g["avg_price"].mean()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["lead_label"] = out["lead_bin"].map(dict(enumerate(LEAD_LABELS)))
    out.to_csv(OUT_CSV, index=False)
    summary["n_buckets"] = int(len(out))
    summary["brier_by_lead"] = (
        out.groupby("lead_label")["brier"].mean().to_dict()
        if not out.empty else {}
    )
    summary["gap_by_lead"] = (
        out.groupby("lead_label")["reliability_gap"].mean().to_dict()
        if not out.empty else {}
    )
    summary["gap_by_price_bin"] = (
        out.groupby("price_bin")["reliability_gap"].mean().to_dict()
        if not out.empty else {}
    )
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"summary: {OUT_SUMMARY}")
    print(f"global Brier: {global_brier:.4f}")
    print("brier by lead bin:")
    for k, v in summary["brier_by_lead"].items():
        print(f"  {k}: {v:.4f}")
    print("reliability gap by lead bin:")
    for k, v in summary["gap_by_lead"].items():
        print(f"  {k}: {v:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
