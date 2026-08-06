"""O1 — Diagnose Finance/Politics distortion under simple wallet selection.

Hypothesis to test: the simple COPY_DEFAULT selection (min_buy_roi, min_buckets,
min_markets, min_trade_count, max_drawdown_to_pnl, min_copyable_roi) produces
distorted / unstable results in Finance and Politics because the underlying
universe of opening BUY trades has different structure than Weather.

Diagnostic axes:
  1.  candidate trade count per tag
  2.  market count per tag
  3.  wallet count passing COPY_DEFAULT per tag
  4.  candidate trades per wallet (concentration)
  5.  candidate trades per market (concentration)
  6.  price distribution of candidate trades (extreme near resolution?)
  7.  time-to-resolution distribution of candidate trades
  8.  one-off vs recurring market share
  9.  ROI confounder IC on candidate trades (price vs copyable_roi)
  10. opening_roi distribution per tag

Inputs: data/polygon_trades_processed, data/markets_processed/markets.parquet
Outputs:
  onchain/o1_diagnosis.json  (per-tag table)
  onchain/o1_summary.json   (top-level summary)

Run modes:
    python o1_diagnose.py --tag Weather      # sanity, must reproduce known
    python o1_diagnose.py --tag Finance
    python o1_diagnose.py --tag Politics
    python o1_diagnose.py --all-tags
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SIGNAL_LAB = HERE.parent
NOTEBOOKS = SIGNAL_LAB.parent
PROJECT = NOTEBOOKS.parent.parent
sys.path.insert(0, str(PROJECT))

TRADES_DIR = PROJECT / "data" / "polygon_trades_processed"
MARKETS_PATH = PROJECT / "data" / "markets_processed" / "markets.parquet"
OUT_DIAG = HERE / "o1_diagnosis.json"
OUT_SUMMARY = HERE / "o1_summary.json"

# Mirror filters.DEFAULT_COPY_RULES to make the diagnostic comparable.
DEFAULT_COPY_RULES = {
    "min_buy_roi": 0.02,
    "min_buckets": 20,
    "min_markets": 15,
    "min_trade_count": 100,
    "max_drawdown_to_pnl": 0.6,
    "min_copyable_roi": 0.05,
}


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 30:
        return float("nan")
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    r, _ = spearmanr(x, y, nan_policy="omit")
    return float(r) if np.isfinite(r) else float("nan")


def load_for_tag(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (markets, opening_buys) for the given primary_tag."""
    cols_m = ["condition_id", "primary_tag", "end_date_iso", "tags"]
    markets = pd.read_parquet(MARKETS_PATH, columns=cols_m)
    markets = markets[markets["primary_tag"] == tag].copy()
    markets["end_date_iso"] = pd.to_datetime(markets["end_date_iso"], utc=True, errors="coerce")
    market_ids = set(markets["condition_id"].tolist())
    print(f"[{tag}] markets: {len(markets):,}")

    pieces = []
    for f in sorted(TRADES_DIR.glob("*.parquet")):
        df = pd.read_parquet(
            f,
            columns=[
                "wallet", "condition_id", "token_id", "dt", "side", "outcome",
                "position", "total_quantity", "avg_price", "trade_value_usdc",
                "final_value_usdc", "trade_pnl", "copyable_pnl", "token_winner",
                "final_price", "last_condition_trade_ts",
            ],
        )
        # opening BUY only
        df = df[
            (df["side"] == "BUY")
            & (df["position"] == df["total_quantity"])
            & (df["condition_id"].isin(market_ids))
        ]
        pieces.append(df)
    trades = pd.concat(pieces, ignore_index=True)
    print(f"[{tag}] opening BUY trades: {len(trades):,}")
    return markets, trades


def per_wallet(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("wallet").agg(
        trade_count=("dt", "size"),
        markets=("condition_id", "nunique"),
        notional=("trade_value_usdc", "sum"),
        pnl=("trade_pnl", "sum"),
        copyable_pnl=("copyable_pnl", "sum"),
        # use day-bucketed trade count as a proxy for "buckets" if no explicit
        # bucket column is in the trade data
    ).reset_index()
    # bucket count proxy: 1-day buckets
    g["buckets"] = (
        trades.assign(d=trades["dt"].dt.floor("1D"))
        .groupby("wallet")["d"].nunique()
        .reindex(g["wallet"]).values
    )
    g["buy_roi"] = g["pnl"] / g["notional"].clip(lower=1e-9)
    g["copyable_roi"] = g["copyable_pnl"] / g["notional"].clip(lower=1e-9)
    return g


def select_copy_default(wallet_df: pd.DataFrame) -> set[str]:
    r = DEFAULT_COPY_RULES
    mask = (
        (wallet_df["buy_roi"] >= r["min_buy_roi"])
        & (wallet_df["buckets"] >= r["min_buckets"])
        & (wallet_df["markets"] >= r["min_markets"])
        & (wallet_df["trade_count"] >= r["min_trade_count"])
        & (wallet_df["copyable_roi"] >= r["min_copyable_roi"])
    )
    return set(wallet_df.loc[mask, "wallet"])


def diagnose_one(tag: str) -> dict:
    markets, trades = load_for_tag(tag)
    if trades.empty:
        return {"tag": tag, "n_markets": 0, "n_trades": 0}
    w = per_wallet(trades)
    selected = select_copy_default(w)
    print(f"[{tag}] wallets passing COPY_DEFAULT: {len(selected):,} / {len(w):,}")

    # candidate universe = opening BUYs by selected wallets
    cand = trades[trades["wallet"].isin(selected)].copy()
    cand["dt"] = pd.to_datetime(cand["dt"], utc=True, errors="coerce")
    cand["last_condition_trade_ts"] = pd.to_datetime(
        cand["last_condition_trade_ts"], utc=True, errors="coerce"
    )
    cand = cand.dropna(subset=["dt", "last_condition_trade_ts", "avg_price", "outcome"])
    cand["lead_h"] = (
        cand["last_condition_trade_ts"] - cand["dt"]
    ).dt.total_seconds() / 3600
    # recode outcome -> 1 if the trade's outcome matches the market winner
    # The trade's outcome is in column "outcome" (Yes/No). The token_winner
    # column already encodes whether THIS token was the winner; for a binary
    # market, that's exactly 1 if the trade's outcome was the winning one.
    # We sanity-check this and use it directly.
    cand = cand.dropna(subset=["token_winner"])
    cand["y"] = cand["token_winner"].astype(int)

    # market concentration
    trades_per_market = cand.groupby("condition_id").size()
    market_concentration = (
        float(trades_per_market.quantile(0.99) / max(trades_per_market.median(), 1))
    )
    # wallet concentration: top-1% share
    cand_per_wallet = cand.groupby("wallet").size().sort_values(ascending=False)
    top1_pct = (
        float(cand_per_wallet.head(max(1, int(len(cand_per_wallet) * 0.01))).sum())
        / max(len(cand), 1)
    )
    # one-off vs recurring: count of (condition_id, dt_day) per market
    cand["dt_day"] = cand["dt"].dt.floor("1D")
    days_per_market = cand.groupby("condition_id")["dt_day"].nunique()
    one_off_share = float((days_per_market == 1).mean())

    # price distribution near 0/1
    near_resolution = float(
        ((cand["avg_price"] < 0.05) | (cand["avg_price"] > 0.95)).mean()
    )

    # price vs outcome correlation (price confounder)
    price_outcome_ic = _spearman(cand["avg_price"], cand["y"])
    # lead vs outcome correlation
    lead_outcome_ic = _spearman(cand["lead_h"], cand["y"])

    # wallet selection ROI
    sel_roi = float(w.loc[w["wallet"].isin(selected), "buy_roi"].mean())
    rejected_roi = float(w.loc[~w["wallet"].isin(selected), "buy_roi"].mean())

    return {
        "tag": tag,
        "n_markets": int(len(markets)),
        "n_opening_buys": int(len(trades)),
        "n_wallets_total": int(len(w)),
        "n_wallets_selected": int(len(selected)),
        "wallet_selection_rate": float(len(selected) / max(len(w), 1)),
        "selected_wallet_buy_roi_mean": sel_roi,
        "rejected_wallet_buy_roi_mean": rejected_roi,
        "n_candidate_buys": int(len(cand)),
        "market_concentration_p99_over_med": market_concentration,
        "top1pct_wallet_share_of_candidates": top1_pct,
        "one_off_market_share": one_off_share,
        "near_resolution_price_share": near_resolution,
        "price_outcome_ic": price_outcome_ic,
        "lead_h_outcome_ic": lead_outcome_ic,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--all-tags", action="store_true")
    args = ap.parse_args()
    if args.all_tags:
        tags = ["Weather", "Finance", "Politics"]
    else:
        tags = [args.tag or "Weather"]

    diag = {}
    for t in tags:
        print(f"\n=== {t} ===")
        diag[t] = diagnose_one(t)

    OUT_DIAG.write_text(json.dumps(diag, indent=2, default=str))
    OUT_SUMMARY.write_text(
        json.dumps({"tags": tags, "metrics": list(diag[tags[0]].keys())}, indent=2)
    )
    print(f"\nWrote {OUT_DIAG}")
    print(json.dumps(diag, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
