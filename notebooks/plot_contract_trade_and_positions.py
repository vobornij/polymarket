import datetime
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TRADES_DIR = Path("/Users/vobornij/projects/polymarket/data/polygon_trades_processed")
RAW_TRADES_DIR = Path("/Users/vobornij/projects/polymarket/data/trades_processed")
OUTPUT_HTML = Path("/Users/vobornij/projects/polymarket/notebooks/contract_trade_price_wallet_positions.html")
END_DATE_TRAIN = datetime.date(2026, 2, 10)


def pick_most_traded_condition(dataset, batch_size=400_000):
    counter = Counter()
    columns = ["condition_id", "dt"]
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df = batch.to_pandas()
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        df = df[df["dt"].dt.date > END_DATE_TRAIN]
        if df.empty:
            continue
        counter.update(df["condition_id"].tolist())

    if not counter:
        raise ValueError("No trades found in test period")
    condition_id, n = counter.most_common(1)[0]
    return condition_id, n


def load_condition_trades(dataset, condition_id, batch_size=400_000):
    columns = ["condition_id", "dt", "outcome", "wallet", "side", "price", "quantity", "usdc_amount"]
    chunks = []
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df = batch.to_pandas()
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        df = df[(df["condition_id"] == condition_id) & (df["dt"].dt.date > END_DATE_TRAIN)]
        if not df.empty:
            chunks.append(df)

    if not chunks:
        raise ValueError(f"No test-period trades found for {condition_id}")

    trades = pd.concat(chunks, ignore_index=True)
    trades = trades.sort_values("dt").reset_index(drop=True)
    return trades


def find_market_question(condition_id):
    pattern = f"**/{condition_id[:16]}*.jsonl"
    for fp in RAW_TRADES_DIR.glob(pattern):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("condition_id") == condition_id and row.get("title"):
                        return str(row["title"])
        except Exception:
            continue
    return "Unknown market question"


def build_aggregates(trades):
    trades = trades.copy()
    trades["signed_qty"] = np.where(trades["side"] == "BUY", trades["quantity"], -trades["quantity"])
    trades["abs_usdc"] = trades["usdc_amount"].abs()

    pos = (
        trades.groupby(["dt", "outcome"], as_index=False)
        .agg(signed_qty=("signed_qty", "sum"))
        .sort_values("dt")
    )
    pos["agg_position"] = pos.groupby("outcome", sort=False)["signed_qty"].cumsum()

    vol = trades.groupby("dt", as_index=False).agg(usdc_volume=("abs_usdc", "sum"))
    vol = vol.sort_values("dt").reset_index(drop=True)
    vol["cum_volume_usdc"] = vol["usdc_volume"].cumsum()

    wallet_first = trades.groupby("wallet", as_index=False)["dt"].min().sort_values("dt")
    wallet_first["cum_wallets"] = np.arange(1, len(wallet_first) + 1)

    wallet_curve = (
        vol[["dt"]]
        .merge(wallet_first[["dt", "cum_wallets"]], on="dt", how="left")
        .sort_values("dt")
        .reset_index(drop=True)
    )
    wallet_curve["cum_wallets"] = wallet_curve["cum_wallets"].ffill().fillna(0)

    return pos, vol, wallet_curve


def summarize_wallet_notional(trades, top_n=10):
    t = trades.copy()
    t["abs_usdc"] = t["usdc_amount"].abs()

    wallet_rank = (
        t.groupby("wallet", as_index=False)
        .agg(total_usdc=("abs_usdc", "sum"), trade_count=("wallet", "size"))
        .sort_values(["total_usdc", "trade_count"], ascending=False)
    )
    return wallet_rank.head(top_n)


def make_plot(trades, agg_positions, agg_volume, agg_wallets, condition_id, market_question, n_rows):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=(
            f"Trade Prices Over Time ({condition_id[:12]}...)",
            "Aggregate Position by Outcome",
            "Cumulative Volume and Wallet Count",
        ),
    )

    outcomes = sorted(trades["outcome"].dropna().astype(str).unique().tolist())
    for outcome in outcomes:
        sub = trades[trades["outcome"] == outcome]
        fig.add_trace(
            go.Scatter(
                x=sub["dt"],
                y=sub["price"],
                mode="markers",
                name=f"price:{outcome}",
                marker=dict(size=5, opacity=0.55),
                hovertemplate=(
                    "time=%{x|%Y-%m-%d %H:%M:%S}<br>"
                    "outcome=" + outcome + "<br>"
                    "price=%{y:.4f}<br>"
                    "side=%{customdata[0]}<br>"
                    "wallet=%{customdata[1]}<br>"
                    "qty=%{customdata[2]:.2f}<br>"
                    "usdc=%{customdata[3]:.2f}<extra></extra>"
                ),
                customdata=sub[["side", "wallet", "quantity", "usdc_amount"]].to_numpy(),
            ),
            row=1,
            col=1,
        )

    for outcome, sub in agg_positions.groupby("outcome", sort=False):
        fig.add_trace(
            go.Scatter(
                x=sub["dt"],
                y=sub["agg_position"],
                mode="lines",
                name=f"agg_pos:{outcome}",
                line=dict(width=2.0),
                hovertemplate=(
                    "time=%{x|%Y-%m-%d %H:%M:%S}<br>"
                    f"outcome={outcome}<br>"
                    "aggregate position=%{y:.2f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=agg_volume["dt"],
            y=agg_volume["cum_volume_usdc"],
            mode="lines",
            name="cum_volume_usdc",
            line=dict(width=2.2),
            hovertemplate="time=%{x|%Y-%m-%d %H:%M:%S}<br>cum volume=%{y:,.2f} USDC<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=agg_wallets["dt"],
            y=agg_wallets["cum_wallets"],
            mode="lines",
            name="cum_wallets",
            line=dict(width=2.2, dash="dot"),
            hovertemplate="time=%{x|%Y-%m-%d %H:%M:%S}<br>cum wallets=%{y:.0f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Most Traded Contract in Test Window (n={n_rows:,} trades)<br><sup>{market_question}</sup>",
        template="plotly_white",
        hovermode="x unified",
        height=1180,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=40, t=100, b=60),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Aggregate Position (shares)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Volume (USDC)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Wallets", row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    return fig


def main():
    dataset = ds.dataset(TRADES_DIR, format="parquet")
    condition_id, count = pick_most_traded_condition(dataset)
    trades = load_condition_trades(dataset, condition_id)
    market_question = find_market_question(condition_id)
    agg_positions, agg_volume, agg_wallets = build_aggregates(trades)
    wallet_rank_top = summarize_wallet_notional(trades, top_n=10)

    fig = make_plot(trades, agg_positions, agg_volume, agg_wallets, condition_id, market_question, len(trades))
    fig.write_html(OUTPUT_HTML, include_plotlyjs=True, auto_open=False)

    print(f"condition_id: {condition_id}")
    print(f"market_question: {market_question}")
    print(f"test_period_rows: {count:,}")
    print(f"outcomes: {sorted(trades['outcome'].dropna().unique().tolist())}")
    print("top wallets by notional:")
    print(wallet_rank_top.to_string(index=False))
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()
