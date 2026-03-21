import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


END_DATE_TRAIN = datetime.date(2026, 2, 10)
TRADES_DIR = Path("/Users/vobornij/projects/polymarket/data/polygon_trades_processed")
WORKSPACE_DIR = Path("/Users/vobornij/projects/polymarket/data/trade_signals_workspace")

WALLET_STATS_PATH = WORKSPACE_DIR / "wallet_stats.parquet"
SELECTED_WALLETS_PATH = WORKSPACE_DIR / "selected_wallets.parquet"
OPEN_BUYS_PATH = WORKSPACE_DIR / "open_buys_test.parquet"
EXEC_TAPE_PATH = WORKSPACE_DIR / "execution_tape_test.parquet"


def build_wallet_stats(dataset, batch_size=300_000):
    stats = defaultdict(
        lambda: {
            "train_trades": 0,
            "train_pnl": 0.0,
            "train_volume": 0.0,
            "train_wins": 0,
            "test_trades": 0,
            "test_pnl": 0.0,
            "test_volume": 0.0,
            "test_wins": 0,
        }
    )

    columns = ["wallet", "dt", "trade_pnl", "usdc_amount"]
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df = batch.to_pandas()
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        df["is_train"] = df["dt"].dt.date <= END_DATE_TRAIN
        grouped = (
            df.groupby(["wallet", "is_train"])
            .agg(
                trades=("trade_pnl", "size"),
                pnl=("trade_pnl", "sum"),
                volume=("usdc_amount", "sum"),
                wins=("trade_pnl", lambda s: (s > 0).sum()),
            )
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            prefix = "train" if row.is_train else "test"
            d = stats[row.wallet]
            d[f"{prefix}_trades"] += int(row.trades)
            d[f"{prefix}_pnl"] += float(row.pnl)
            d[f"{prefix}_volume"] += float(row.volume)
            d[f"{prefix}_wins"] += int(row.wins)

    wallet_df = pd.DataFrame(
        [
            {
                "wallet": wallet,
                **d,
                "train_hit_rate": d["train_wins"] / d["train_trades"] if d["train_trades"] else np.nan,
                "test_hit_rate": d["test_wins"] / d["test_trades"] if d["test_trades"] else np.nan,
                "train_avg_pnl": d["train_pnl"] / d["train_trades"] if d["train_trades"] else np.nan,
                "test_avg_pnl": d["test_pnl"] / d["test_trades"] if d["test_trades"] else np.nan,
            }
            for wallet, d in stats.items()
        ]
    )
    wallet_df["hit_rate_shrunk"] = (wallet_df["train_wins"] + 5.0) / (wallet_df["train_trades"] + 10.0)
    wallet_df["score"] = (
        np.clip(wallet_df["hit_rate_shrunk"] - 0.5, 0, None)
        * np.log1p(wallet_df["train_trades"])
        * np.log1p(wallet_df["train_volume"] / 1000.0)
    )
    return wallet_df


def select_wallets(wallet_stats, top_n=300):
    selected = wallet_stats[
        (wallet_stats["train_trades"] >= 50)
        & (wallet_stats["train_volume"] >= 1000)
        & (wallet_stats["train_pnl"] > 0)
    ].nlargest(top_n, "score")
    return selected.reset_index(drop=True)


def build_open_buys(dataset, selected_wallets, batch_size=300_000):
    selected_wallet_set = set(selected_wallets["wallet"])
    columns = [
        "wallet",
        "condition_id",
        "outcome",
        "dt",
        "side",
        "price",
        "quantity",
        "usdc_amount",
        "position",
        "trade_pnl",
        "token_winner",
    ]
    chunks = []
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df = batch.to_pandas()
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        split_mask = df["dt"].dt.date <= END_DATE_TRAIN
        df = df[(~split_mask) & (df["wallet"].isin(selected_wallet_set))]
        if df.empty:
            continue
        chunks.append(df)

    test_trades = pd.concat(chunks, ignore_index=True)
    test_trades = test_trades.sort_values(["dt", "wallet", "condition_id", "outcome"]).reset_index(drop=True)
    test_trades["market_key"] = test_trades["condition_id"] + "|" + test_trades["outcome"]
    test_trades["signed_quantity"] = np.where(test_trades["side"] == "BUY", test_trades["quantity"], -test_trades["quantity"])
    test_trades["prev_position"] = test_trades["position"] - test_trades["signed_quantity"]
    test_trades["position_change"] = test_trades["signed_quantity"]
    test_trades["event_type"] = np.where(
        (test_trades["side"] == "BUY") & (test_trades["prev_position"] <= 1e-9),
        "open_buy",
        np.where(
            (test_trades["side"] == "BUY") & (test_trades["position_change"] > 1e-9),
            "add_buy",
            np.where(
                (test_trades["side"] == "SELL") & (test_trades["position_change"] < -1e-9),
                "reduce_sell",
                "other",
            ),
        ),
    )

    open_buys = test_trades[test_trades["event_type"] == "open_buy"].copy()
    seen_by_market = defaultdict(lambda: defaultdict(set))
    same_prior = []
    opp_prior = []
    for row in open_buys[["condition_id", "outcome", "wallet"]].itertuples(index=False):
        same_prior.append(len(seen_by_market[row.condition_id][row.outcome] - {row.wallet}))
        opposite_wallets = set()
        for outcome_name, wallets in seen_by_market[row.condition_id].items():
            if outcome_name != row.outcome:
                opposite_wallets |= wallets
        opp_prior.append(len(opposite_wallets - {row.wallet}))
        seen_by_market[row.condition_id][row.outcome].add(row.wallet)

    open_buys["prior_same"] = same_prior
    open_buys["prior_opp"] = opp_prior
    open_buys["roi_if_copy"] = np.where(open_buys["token_winner"], 1.0 / open_buys["price"] - 1.0, -1.0)
    open_buys["price_bucket"] = pd.cut(
        open_buys["price"],
        bins=[0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0],
        include_lowest=True,
    ).astype(str)
    return open_buys.reset_index(drop=True)


def build_execution_tape(dataset, condition_ids, batch_size=300_000):
    condition_ids = set(condition_ids)
    columns = ["condition_id", "outcome", "dt", "price", "quantity", "side"]
    chunks = []
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df = batch.to_pandas()
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        split_mask = df["dt"].dt.date <= END_DATE_TRAIN
        df = df[(~split_mask) & (df["condition_id"].isin(condition_ids))]
        if df.empty:
            continue
        chunks.append(df[["condition_id", "outcome", "dt", "price", "quantity", "side"]])

    raw_tape = pd.concat(chunks, ignore_index=True)
    raw_tape = raw_tape.rename(columns={"dt": "tape_dt"})
    raw_tape["market_key"] = raw_tape["condition_id"] + "|" + raw_tape["outcome"]

    outcome_map_rows = []
    for condition_id, outcomes in raw_tape.groupby("condition_id")["outcome"].unique().items():
        outcomes = sorted(outcomes.tolist())
        if len(outcomes) == 2:
            outcome_map_rows.append({"condition_id": condition_id, "outcome": outcomes[0], "opposite_outcome": outcomes[1]})
            outcome_map_rows.append({"condition_id": condition_id, "outcome": outcomes[1], "opposite_outcome": outcomes[0]})

    # For a long entry in a token, only BUY prints represent directly executable
    # buy-side liquidity at that traded price.
    same_side = raw_tape[raw_tape["side"] == "BUY"][["market_key", "tape_dt", "price", "quantity"]].copy()
    same_side = same_side.rename(columns={"price": "exec_price_raw"})
    same_side["exec_source"] = "same_token"
    same_side["available_qty"] = same_side["quantity"].astype(float)
    same_side["available_usdc"] = same_side["available_qty"] * same_side["exec_price_raw"]
    same_side = same_side[["market_key", "tape_dt", "exec_price_raw", "exec_source", "available_qty", "available_usdc"]]

    if outcome_map_rows:
        outcome_map = pd.DataFrame(outcome_map_rows)
        # Synthetic long(target) via short(opposite) is only executable on
        # opposite-token SELL prints, because those represent sell-side execution.
        opposite_side = raw_tape[raw_tape["side"] == "SELL"].merge(
            outcome_map, on=["condition_id", "outcome"], how="inner"
        )
        opposite_side["market_key"] = opposite_side["condition_id"] + "|" + opposite_side["opposite_outcome"]
        opposite_side["exec_price_raw"] = 1.0 - opposite_side["price"]
        opposite_side["exec_source"] = "opposite_token"
        opposite_side["available_qty"] = opposite_side["quantity"].astype(float)
        opposite_side["available_usdc"] = opposite_side["available_qty"] * opposite_side["exec_price_raw"]
        opposite_side = opposite_side[["market_key", "tape_dt", "exec_price_raw", "exec_source", "available_qty", "available_usdc"]]
    else:
        opposite_side = pd.DataFrame(columns=["market_key", "tape_dt", "exec_price_raw", "exec_source", "available_qty", "available_usdc"])

    execution_tape = pd.concat([same_side, opposite_side], ignore_index=True)
    execution_tape = (
        execution_tape.groupby(["market_key", "tape_dt", "exec_price_raw", "exec_source"], as_index=False)
        .agg(available_qty=("available_qty", "sum"), available_usdc=("available_usdc", "sum"))
        .sort_values(["market_key", "tape_dt"])
        .reset_index(drop=True)
    )
    return execution_tape


def main():
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(TRADES_DIR, format="parquet")

    print("Building wallet stats...")
    wallet_stats = build_wallet_stats(dataset)
    wallet_stats.to_parquet(WALLET_STATS_PATH, index=False)
    print(f"Saved {len(wallet_stats):,} wallet stats -> {WALLET_STATS_PATH}")

    selected_wallets = select_wallets(wallet_stats)
    selected_wallets.to_parquet(SELECTED_WALLETS_PATH, index=False)
    print(f"Saved {len(selected_wallets):,} selected wallets -> {SELECTED_WALLETS_PATH}")

    print("Building test open-buy signals...")
    open_buys = build_open_buys(dataset, selected_wallets)
    open_buys.to_parquet(OPEN_BUYS_PATH, index=False)
    print(f"Saved {len(open_buys):,} open-buy rows -> {OPEN_BUYS_PATH}")

    print("Building test execution tape...")
    execution_tape = build_execution_tape(dataset, open_buys["condition_id"].unique())
    execution_tape.to_parquet(EXEC_TAPE_PATH, index=False)
    print(f"Saved {len(execution_tape):,} execution tape rows -> {EXEC_TAPE_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
