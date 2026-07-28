"""Run stage1 implied notebook cells as a script for iteration."""
import sys
sys.path.insert(0, "/Users/vobornij/projects/polymarket/src")

import numpy as np
import pandas as pd

from lib import (
    load_trades,
    split_data,
    compute_copyable_notional,
    compute_opening_metrics,
    select_follower_wallets,
    select_leader_wallets,
    detect_implied_buys,
    score_leaders,
    evaluate_implied_pnl,
    evaluate_follower_buy_performance,
    iterative_leader_follower_filter,
    run_implied_grid_search,
    DEFAULT_TAGS,
)
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

pd.options.display.float_format = "{:.4f}".format
pd.options.display.max_rows = 100

# --- Load data ---
df_full = load_trades()
df_full = compute_copyable_notional(df_full)

train_cutoff = pd.Timestamp("2026-06-01", tz="UTC")
val_cutoff = pd.Timestamp("2026-07-01", tz="UTC")

df_train = df_full[df_full["dt"] < train_cutoff].copy()
df_val = df_full[(df_full["dt"] >= train_cutoff) & (df_full["dt"] < val_cutoff)].copy()
df_test = df_full[df_full["dt"] >= val_cutoff].copy()

print(f"Split by trade date:")
print(f"  Train: {len(df_train):>10,} trades  ({df_train['condition_id'].nunique():>5,} markets)  < {train_cutoff.date()}")
print(f"  Val:   {len(df_val):>10,} trades  ({df_val['condition_id'].nunique():>5,} markets)  {train_cutoff.date()} .. {val_cutoff.date()}")
print(f"  Test:  {len(df_test):>10,} trades  ({df_test['condition_id'].nunique():>5,} markets)  >= {val_cutoff.date()}")
print(f"  Total: {len(df_full):>10,} trades  ({df_full['condition_id'].nunique():>5,} markets)")

train_markets = set(df_train["condition_id"].unique())
val_markets = set(df_val["condition_id"].unique())
test_markets = set(df_test["condition_id"].unique())
print(f"\n  Markets overlapping train/val: {len(train_markets & val_markets)}")
print(f"  Markets overlapping train/test: {len(train_markets & test_markets)}")
print(f"  Markets overlapping val/test: {len(val_markets & test_markets)}")

# --- Compute wallet metrics on training data ---
wallet_vol, _ = compute_wallet_metrics(df_train)

wallet_vol["copyable_pnl_factor"] = np.clip(
    wallet_vol["copyable_pnl"] / wallet_vol["total_pnl"].replace(0, np.nan),
    0, 1.0,
).fillna(0.0)
wallet_vol["copyable_roi"] = wallet_vol["average_roi"] * wallet_vol["copyable_pnl_factor"]

opening_metrics = compute_opening_metrics(df_train)
wallet_vol = wallet_vol.merge(opening_metrics, on="wallet", how="left")
for c in ["opening_roi", "opening_pnl", "opening_copyable_roi", "opening_copyable_pnl"]:
    wallet_vol[c] = wallet_vol[c].fillna(0.0)

print(f"\nWallets with metrics: {len(wallet_vol)}")

# --- Baseline selection ---
follower_wallets = select_follower_wallets(
    wallet_vol,
    min_copyable_roi=0.05,
    min_trade_value=100,
    min_num_buckets=10,
    max_market_pnl_hhi=0.3,
)
print(f"Followers: {len(follower_wallets)}")

buy_leaders = select_leader_wallets(
    wallet_vol,
    min_trade_count=20,
    min_roi=None,
    max_market_pnl_hhi=1,
    side="BUY",
)
print(f"Buy leaders: {len(buy_leaders)}")

sell_leaders = select_leader_wallets(
    wallet_vol,
    min_trade_count=20,
    min_roi=None,
    max_market_pnl_hhi=1,
    side="SELL",
)
print(f"Sell leaders: {len(sell_leaders)}")

# --- Baseline evaluation ---
follower_ws = set(follower_wallets['wallet'])
buy_leader_ws = set(buy_leaders['wallet'])
sell_leader_ws = set(sell_leaders['wallet'])

print("\n=== BASELINE (before grid search) ===")
for split_name, df_split in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
    buy_ev = evaluate_implied_pnl(df_split, follower_ws, buy_leader_ws, time_window_minutes=5, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_split, follower_ws, sell_leader_ws, time_window_minutes=5, leader_side="SELL")
    total = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    print(f"{split_name}: buy_pnl={buy_ev['followed_copyable_pnl']:.2f} ({buy_ev['trade_count']} trades, {buy_ev['leader_count']} leaders)  sell_pnl={sell_ev['followed_copyable_pnl']:.2f} ({sell_ev['trade_count']} trades, {sell_ev['leader_count']} leaders)  total={total:.2f}")

# --- Grid search ---
param_grid = dict(
    min_follower_copyable_roi=[0, 0.05],
    min_follower_trade_value=[100],
    min_follower_num_buckets=[10],
    max_follower_hhi=[1],
    min_buy_leader_trade_count=[20],
    min_buy_leader_roi=[0.0],
    max_buy_leader_hhi=[1],
    min_sell_leader_trade_count=[20],
    min_sell_leader_roi=[-1],
    max_sell_leader_hhi=[1],
    time_window_minutes=[10],
    min_pair_interactions=[0],
    n_iterations=[2],
    leader_min_copyable_pnl=[0],
    follower_min_copyable_pnl=[20.0],
    follower_min_copyable_roi=[0.05],
    follower_min_copyable_roi_cutoff=[0.1],
    follower_min_copyable_pnl_cutoff=[100.0],
    follower_max_market_hhi_cutoff=[None],
    follower_max_copyable_dd_ratio_cutoff=[None],
)

res_df = run_implied_grid_search(param_grid, wallet_vol, df_train, df_val)

best_row = res_df.iloc[0]
best_params = {k: best_row[k] for k in param_grid.keys()}

print(f"\nBest config (val implied_pnl={best_row['implied_copyable_pnl']:.2f}):")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"  followers={best_row['followers']:.0f}  buy_leaders={best_row['buy_leaders']:.0f}  sell_leaders={best_row['sell_leaders']:.0f}")
print(f"  buy_pnl={best_row['buy_pnl']:.2f}  sell_pnl={best_row['sell_pnl']:.2f}")

# --- Evaluate best config on all splits ---
best_followers = select_follower_wallets(
    wallet_vol,
    min_copyable_roi=best_params["min_follower_copyable_roi"],
    min_trade_value=best_params["min_follower_trade_value"],
    min_num_buckets=best_params["min_follower_num_buckets"],
    max_market_pnl_hhi=best_params["max_follower_hhi"],
)
best_buy_leaders = select_leader_wallets(
    wallet_vol,
    min_trade_count=best_params["min_buy_leader_trade_count"],
    min_roi=best_params["min_buy_leader_roi"],
    max_market_pnl_hhi=best_params["max_buy_leader_hhi"],
    side="BUY",
)
best_sell_leaders = select_leader_wallets(
    wallet_vol,
    min_trade_count=best_params["min_sell_leader_trade_count"],
    min_roi=best_params["min_sell_leader_roi"],
    max_market_pnl_hhi=best_params["max_sell_leader_hhi"],
    side="SELL",
)

b_fw = set(best_followers["wallet"])
b_blw = set(best_buy_leaders["wallet"])
b_slw = set(best_sell_leaders["wallet"])
tw = best_params["time_window_minutes"]

print(f"\nInitial: followers={len(b_fw)}  buy_leaders={len(b_blw)}  sell_leaders={len(b_slw)}")

n_iter = int(best_params.get("n_iterations", 0))
if n_iter > 0:
    leader_pnl = best_params.get("leader_min_copyable_pnl")
    follower_pnl = best_params.get("follower_min_copyable_pnl", 20.0)
    follower_roi = best_params.get("follower_min_copyable_roi")
    follower_mkt_hhi = best_params.get("follower_max_market_hhi")
    follower_dd_ratio = best_params.get("follower_max_copyable_dd_ratio")
    b_fw, b_blw, b_slw, ref_log, ref_snaps = iterative_leader_follower_filter(
        df_train, b_fw, b_blw, b_slw,
        time_window_minutes=tw,
        n_iterations=n_iter,
        leader_min_copyable_pnl=leader_pnl,
        follower_min_copyable_pnl=follower_pnl,
        follower_min_copyable_roi=follower_roi,
        follower_max_market_hhi=follower_mkt_hhi,
        follower_max_copyable_dd_ratio=follower_dd_ratio,
    )

follower_roi_cutoff = best_params.get("follower_min_copyable_roi_cutoff")
follower_pnl_thresh = best_params.get("follower_min_copyable_pnl_cutoff")
follower_mkt_hhi_thresh = best_params.get("follower_max_market_hhi_cutoff")
follower_dd_ratio_thresh = best_params.get("follower_max_copyable_dd_ratio_cutoff")

buy_imp = detect_implied_buys(df_val, b_fw, b_blw, time_window_minutes=tw, leader_side="BUY")
sell_imp = detect_implied_buys(df_val, b_fw, b_slw, time_window_minutes=tw, leader_side="SELL")
combined = pd.concat([buy_imp, sell_imp], ignore_index=True)
if not combined.empty:
    f_scores = combined.groupby("follower_wallet", sort=False).agg(
        total_copyable_pnl=("copyable_pnl", "sum"),
        total_copyable_notional=("copyable_notional", "sum"),
    ).reset_index()
    f_scores["copyable_roi"] = f_scores["total_copyable_pnl"] / f_scores["total_copyable_notional"].clip(lower=1e-9)
    mask = f_scores["total_copyable_pnl"] >= follower_pnl_thresh
    if follower_roi_cutoff is not None:
        mask = mask & (f_scores["copyable_roi"] >= follower_roi_cutoff)
    b_fw = b_fw & set(f_scores.loc[mask, "follower_wallet"])

print(f"Final: followers={len(b_fw)}  buy_leaders={len(b_blw)}  sell_leaders={len(b_slw)}")

if n_iter > 0:
    print("\nRefinement convergence (train set):")
    for i, (fw_i, bl_i, sl_i) in enumerate(ref_snaps):
        buy_ev = evaluate_implied_pnl(df_train, fw_i, bl_i, time_window_minutes=tw, leader_side="BUY")
        sell_ev = evaluate_implied_pnl(df_train, fw_i, sl_i, time_window_minutes=tw, leader_side="SELL")
        cpnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
        cn = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
        croi = cpnl / cn if cn > 0 else 0.0
        print(f"  iter {i}: copyable_pnl={cpnl:>10.2f}  ROI={croi:>7.4f}  followers={len(fw_i)}  buy_leaders={len(bl_i)}  sell_leaders={len(sl_i)}")

print("\n=== FINAL RESULTS ===")
for split_name, df_split in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
    buy_ev = evaluate_implied_pnl(df_split, b_fw, b_blw, time_window_minutes=tw, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_split, b_fw, b_slw, time_window_minutes=tw, leader_side="SELL")
    follower_buy = evaluate_follower_buy_performance(df_split, b_fw)

    n_active = len(set(df_split[df_split["wallet"].isin(b_fw)]["wallet"]))
    n_markets = df_split["condition_id"].nunique()

    imp_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    imp_notional = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
    imp_trades = buy_ev["trade_count"] + sell_ev["trade_count"]
    imp_roi = imp_pnl / imp_notional if imp_notional > 0 else 0.0

    print(f"{split_name} ({n_markets} markets, {n_active} active followers):")
    print(f"  Implied BUY:   Copyable PnL: {buy_ev['followed_copyable_pnl']:>10.2f}  ROI: {buy_ev['followed_copyable_pnl']/buy_ev['followed_copyable_notional'] if buy_ev['followed_copyable_notional']>0 else 0:>7.4f}  ({buy_ev['trade_count']} trades, {buy_ev['leader_count']} leaders)")
    print(f"  Implied SELL:  Copyable PnL: {sell_ev['followed_copyable_pnl']:>10.2f}  ROI: {sell_ev['followed_copyable_pnl']/sell_ev['followed_copyable_notional'] if sell_ev['followed_copyable_notional']>0 else 0:>7.4f}  ({sell_ev['trade_count']} trades, {sell_ev['leader_count']} leaders)")
    print(f"  Implied total: Copyable PnL: {imp_pnl:>10.2f}  ROI: {imp_roi:>7.4f}  ({imp_trades} trades)")
    print(f"  All buys:      Copyable PnL: {follower_buy['followed_copyable_pnl']:>10.2f}  ROI: {follower_buy['followed_copyable_roi']:>7.4f}  (wallet PnL: {follower_buy['wallet_pnl']:>10.2f}, {follower_buy['trade_count']} trades)")
    print()

# --- Concentration diagnostics (test split) ---
buy_imp_test = detect_implied_buys(df_test, b_fw, b_blw, time_window_minutes=tw, leader_side="BUY")
sell_imp_test = detect_implied_buys(df_test, b_fw, b_slw, time_window_minutes=tw, leader_side="SELL")
imp_test = pd.concat([buy_imp_test, sell_imp_test], ignore_index=True)

if imp_test.empty:
    print("No implied trades on test set.")
else:
    total_pnl = imp_test["copyable_pnl"].sum()
    total_trades = len(imp_test)
    print(f"\n=== CONCENTRATION DIAGNOSTICS (test) ===")
    print(f"Test implied trades: {total_trades:,}   Total copyable PnL: ${total_pnl:,.2f}\n")

    leader_pnl = (
        imp_test.groupby("leader_wallet", sort=False)["copyable_pnl"]
        .agg(["sum", "count", "nunique"])
        .rename(columns={"sum": "pnl", "count": "trades", "nunique": "followers"})
        .sort_values("pnl", ascending=False)
    )
    leader_pnl["cum_pnl"] = leader_pnl["pnl"].cumsum()
    leader_pnl["cum_pct"] = leader_pnl["cum_pnl"] / total_pnl
    n_leaders = len(leader_pnl)
    print(f"Leaders contributing to test PnL: {n_leaders}")
    for k in [1, 3, 5, 10]:
        if k <= n_leaders:
            pct = leader_pnl.iloc[k - 1]["cum_pct"]
            print(f"  Top {k:>2} leader(s): ${leader_pnl.iloc[k - 1]['cum_pnl']:>10,.2f}  ({pct:.1%} of total)")
    print()
    print("Top 10 leaders:")
    print(leader_pnl.head(10).to_string())

    follower_pnl = (
        imp_test.groupby("follower_wallet", sort=False)["copyable_pnl"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "pnl", "count": "trades"})
        .sort_values("pnl", ascending=False)
    )
    follower_pnl["cum_pnl"] = follower_pnl["pnl"].cumsum()
    follower_pnl["cum_pct"] = follower_pnl["cum_pnl"] / total_pnl
    n_followers = len(follower_pnl)
    print(f"\nFollowers active on test: {n_followers}")
    for k in [1, 5, 10, 20]:
        if k <= n_followers:
            pct = follower_pnl.iloc[k - 1]["cum_pct"]
            print(f"  Top {k:>2} follower(s): ${follower_pnl.iloc[k - 1]['cum_pnl']:>10,.2f}  ({pct:.1%} of total)")
    print()
    print("Top 10 followers:")
    print(follower_pnl.head(10).to_string())

    market_pnl = (
        imp_test.groupby("condition_id", sort=False)["copyable_pnl"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "pnl", "count": "trades"})
        .sort_values("pnl", ascending=False)
    )
    market_pnl["cum_pnl"] = market_pnl["pnl"].cumsum()
    market_pnl["cum_pct"] = market_pnl["cum_pnl"] / total_pnl
    n_markets = len(market_pnl)
    print(f"\nMarkets with implied trades: {n_markets}")
    for k in [1, 3, 5, 10]:
        if k <= n_markets:
            pct = market_pnl.iloc[k - 1]["cum_pct"]
            print(f"  Top {k:>2} market(s):  ${market_pnl.iloc[k - 1]['cum_pnl']:>10,.2f}  ({pct:.1%} of total)")
    print()
    print("Top 10 markets:")
    print(market_pnl.head(10).to_string())

    neg_followers = (follower_pnl["pnl"] < 0).sum()
    neg_pnl = follower_pnl.loc[follower_pnl["pnl"] < 0, "pnl"].sum()
    print(f"\nFollowers with negative PnL: {neg_followers}  (total: ${neg_pnl:,.2f})")
    pos_followers = (follower_pnl["pnl"] > 0).sum()
    pos_pnl = follower_pnl.loc[follower_pnl["pnl"] > 0, "pnl"].sum()
    print(f"Followers with positive PnL: {pos_followers}  (total: ${pos_pnl:,.2f})")

    vals = leader_pnl["pnl"].values
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    cum = np.cumsum(vals_sorted)
    gini = 1 - 2 * np.sum(cum) / (n * cum[-1]) if cum[-1] > 0 else 0.0
    print(f"\nLeader PnL Gini coefficient: {gini:.4f}  (1 = perfect concentration, 0 = equal)")
