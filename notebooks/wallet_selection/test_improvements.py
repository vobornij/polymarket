"""Test improved leader/follower selection with stability and pair filtering."""
import sys
sys.path.insert(0, "/Users/vobornij/projects/polymarket/src")

import numpy as np
import pandas as pd

from lib import (
    load_trades,
    compute_copyable_notional,
    compute_opening_metrics,
    select_follower_wallets,
    select_leader_wallets,
    detect_implied_buys,
    evaluate_implied_pnl,
    evaluate_follower_buy_performance,
    iterative_leader_follower_filter,
    filter_stable_leaders,
    filter_pairs_by_frequency,
)
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

pd.options.display.float_format = "{:.4f}".format
pd.options.display.max_rows = 100

# ─── Load data ───
df_full = load_trades()
df_full = compute_copyable_notional(df_full)

train_cutoff = pd.Timestamp("2026-06-01", tz="UTC")
val_cutoff = pd.Timestamp("2026-07-01", tz="UTC")

df_train = df_full[df_full["dt"] < train_cutoff].copy()
df_val = df_full[(df_full["dt"] >= train_cutoff) & (df_full["dt"] < val_cutoff)].copy()
df_test = df_full[df_full["dt"] >= val_cutoff].copy()

print(f"Train: {len(df_train):,} trades ({df_train['condition_id'].nunique()} markets)")
print(f"Val:   {len(df_val):,} trades ({df_val['condition_id'].nunique()} markets)")
print(f"Test:  {len(df_test):,} trades ({df_test['condition_id'].nunique()} markets)")

# ─── Wallet metrics on train ───
wallet_vol, _ = compute_wallet_metrics(df_train)
wallet_vol["copyable_pnl_factor"] = np.clip(
    wallet_vol["copyable_pnl"] / wallet_vol["total_pnl"].replace(0, np.nan), 0, 1.0
).fillna(0.0)
wallet_vol["copyable_roi"] = wallet_vol["average_roi"] * wallet_vol["copyable_pnl_factor"]
opening_metrics = compute_opening_metrics(df_train)
wallet_vol = wallet_vol.merge(opening_metrics, on="wallet", how="left")
for c in ["opening_roi", "opening_pnl", "opening_copyable_roi", "opening_copyable_pnl"]:
    wallet_vol[c] = wallet_vol[c].fillna(0.0)

# ─── Default selections ───
bl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="BUY")
sl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="SELL")
bl_ws = set(bl["wallet"])
sl_ws = set(sl["wallet"])

fw = select_follower_wallets(wallet_vol, min_copyable_roi=0.05, min_trade_value=100, min_num_buckets=10, max_market_pnl_hhi=0.3)
fw_ws = set(fw["wallet"])

print(f"\nDefault: followers={len(fw_ws)}  buy_leaders={len(bl_ws)}  sell_leaders={len(sl_ws)}")


def eval_all(label, fw_set, bl_set, sl_set, tw=30):
    """Evaluate on all splits and print results."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    for split_name, df_split in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
        buy_ev = evaluate_implied_pnl(df_split, fw_set, bl_set, time_window_minutes=tw, leader_side="BUY")
        sell_ev = evaluate_implied_pnl(df_split, fw_set, sl_set, time_window_minutes=tw, leader_side="SELL")
        imp_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
        imp_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
        roi = imp_pnl / imp_not if imp_not > 0 else 0
        n_active = len(set(df_split[df_split["wallet"].isin(fw_set)]["wallet"]))
        print(f"  {split_name}: pnl=${imp_pnl:>10,.2f}  roi={roi:.4f}  trades={buy_ev['trade_count']+sell_ev['trade_count']:6d}  active_fw={n_active:4d}")


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Baseline with longer time window
# ══════════════════════════════════════════════════════════════════════════════
eval_all("A1: Baseline (default fw/bl) + tw=30min", fw_ws, bl_ws, sl_ws, tw=30)

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: Stability-filtered leaders
# ══════════════════════════════════════════════════════════════════════════════
print("\nFiltering stable buy leaders...")
stable_bl_ws = filter_stable_leaders(
    df_train, bl_ws, fw_ws,
    time_window_minutes=30, leader_side="BUY", n_splits=3, min_profitable_splits=2,
)
print(f"  Stable buy leaders: {len(stable_bl_ws)} / {len(bl_ws)}")

print("Filtering stable sell leaders...")
stable_sl_ws = filter_stable_leaders(
    df_train, sl_ws, fw_ws,
    time_window_minutes=30, leader_side="SELL", n_splits=3, min_profitable_splits=2,
)
print(f"  Stable sell leaders: {len(stable_sl_ws)} / {len(sl_ws)}")

eval_all("A2: Stability-filtered leaders + tw=30min", fw_ws, stable_bl_ws, stable_sl_ws, tw=30)

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 3: Stability + pair frequency filtering
# ══════════════════════════════════════════════════════════════════════════════
# Detect implied on train with stable leaders, then filter pairs
buy_imp_train = detect_implied_buys(df_train, fw_ws, stable_bl_ws, time_window_minutes=30, leader_side="BUY")
sell_imp_train = detect_implied_buys(df_train, fw_ws, stable_sl_ws, time_window_minutes=30, leader_side="SELL")

print(f"\nTrain implied before pair filter: buy={len(buy_imp_train)}, sell={len(sell_imp_train)}")

# Get good pairs from training
buy_imp_filtered = filter_pairs_by_frequency(buy_imp_train, min_observations=3, min_total_pnl=0)
sell_imp_filtered = filter_pairs_by_frequency(sell_imp_train, min_observations=3, min_total_pnl=0)

print(f"Train implied after pair filter (min_obs=3): buy={len(buy_imp_filtered)}, sell={len(sell_imp_filtered)}")

# Extract filtered leader-follower pairs
if not buy_imp_filtered.empty:
    good_buy_pairs = buy_imp_filtered[["follower_wallet", "leader_wallet"]].drop_duplicates()
    filtered_bl_from_pairs = set(good_buy_pairs["leader_wallet"])
    filtered_fw_from_pairs = set(good_buy_pairs["follower_wallet"])
else:
    filtered_bl_from_pairs = set()
    filtered_fw_from_pairs = set()

if not sell_imp_filtered.empty:
    good_sell_pairs = sell_imp_filtered[["follower_wallet", "leader_wallet"]].drop_duplicates()
    filtered_sl_from_pairs = set(good_sell_pairs["leader_wallet"])
    filtered_fw_from_pairs = filtered_fw_from_pairs | set(good_sell_pairs["follower_wallet"])
else:
    filtered_sl_from_pairs = set()

print(f"Filtered buy leaders (from pairs): {len(filtered_bl_from_pairs)}")
print(f"Filtered sell leaders (from pairs): {len(filtered_sl_from_pairs)}")
print(f"Filtered followers (from pairs): {len(filtered_fw_from_pairs)}")

# For test evaluation, use the filtered leader sets but all original followers
# (since pair filtering on train tells us which leaders are reliable)
eval_all("A3: Pair-filtered leaders (min_obs=3) + stable + tw=30min",
         fw_ws, filtered_bl_from_pairs, filtered_sl_from_pairs, tw=30)

# Also test with filtered followers
eval_all("A3b: Pair-filtered leaders + pair-filtered followers + tw=30min",
         filtered_fw_from_pairs, filtered_bl_from_pairs, filtered_sl_from_pairs, tw=30)

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 4: High-quality followers + stable leaders
# ══════════════════════════════════════════════════════════════════════════════
hq_fw = select_follower_wallets(wallet_vol, min_copyable_roi=0.30, min_trade_value=200, min_num_buckets=30, max_market_pnl_hhi=0.2)
hq_fw_ws = set(hq_fw["wallet"])
print(f"\nHigh-quality followers: {len(hq_fw_ws)}")

eval_all("A4: HQ followers + stable leaders + tw=30min", hq_fw_ws, stable_bl_ws, stable_sl_ws, tw=30)

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 5: Combined best - HQ fw + stable leaders + pair filter + refinement
# ══════════════════════════════════════════════════════════════════════════════
# Start with HQ followers and stable leaders, do pair filtering on train
hq_buy_imp = detect_implied_buys(df_train, hq_fw_ws, stable_bl_ws, time_window_minutes=30, leader_side="BUY")
hq_sell_imp = detect_implied_buys(df_train, hq_fw_ws, stable_sl_ws, time_window_minutes=30, leader_side="SELL")

hq_buy_filtered = filter_pairs_by_frequency(hq_buy_imp, min_observations=2, min_total_pnl=0)
hq_sell_filtered = filter_pairs_by_frequency(hq_sell_imp, min_observations=2, min_total_pnl=0)

if not hq_buy_filtered.empty:
    a5_bl = set(hq_buy_filtered["leader_wallet"])
    a5_fw = set(hq_buy_filtered["follower_wallet"])
else:
    a5_bl = set()
    a5_fw = set()

if not hq_sell_filtered.empty:
    a5_sl = set(hq_sell_filtered["leader_wallet"])
    a5_fw = a5_fw | set(hq_sell_filtered["follower_wallet"])
else:
    a5_sl = set()

print(f"\nA5 pre-refinement: followers={len(a5_fw)}  buy_leaders={len(a5_bl)}  sell_leaders={len(a5_sl)}")

# Iterative refinement on train
if a5_fw and a5_bl:
    a5_fw, a5_bl, a5_sl, ref_log, ref_snaps = iterative_leader_follower_filter(
        df_train, a5_fw, a5_bl, a5_sl,
        time_window_minutes=30,
        n_iterations=3,
        leader_min_copyable_pnl=50,
        follower_min_copyable_pnl=50.0,
        follower_min_copyable_roi=0.05,
    )
    print(f"A5 post-refinement: followers={len(a5_fw)}  buy_leaders={len(a5_bl)}  sell_leaders={len(a5_sl)}")
    eval_all("A5: HQ fw + stable leaders + pair filter + refinement + tw=30min", a5_fw, a5_bl, a5_sl, tw=30)

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 6: Wider time window (60min) + stable leaders + HQ followers
# ══════════════════════════════════════════════════════════════════════════════
eval_all("A6: HQ followers + stable leaders + tw=60min", hq_fw_ws, stable_bl_ws, stable_sl_ws, tw=60)

# Pair filter with 60min window
hq_buy_imp60 = detect_implied_buys(df_train, hq_fw_ws, stable_bl_ws, time_window_minutes=60, leader_side="BUY")
hq_sell_imp60 = detect_implied_buys(df_train, hq_fw_ws, stable_sl_ws, time_window_minutes=60, leader_side="SELL")
hq_buy_f60 = filter_pairs_by_frequency(hq_buy_imp60, min_observations=2, min_total_pnl=0)
hq_sell_f60 = filter_pairs_by_frequency(hq_sell_imp60, min_observations=2, min_total_pnl=0)

if not hq_buy_f60.empty:
    a6_bl = set(hq_buy_f60["leader_wallet"])
    a6_fw = set(hq_buy_f60["follower_wallet"])
else:
    a6_bl = set()
    a6_fw = set()
if not hq_sell_f60.empty:
    a6_sl = set(hq_sell_f60["leader_wallet"])
    a6_fw = a6_fw | set(hq_sell_f60["follower_wallet"])
else:
    a6_sl = set()

print(f"\nA6 pre-refinement: followers={len(a6_fw)}  buy_leaders={len(a6_bl)}  sell_leaders={len(a6_sl)}")

if a6_fw and a6_bl:
    a6_fw, a6_bl, a6_sl, _, _ = iterative_leader_follower_filter(
        df_train, a6_fw, a6_bl, a6_sl,
        time_window_minutes=60,
        n_iterations=3,
        leader_min_copyable_pnl=50,
        follower_min_copyable_pnl=50.0,
        follower_min_copyable_roi=0.05,
    )
    print(f"A6 post-refinement: followers={len(a6_fw)}  buy_leaders={len(a6_bl)}  sell_leaders={len(a6_sl)}")
    eval_all("A6: HQ fw + stable leaders + pair filter + refinement + tw=60min", a6_fw, a6_bl, a6_sl, tw=60)

# ══════════════════════════════════════════════════════════════════════════════
# CONCENTRATION on best approach
# ══════════════════════════════════════════════════════════════════════════════
# Use A6 as the best candidate
if a6_fw and a6_bl:
    print("\n" + "=" * 70)
    print("CONCENTRATION DIAGNOSTICS (best approach - A6)")
    print("=" * 70)
    buy_imp_test = detect_implied_buys(df_test, a6_fw, a6_bl, time_window_minutes=60, leader_side="BUY")
    sell_imp_test = detect_implied_buys(df_test, a6_fw, a6_sl, time_window_minutes=60, leader_side="SELL")
    imp_test = pd.concat([buy_imp_test, sell_imp_test], ignore_index=True)

    if imp_test.empty:
        print("No implied trades on test set.")
    else:
        total_pnl = imp_test["copyable_pnl"].sum()
        total_trades = len(imp_test)
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
        print(f"Leaders: {n_leaders}")
        for k in [1, 3, 5]:
            if k <= n_leaders:
                pct = leader_pnl.iloc[k - 1]["cum_pct"]
                print(f"  Top {k:>2}: ${leader_pnl.iloc[k - 1]['cum_pnl']:>10,.2f}  ({pct:.1%})")
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
        print(f"\nFollowers: {n_followers}")
        for k in [1, 5, 10]:
            if k <= n_followers:
                pct = follower_pnl.iloc[k - 1]["cum_pct"]
                print(f"  Top {k:>2}: ${follower_pnl.iloc[k - 1]['cum_pnl']:>10,.2f}  ({pct:.1%})")
        print(follower_pnl.head(10).to_string())

        neg = (follower_pnl["pnl"] < 0).sum()
        pos = (follower_pnl["pnl"] > 0).sum()
        print(f"\nPositive followers: {pos}  Negative followers: {neg}")
