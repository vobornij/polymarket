"""Diagnostic script to understand why the approach fails and test improvements."""
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

print(f"\nWallets with metrics: {len(wallet_vol)}")

# Default leaders for diagnostics
bl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="BUY")
sl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="SELL")
bl_ws = set(bl["wallet"])
sl_ws = set(sl["wallet"])

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Basic selection - what happens at different thresholds
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 1: Varying follower selection thresholds")
print("=" * 80)

for min_roi in [0.0, 0.05, 0.10, 0.20, 0.30]:
    for min_buckets in [10, 30, 50]:
        fw = select_follower_wallets(
            wallet_vol,
            min_copyable_roi=min_roi,
            min_trade_value=100,
            min_num_buckets=min_buckets,
            max_market_pnl_hhi=0.3,
        )
        fw_ws = set(fw["wallet"])

        buy_ev = evaluate_implied_pnl(df_test, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")
        sell_ev = evaluate_implied_pnl(df_test, fw_ws, sl_ws, time_window_minutes=10, leader_side="SELL")
        total_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
        total_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
        roi = total_pnl / total_not if total_not > 0 else 0
        print(f"  followers(min_roi={min_roi:.2f}, min_buckets={min_buckets:2d}): "
              f"n={len(fw):5d}  test_pnl=${total_pnl:>10,.2f}  test_roi={roi:.4f}  "
              f"trades={buy_ev['trade_count']+sell_ev['trade_count']:6d}")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Leader quality analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 2: Leader quality - train vs test persistence")
print("=" * 80)

# How many leaders are active on train vs test?
train_buy_leaders_active = set(df_train[(df_train["wallet"].isin(bl_ws)) & (df_train["side"] == "BUY")]["wallet"].unique())
test_buy_leaders_active = set(df_test[(df_test["wallet"].isin(bl_ws)) & (df_test["side"] == "BUY")]["wallet"].unique())
print(f"Buy leaders active on train: {len(train_buy_leaders_active)}")
print(f"Buy leaders active on test:  {len(test_buy_leaders_active)}")
print(f"Overlap: {len(train_buy_leaders_active & test_buy_leaders_active)}")

# Look at per-leader implied PnL on train vs test
fw = select_follower_wallets(wallet_vol, min_copyable_roi=0.05, min_trade_value=100, min_num_buckets=10, max_market_pnl_hhi=0.3)
fw_ws = set(fw["wallet"])

buy_imp_train = detect_implied_buys(df_train, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")
buy_imp_test = detect_implied_buys(df_test, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")

if not buy_imp_train.empty and not buy_imp_test.empty:
    train_leader_pnl = buy_imp_train.groupby("leader_wallet")["copyable_pnl"].agg(["sum", "count"]).rename(columns={"sum": "train_pnl", "count": "train_trades"})
    test_leader_pnl = buy_imp_test.groupby("leader_wallet")["copyable_pnl"].agg(["sum", "count"]).rename(columns={"sum": "test_pnl", "count": "test_trades"})
    
    comparison = train_leader_pnl.join(test_leader_pnl, how="inner")
    comparison = comparison.sort_values("train_pnl", ascending=False)
    
    print(f"\nLeaders active in both train and test implied: {len(comparison)}")
    print(f"Leaders with positive train PnL AND positive test PnL: {((comparison['train_pnl'] > 0) & (comparison['test_pnl'] > 0)).sum()}")
    print(f"Leaders with positive train PnL AND negative test PnL: {((comparison['train_pnl'] > 0) & (comparison['test_pnl'] < 0)).sum()}")
    
    print("\nTop 20 train leaders - how they do on test:")
    print(comparison.head(20).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Leader stability - how consistent are leaders across time?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 3: Leader stability across train halves")
print("=" * 80)

mid_train = df_train["dt"].quantile(0.5)
df_train_early = df_train[df_train["dt"] <= mid_train].copy()
df_train_late = df_train[df_train["dt"] > mid_train].copy()
print(f"Train early: {len(df_train_early):,} trades  < {mid_train.date()}")
print(f"Train late:  {len(df_train_late):,} trades  >= {mid_train.date()}")

buy_imp_early = detect_implied_buys(df_train_early, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")
buy_imp_late = detect_implied_buys(df_train_late, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")
buy_imp_val = detect_implied_buys(df_val, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")
buy_imp_test2 = detect_implied_buys(df_test, fw_ws, bl_ws, time_window_minutes=10, leader_side="BUY")

def _leader_pnl(imp, label):
    if imp.empty:
        return pd.DataFrame()
    return imp.groupby("leader_wallet")["copyable_pnl"].agg(["sum", "count"]).rename(columns={"sum": f"{label}_pnl", "count": f"{label}_trades"})

lp_early = _leader_pnl(buy_imp_early, "early")
lp_late = _leader_pnl(buy_imp_late, "late")
lp_val = _leader_pnl(buy_imp_val, "val")
lp_test = _leader_pnl(buy_imp_test2, "test")

stab = lp_early.join(lp_late, how="inner").join(lp_val, how="outer").join(lp_test, how="outer").fillna(0)
stab = stab.sort_values("early_pnl", ascending=False)

print(f"\nLeaders with implied in early train: {len(lp_early)}")
print(f"Leaders with implied in late train:  {len(lp_late)}")
print(f"Overlap: {len(set(lp_early.index) & set(lp_late.index))}")

# How many early leaders are still profitable in late?
stab_both = stab[(stab["early_pnl"] > 0) & (stab["late_pnl"] > 0)]
print(f"Leaders profitable in BOTH halves: {len(stab_both)} / {len(stab[(stab['early_pnl'] > 0)])} profitable in early")

stab_all = stab[(stab["early_pnl"] > 0) & (stab["late_pnl"] > 0) & (stab["val_pnl"] > 0)]
print(f"Leaders profitable in ALL 3 periods: {len(stab_all)}")

print("\nTop 20 leaders - consistency across periods:")
print(stab.head(20).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Test with stricter leader selection
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 4: Stricter leader selection (stability + ROI thresholds)")
print("=" * 80)

# Find stable leaders: profitable in both train halves AND on val
stable_leaders = set(stab_all.index)
print(f"Stable buy leaders (profitable early+late+val): {len(stable_leaders)}")

# Also try: leaders with high min trade count
for min_tc in [20, 50, 100, 200]:
    for min_roi in [0.0, 0.05, 0.10]:
        strict_bl = select_leader_wallets(wallet_vol, min_trade_count=min_tc, min_roi=min_roi, max_market_pnl_hhi=0.5, side="BUY")
        strict_sl = select_leader_wallets(wallet_vol, min_trade_count=min_tc, min_roi=min_roi, max_market_pnl_hhi=0.5, side="SELL")
        sbl_ws = set(strict_bl["wallet"])
        ssl_ws = set(strict_sl["wallet"])
        
        buy_ev = evaluate_implied_pnl(df_test, fw_ws, sbl_ws, time_window_minutes=10, leader_side="BUY")
        sell_ev = evaluate_implied_pnl(df_test, fw_ws, ssl_ws, time_window_minutes=10, leader_side="SELL")
        total_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
        total_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
        roi = total_pnl / total_not if total_not > 0 else 0
        print(f"  leaders(tc>={min_tc}, roi>={min_roi:.2f}, hhi<=0.5): "
              f"n_buy={len(sbl_ws):5d} n_sell={len(ssl_ws):5d}  "
              f"test_pnl=${total_pnl:>10,.2f}  test_roi={roi:.4f}")

# Test with stable leaders specifically
buy_ev = evaluate_implied_pnl(df_test, fw_ws, stable_leaders, time_window_minutes=10, leader_side="BUY")
sell_ev = evaluate_implied_pnl(df_test, fw_ws, stable_leaders, time_window_minutes=10, leader_side="SELL")
total_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
total_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
roi = total_pnl / total_not if total_not > 0 else 0
print(f"\n  STABLE leaders only: test_pnl=${total_pnl:>10,.2f}  test_roi={roi:.4f}  trades={buy_ev['trade_count']+sell_ev['trade_count']}")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: Time window sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 5: Time window sensitivity")
print("=" * 80)

for tw in [5, 10, 15, 30, 60]:
    buy_ev = evaluate_implied_pnl(df_test, fw_ws, bl_ws, time_window_minutes=tw, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_test, fw_ws, sl_ws, time_window_minutes=tw, leader_side="SELL")
    total_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    total_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
    roi = total_pnl / total_not if total_not > 0 else 0
    print(f"  tw={tw:3d}min: test_pnl=${total_pnl:>10,.2f}  test_roi={roi:.4f}  trades={buy_ev['trade_count']+sell_ev['trade_count']:6d}")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 6: Combined strict leaders + stable + tighter followers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 6: Combined improvements")
print("=" * 80)

# Use stable leaders + strict followers + iterative refinement
strict_bl = select_leader_wallets(wallet_vol, min_trade_count=100, min_roi=0.05, max_market_pnl_hhi=0.5, side="BUY")
strict_sl = select_leader_wallets(wallet_vol, min_trade_count=100, min_roi=0.05, max_market_pnl_hhi=0.5, side="SELL")

# Also require leaders to be in the stable set
strict_bl_stable = strict_bl[strict_bl["wallet"].isin(stable_leaders)]
strict_sl_stable = strict_sl[strict_sl["wallet"].isin(stable_leaders)]

sbl_ws = set(strict_bl_stable["wallet"])
ssl_ws = set(strict_sl_stable["wallet"])
print(f"Strict stable buy leaders: {len(sbl_ws)}")
print(f"Strict stable sell leaders: {len(ssl_ws)}")

# Tighter followers
tight_fw = select_follower_wallets(wallet_vol, min_copyable_roi=0.10, min_trade_value=200, min_num_buckets=30, max_market_pnl_hhi=0.2)
tight_fw_ws = set(tight_fw["wallet"])
print(f"Tight followers: {len(tight_fw_ws)}")

# Baseline with strict
buy_ev = evaluate_implied_pnl(df_test, tight_fw_ws, sbl_ws, time_window_minutes=10, leader_side="BUY")
sell_ev = evaluate_implied_pnl(df_test, tight_fw_ws, ssl_ws, time_window_minutes=10, leader_side="SELL")
total_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
total_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
roi = total_pnl / total_not if total_not > 0 else 0
print(f"\nStrict stable + tight followers (no refinement):")
print(f"  test_pnl=${total_pnl:>10,.2f}  test_roi={roi:.4f}  trades={buy_ev['trade_count']+sell_ev['trade_count']}")

# With iterative refinement
fw_r, bl_r, sl_r, _, _ = iterative_leader_follower_filter(
    df_train, tight_fw_ws, sbl_ws, ssl_ws,
    time_window_minutes=10,
    n_iterations=3,
    leader_min_copyable_pnl=50,
    follower_min_copyable_pnl=50.0,
    follower_min_copyable_roi=0.05,
)
print(f"\nAfter iterative refinement:")
print(f"  followers={len(fw_r)}  buy_leaders={len(bl_r)}  sell_leaders={len(sl_r)}")

# Evaluate on all splits
for split_name, df_split in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
    buy_ev = evaluate_implied_pnl(df_split, fw_r, bl_r, time_window_minutes=10, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_split, fw_r, sl_r, time_window_minutes=10, leader_side="SELL")
    imp_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    imp_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
    roi = imp_pnl / imp_not if imp_not > 0 else 0
    print(f"  {split_name}: pnl=${imp_pnl:>10,.2f}  roi={roi:.4f}  trades={buy_ev['trade_count']+sell_ev['trade_count']:6d}")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC 7: Per-follower leader mapping (individual leader sets)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DIAGNOSTIC 7: Per-follower leader analysis (who follows whom?)")
print("=" * 80)

if not buy_imp_train.empty:
    pairs = buy_imp_train.groupby(["follower_wallet", "leader_wallet"]).agg(
        pnl=("copyable_pnl", "sum"),
        trades=("copyable_pnl", "count"),
    ).reset_index()
    
    pairs_test = buy_imp_test2.groupby(["follower_wallet", "leader_wallet"]).agg(
        pnl=("copyable_pnl", "sum"),
        trades=("copyable_pnl", "count"),
    ).reset_index() if not buy_imp_test2.empty else pd.DataFrame(columns=["follower_wallet", "leader_wallet", "pnl", "trades"])
    
    train_pair_set = set(zip(pairs["follower_wallet"], pairs["leader_wallet"]))
    test_pair_set = set(zip(pairs_test["follower_wallet"], pairs_test["leader_wallet"]))
    
    print(f"Train pairs (follower, leader): {len(train_pair_set)}")
    print(f"Test pairs: {len(test_pair_set)}")
    print(f"Overlap: {len(train_pair_set & test_pair_set)}")
    
    # Which pairs are profitable in both?
    train_profitable = set(zip(pairs.loc[pairs["pnl"] > 0, "follower_wallet"], pairs.loc[pairs["pnl"] > 0, "leader_wallet"]))
    test_profitable = set(zip(pairs_test.loc[pairs_test["pnl"] > 0, "follower_wallet"], pairs_test.loc[pairs_test["pnl"] > 0, "leader_wallet"]))
    print(f"Train profitable pairs: {len(train_profitable)}")
    print(f"Test profitable pairs: {len(test_profitable)}")
    print(f"Profitable in BOTH: {len(train_profitable & test_profitable)}")
    
    # For followers with multiple leaders on train, how many leaders persist?
    follower_leader_counts = pairs.groupby("follower_wallet")["leader_wallet"].nunique()
    print(f"\nFollowers with 1 leader: {(follower_leader_counts == 1).sum()}")
    print(f"Followers with 2-5 leaders: {((follower_leader_counts >= 2) & (follower_leader_counts <= 5)).sum()}")
    print(f"Followers with 5+ leaders: {(follower_leader_counts > 5).sum()}")
