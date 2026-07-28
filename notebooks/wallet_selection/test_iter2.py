"""Iteration 2: Tune pair thresholds, add concentration limits, drop refinement."""
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
    filter_stable_leaders,
    filter_pairs_by_frequency,
)
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

pd.options.display.float_format = "{:.4f}".format

# ─── Load data ───
df_full = load_trades()
df_full = compute_copyable_notional(df_full)

train_cutoff = pd.Timestamp("2026-06-01", tz="UTC")
val_cutoff = pd.Timestamp("2026-07-01", tz="UTC")

df_train = df_full[df_full["dt"] < train_cutoff].copy()
df_val = df_full[(df_full["dt"] >= train_cutoff) & (df_full["dt"] < val_cutoff)].copy()
df_test = df_full[df_full["dt"] >= val_cutoff].copy()

# ─── Wallet metrics ───
wallet_vol, _ = compute_wallet_metrics(df_train)
wallet_vol["copyable_pnl_factor"] = np.clip(
    wallet_vol["copyable_pnl"] / wallet_vol["total_pnl"].replace(0, np.nan), 0, 1.0
).fillna(0.0)
wallet_vol["copyable_roi"] = wallet_vol["average_roi"] * wallet_vol["copyable_pnl_factor"]
opening_metrics = compute_opening_metrics(df_train)
wallet_vol = wallet_vol.merge(opening_metrics, on="wallet", how="left")
for c in ["opening_roi", "opening_pnl", "opening_copyable_roi", "opening_copyable_pnl"]:
    wallet_vol[c] = wallet_vol[c].fillna(0.0)

# ─── Stable leaders (computed once) ───
bl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="BUY")
sl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="SELL")
bl_ws = set(bl["wallet"])
sl_ws = set(sl["wallet"])

# Need a follower set to compute stability
fw_base = set(select_follower_wallets(wallet_vol, min_copyable_roi=0.0, min_trade_value=100, min_num_buckets=10, max_market_pnl_hhi=1)["wallet"])

stable_bl_ws = filter_stable_leaders(df_train, bl_ws, fw_base, time_window_minutes=30, leader_side="BUY", n_splits=3, min_profitable_splits=2)
stable_sl_ws = filter_stable_leaders(df_train, sl_ws, fw_base, time_window_minutes=30, leader_side="SELL", n_splits=3, min_profitable_splits=2)
print(f"Stable buy leaders: {len(stable_bl_ws)}  Stable sell leaders: {len(stable_sl_ws)}")


def eval_test(label, fw_set, bl_set, sl_set, tw=30):
    buy_ev = evaluate_implied_pnl(df_test, fw_set, bl_set, time_window_minutes=tw, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_test, fw_set, sl_set, time_window_minutes=tw, leader_side="SELL")
    pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    notional = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
    roi = pnl / notional if notional > 0 else 0
    trades = buy_ev["trade_count"] + sell_ev["trade_count"]
    n_active = len(set(df_test[df_test["wallet"].isin(fw_set)]["wallet"]))
    print(f"  {label:70s}  pnl=${pnl:>10,.2f}  roi={roi:.4f}  trades={trades:6d}  fw={n_active:4d}")
    return pnl, roi, trades


def eval_val(label, fw_set, bl_set, sl_set, tw=30):
    buy_ev = evaluate_implied_pnl(df_val, fw_set, bl_set, time_window_minutes=tw, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_val, fw_set, sl_set, time_window_minutes=tw, leader_side="SELL")
    pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    notional = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
    roi = pnl / notional if notional > 0 else 0
    trades = buy_ev["trade_count"] + sell_ev["trade_count"]
    return pnl, roi, trades


def concentration_stats(imp_test, label):
    if imp_test.empty:
        return
    total_pnl = imp_test["copyable_pnl"].sum()
    fw_pnl = imp_test.groupby("follower_wallet")["copyable_pnl"].sum().sort_values(ascending=False)
    bl_pnl = imp_test.groupby("leader_wallet")["copyable_pnl"].sum().sort_values(ascending=False)
    pos_fw = (fw_pnl > 0).sum()
    neg_fw = (fw_pnl < 0).sum()
    top1_fw = fw_pnl.iloc[0] if len(fw_pnl) > 0 else 0
    top5_fw = fw_pnl.head(5).sum() if len(fw_pnl) >= 5 else fw_pnl.sum()
    top1_bl = bl_pnl.iloc[0] if len(bl_pnl) > 0 else 0
    print(f"    {label}: total=${total_pnl:,.0f}  fw_pos={pos_fw} fw_neg={neg_fw}  "
          f"top1_fw={top1_fw/total_pnl:.1%}  top5_fw={top5_fw/total_pnl:.1%}  "
          f"top1_bl={top1_bl/total_pnl:.1%}")


def filter_by_leader_pnl_cap(implied, max_leader_pnl_fraction=0.30):
    """Cap each leader's contribution to max_leader_pnl_fraction of total."""
    if implied.empty:
        return implied
    total = implied["copyable_pnl"].sum()
    if total <= 0:
        return implied
    leader_pnl = implied.groupby("leader_wallet")["copyable_pnl"].sum()
    cap = total * max_leader_pnl_fraction
    capped_leaders = set(leader_pnl[leader_pnl > cap].index)
    if not capped_leaders:
        return implied
    # For capped leaders, keep only proportionally representative trades
    result_parts = []
    for leader, grp in implied.groupby("leader_wallet"):
        if leader in capped_leaders:
            leader_total = grp["copyable_pnl"].sum()
            keep_frac = cap / leader_total
            n_keep = max(1, int(len(grp) * keep_frac))
            result_parts.append(grp.nlargest(n_keep, "copyable_pnl"))
        else:
            result_parts.append(grp)
    return pd.concat(result_parts, ignore_index=True)


print("\n" + "=" * 80)
print("ITERATION 2: Tuning pair thresholds and concentration")
print("=" * 80)

# ─── Test pair observation thresholds ───
print("\n--- Pair observation thresholds (stable leaders, any fw) ---")
fw_any = set(select_follower_wallets(wallet_vol, min_copyable_roi=0.0, min_trade_value=100, min_num_buckets=10, max_market_pnl_hhi=1)["wallet"])

for min_obs in [1, 2, 3, 5, 8, 10]:
    buy_imp = detect_implied_buys(df_train, fw_any, stable_bl_ws, time_window_minutes=30, leader_side="BUY")
    sell_imp = detect_implied_buys(df_train, fw_any, stable_sl_ws, time_window_minutes=30, leader_side="SELL")

    buy_f = filter_pairs_by_frequency(buy_imp, min_observations=min_obs)
    sell_f = filter_pairs_by_frequency(sell_imp, min_observations=min_obs)

    bl_t = set(buy_f["leader_wallet"]) if not buy_f.empty else set()
    sl_t = set(sell_f["leader_wallet"]) if not sell_f.empty else set()
    fw_t = (set(buy_f["follower_wallet"]) if not buy_f.empty else set()) | (set(sell_f["follower_wallet"]) if not sell_f.empty else set())

    test_pnl, test_roi, test_trades = eval_test(
        f"min_obs={min_obs:2d}", fw_t, bl_t, sl_t, tw=30
    )

# ─── Test follower quality thresholds with pair filtering ───
print("\n--- Follower quality + pair filter (min_obs=3) ---")
for min_roi in [0.0, 0.05, 0.10, 0.20, 0.30]:
    for min_buckets in [10, 30, 50]:
        fw = set(select_follower_wallets(
            wallet_vol, min_copyable_roi=min_roi, min_trade_value=100,
            min_num_buckets=min_buckets, max_market_pnl_hhi=0.3
        )["wallet"])

        buy_imp = detect_implied_buys(df_train, fw, stable_bl_ws, time_window_minutes=30, leader_side="BUY")
        sell_imp = detect_implied_buys(df_train, fw, stable_sl_ws, time_window_minutes=30, leader_side="SELL")
        buy_f = filter_pairs_by_frequency(buy_imp, min_observations=3)
        sell_f = filter_pairs_by_frequency(sell_imp, min_observations=3)

        bl_t = set(buy_f["leader_wallet"]) if not buy_f.empty else set()
        sl_t = set(sell_f["leader_wallet"]) if not sell_f.empty else set()
        fw_t = (set(buy_f["follower_wallet"]) if not buy_f.empty else set()) | (set(sell_f["follower_wallet"]) if not sell_f.empty else set())

        if fw_t:
            test_pnl, test_roi, test_trades = eval_test(
                f"fw_roi>={min_roi:.2f} bk>={min_buckets:2d}", fw_t, bl_t, sl_t, tw=30
            )

# ─── The best combo: high-quality fw + stable leaders + pair filter ───
print("\n" + "=" * 80)
print("REFINED BEST: HQ fw (roi>=0.20, bk>=30) + stable leaders + pair filter")
print("=" * 80)

hq_fw = set(select_follower_wallets(
    wallet_vol, min_copyable_roi=0.20, min_trade_value=100,
    min_num_buckets=30, max_market_pnl_hhi=0.3
)["wallet"])
print(f"HQ followers: {len(hq_fw)}")

for min_obs in [2, 3, 5]:
    for tw in [30, 60]:
        buy_imp = detect_implied_buys(df_train, hq_fw, stable_bl_ws, time_window_minutes=tw, leader_side="BUY")
        sell_imp = detect_implied_buys(df_train, hq_fw, stable_sl_ws, time_window_minutes=tw, leader_side="SELL")
        buy_f = filter_pairs_by_frequency(buy_imp, min_observations=min_obs)
        sell_f = filter_pairs_by_frequency(sell_imp, min_observations=min_obs)

        bl_t = set(buy_f["leader_wallet"]) if not buy_f.empty else set()
        sl_t = set(sell_f["leader_wallet"]) if not sell_f.empty else set()
        fw_t = (set(buy_f["follower_wallet"]) if not buy_f.empty else set()) | (set(sell_f["follower_wallet"]) if not sell_f.empty else set())

        val_pnl, val_roi, _ = eval_val(f"min_obs={min_obs} tw={tw}", fw_t, bl_t, sl_t, tw=tw)
        test_pnl, test_roi, test_trades = eval_test(f"min_obs={min_obs} tw={tw}", fw_t, bl_t, sl_t, tw=tw)

        if test_trades > 0:
            buy_imp_test = detect_implied_buys(df_test, fw_t, bl_t, time_window_minutes=tw, leader_side="BUY")
            sell_imp_test = detect_implied_buys(df_test, fw_t, sl_t, time_window_minutes=tw, leader_side="SELL")
            imp_test = pd.concat([buy_imp_test, sell_imp_test], ignore_index=True)
            concentration_stats(imp_test, f"  min_obs={min_obs} tw={tw}")

# ─── Deep dive on the best config ───
print("\n" + "=" * 80)
print("DEEP DIVE: min_obs=3, tw=30, fw_roi>=0.20, fw_bk>=30")
print("=" * 80)

buy_imp = detect_implied_buys(df_train, hq_fw, stable_bl_ws, time_window_minutes=30, leader_side="BUY")
sell_imp = detect_implied_buys(df_train, hq_fw, stable_sl_ws, time_window_minutes=30, leader_side="SELL")
buy_f = filter_pairs_by_frequency(buy_imp, min_observations=3)
sell_f = filter_pairs_by_frequency(sell_imp, min_observations=3)

bl_t = set(buy_f["leader_wallet"]) if not buy_f.empty else set()
sl_t = set(sell_f["leader_wallet"]) if not sell_f.empty else set()
fw_t = (set(buy_f["follower_wallet"]) if not buy_f.empty else set()) | (set(sell_f["follower_wallet"]) if not sell_f.empty else set())

print(f"Leaders: buy={len(bl_t)} sell={len(sl_t)}  Followers: {len(fw_t)}")

for split_name, df_split in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
    buy_ev = evaluate_implied_pnl(df_split, fw_t, bl_t, time_window_minutes=30, leader_side="BUY")
    sell_ev = evaluate_implied_pnl(df_split, fw_t, sl_t, time_window_minutes=30, leader_side="SELL")
    imp_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
    imp_not = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
    roi = imp_pnl / imp_not if imp_not > 0 else 0
    follower_buy = evaluate_follower_buy_performance(df_split, fw_t)
    n_active = len(set(df_split[df_split["wallet"].isin(fw_t)]["wallet"]))
    print(f"  {split_name}: pnl=${imp_pnl:>10,.2f}  roi={roi:.4f}  trades={buy_ev['trade_count']+sell_ev['trade_count']:6d}  "
          f"active_fw={n_active:4d}  leader_count: buy={buy_ev['leader_count']} sell={sell_ev['leader_count']}")

# Test concentration on this config
buy_imp_test = detect_implied_buys(df_test, fw_t, bl_t, time_window_minutes=30, leader_side="BUY")
sell_imp_test = detect_implied_buys(df_test, fw_t, sl_t, time_window_minutes=30, leader_side="SELL")
imp_test = pd.concat([buy_imp_test, sell_imp_test], ignore_index=True)

if not imp_test.empty:
    total_pnl = imp_test["copyable_pnl"].sum()
    total_trades = len(imp_test)
    print(f"\nTest concentration ({total_trades} trades, ${total_pnl:,.2f} total):")

    fw_pnl = imp_test.groupby("follower_wallet").agg(pnl=("copyable_pnl", "sum"), trades=("copyable_pnl", "count")).sort_values("pnl", ascending=False)
    bl_pnl = imp_test.groupby("leader_wallet").agg(pnl=("copyable_pnl", "sum"), trades=("copyable_pnl", "count"), n_fw=("follower_wallet", "nunique")).sort_values("pnl", ascending=False)
    mk_pnl = imp_test.groupby("condition_id").agg(pnl=("copyable_pnl", "sum"), trades=("copyable_pnl", "count")).sort_values("pnl", ascending=False)

    print(f"\n  Top 10 followers:")
    fw_pnl["cum"] = fw_pnl["pnl"].cumsum()
    fw_pnl["cum%"] = fw_pnl["cum"] / total_pnl
    print(fw_pnl.head(10).to_string())

    print(f"\n  Top 10 leaders:")
    bl_pnl["cum"] = bl_pnl["pnl"].cumsum()
    bl_pnl["cum%"] = bl_pnl["cum"] / total_pnl
    print(bl_pnl.head(10).to_string())

    print(f"\n  Top 10 markets:")
    mk_pnl["cum"] = mk_pnl["pnl"].cumsum()
    mk_pnl["cum%"] = mk_pnl["cum"] / total_pnl
    print(mk_pnl.head(10).to_string())

    pos = (fw_pnl["pnl"] > 0).sum()
    neg = (fw_pnl["pnl"] < 0).sum()
    print(f"\n  Followers: {pos} positive, {neg} negative")
    print(f"  Top 1 follower share: {fw_pnl.iloc[0]['pnl']/total_pnl:.1%}")
    print(f"  Top 5 followers share: {fw_pnl.head(5)['pnl'].sum()/total_pnl:.1%}")
