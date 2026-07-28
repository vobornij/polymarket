"""Iteration 3: Follower stability filter, concentration caps, final tuning."""
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

# ─── Stable leaders ───
bl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="BUY")
sl = select_leader_wallets(wallet_vol, min_trade_count=20, min_roi=None, max_market_pnl_hhi=1, side="SELL")
fw_base = set(select_follower_wallets(wallet_vol, min_copyable_roi=0.0, min_trade_value=100, min_num_buckets=10, max_market_pnl_hhi=1)["wallet"])

stable_bl_ws = filter_stable_leaders(df_train, set(bl["wallet"]), fw_base, time_window_minutes=30, leader_side="BUY", n_splits=3, min_profitable_splits=2)
stable_sl_ws = filter_stable_leaders(df_train, set(sl["wallet"]), fw_base, time_window_minutes=30, leader_side="SELL", n_splits=3, min_profitable_splits=2)
print(f"Stable leaders: buy={len(stable_bl_ws)} sell={len(stable_sl_ws)}")


def filter_stable_followers(df, fw_set, bl_set, sl_set, *, tw=30, n_splits=3, min_profitable_splits=2):
    """Keep followers with positive implied PnL in >= min_profitable_splits time slices."""
    dt_min, dt_max = df["dt"].min(), df["dt"].max()
    edges = pd.date_range(dt_min, dt_max, periods=n_splits + 1, tz=dt_min.tz)

    fw_scores = {w: 0 for w in fw_set}
    for i in range(n_splits):
        chunk = df[(df["dt"] >= edges[i]) & (df["dt"] < edges[i + 1])]
        if chunk.empty:
            continue
        buy_imp = detect_implied_buys(chunk, fw_set, bl_set, time_window_minutes=tw, leader_side="BUY")
        sell_imp = detect_implied_buys(chunk, fw_set, sl_set, time_window_minutes=tw, leader_side="SELL")
        combined = pd.concat([buy_imp, sell_imp], ignore_index=True)
        if combined.empty:
            continue
        fw_pnl = combined.groupby("follower_wallet")["copyable_pnl"].sum()
        for fw, pnl in fw_pnl.items():
            if fw in fw_scores and pnl > 0:
                fw_scores[fw] += 1

    return {w for w, s in fw_scores.items() if s >= min_profitable_splits}


def eval_full(label, fw_set, bl_set, sl_set, tw=30):
    """Evaluate on all splits."""
    results = {}
    for split_name, df_split in [("TRAIN", df_train), ("VAL", df_val), ("TEST", df_test)]:
        buy_ev = evaluate_implied_pnl(df_split, fw_set, bl_set, time_window_minutes=tw, leader_side="BUY")
        sell_ev = evaluate_implied_pnl(df_split, fw_set, sl_set, time_window_minutes=tw, leader_side="SELL")
        pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
        notional = buy_ev["followed_copyable_notional"] + sell_ev["followed_copyable_notional"]
        roi = pnl / notional if notional > 0 else 0
        trades = buy_ev["trade_count"] + sell_ev["trade_count"]
        n_active = len(set(df_split[df_split["wallet"].isin(fw_set)]["wallet"]))
        results[split_name] = {"pnl": pnl, "roi": roi, "trades": trades, "active_fw": n_active}

    t = results["TEST"]
    v = results["VAL"]
    print(f"  {label:70s}  VAL: roi={v['roi']:.4f}  TEST: pnl=${t['pnl']:>8,.0f}  roi={t['roi']:.4f}  trades={t['trades']:5d}  fw={t['active_fw']:3d}")
    return results


print("\n" + "=" * 80)
print("ITERATION 3: Follower stability + concentration analysis")
print("=" * 80)

# ─── Build the recommended pipeline: fw(roi>=0.30, bk>=30) + stable leaders + pair filter ───
hq_fw = set(select_follower_wallets(
    wallet_vol, min_copyable_roi=0.30, min_trade_value=100,
    min_num_buckets=30, max_market_pnl_hhi=0.3
)["wallet"])
print(f"\nHQ followers (roi>=0.30, bk>=30): {len(hq_fw)}")

buy_imp = detect_implied_buys(df_train, hq_fw, stable_bl_ws, time_window_minutes=30, leader_side="BUY")
sell_imp = detect_implied_buys(df_train, hq_fw, stable_sl_ws, time_window_minutes=30, leader_side="SELL")
buy_f = filter_pairs_by_frequency(buy_imp, min_observations=3)
sell_f = filter_pairs_by_frequency(sell_imp, min_observations=3)

bl_t = set(buy_f["leader_wallet"]) if not buy_f.empty else set()
sl_t = set(sell_f["leader_wallet"]) if not sell_f.empty else set()
fw_t = (set(buy_f["follower_wallet"]) if not buy_f.empty else set()) | (set(sell_f["follower_wallet"]) if not sell_f.empty else set())

print(f"After pair filter: fw={len(fw_t)}  bl={len(bl_t)}  sl={len(sl_t)}")

# ─── Test without follower stability ───
r_base = eval_full("BASE: fw(0.30)+stable_bl+pair(3)+tw30", fw_t, bl_t, sl_t, tw=30)

# ─── With follower stability ───
print("\nFiltering stable followers (train, 3 splits)...")
stable_fw = filter_stable_followers(df_train, fw_t, bl_t, sl_t, tw=30, n_splits=3, min_profitable_splits=2)
print(f"Stable followers: {len(stable_fw)} / {len(fw_t)}")

r_stab = eval_full("STABLE FW: fw(0.30)+stable_bl+pair(3)+stab_fw+tw30", stable_fw, bl_t, sl_t, tw=30)

# ─── Try min_profitable_splits=3 (must be profitable in ALL 3 periods) ───
very_stable_fw = filter_stable_followers(df_train, fw_t, bl_t, sl_t, tw=30, n_splits=3, min_profitable_splits=3)
print(f"\nVery stable followers (3/3): {len(very_stable_fw)}")
r_vstab = eval_full("V.STABLE FW (3/3)+fw(0.30)+stable_bl+pair(3)+tw30", very_stable_fw, bl_t, sl_t, tw=30)

# ─── With concentration cap: max 20% of total PnL per follower ───
print("\n" + "-" * 70)
print("CONCENTRATION ANALYSIS on best approach")
print("-" * 70)

imp_test = pd.concat([
    detect_implied_buys(df_test, fw_t, bl_t, time_window_minutes=30, leader_side="BUY"),
    detect_implied_buys(df_test, fw_t, sl_t, time_window_minutes=30, leader_side="SELL"),
], ignore_index=True)

if not imp_test.empty:
    total = imp_test["copyable_pnl"].sum()
    print(f"Total test PnL: ${total:,.2f}")

    fw_pnl = imp_test.groupby("follower_wallet")["copyable_pnl"].sum().sort_values(ascending=False)
    bl_pnl = imp_test.groupby("leader_wallet")["copyable_pnl"].sum().sort_values(ascending=False)
    mk_pnl = imp_test.groupby("condition_id")["copyable_pnl"].sum().sort_values(ascending=False)

    # What if we cap each follower to max 15% of total?
    cap_frac = 0.15
    cap = total * cap_frac
    capped_pnl = fw_pnl.clip(upper=cap).sum()
    print(f"\nWith per-follower cap at {cap_frac:.0%} of total: adjusted PnL=${capped_pnl:,.2f}")

    # What if we exclude the top 1 follower?
    excl_top1 = fw_pnl.iloc[1:].sum()
    print(f"Excluding top 1 follower: PnL=${excl_top1:,.2f}")

    # Excluding top 3 followers
    excl_top3 = fw_pnl.iloc[3:].sum()
    print(f"Excluding top 3 followers: PnL=${excl_top3:,.2f}")

    # What fraction of followers are positive?
    pos = (fw_pnl > 0).sum()
    neg = (fw_pnl < 0).sum()
    print(f"\nFollowers: {pos} positive, {neg} negative ({pos/(pos+neg):.0%} positive)")

    # What if we only keep followers positive on val too?
    imp_val = pd.concat([
        detect_implied_buys(df_val, fw_t, bl_t, time_window_minutes=30, leader_side="BUY"),
        detect_implied_buys(df_val, fw_t, sl_t, time_window_minutes=30, leader_side="SELL"),
    ], ignore_index=True)

    if not imp_val.empty:
        val_fw_pnl = imp_val.groupby("follower_wallet")["copyable_pnl"].sum()
        both_pos = set(fw_pnl[fw_pnl > 0].index) & set(val_fw_pnl[val_fw_pnl > 0].index)
        print(f"Followers positive on BOTH val and test: {len(both_pos)}")
        if both_pos:
            test_only_pos = fw_pnl.loc[fw_pnl.index.isin(both_pos)].sum()
            print(f"  Their test PnL: ${test_only_pos:,.2f}")

    # Same for leaders
    val_bl_pnl = imp_val.groupby("leader_wallet")["copyable_pnl"].sum() if not imp_val.empty else pd.Series(dtype=float)
    both_bl_pos = set(bl_pnl[bl_pnl > 0].index) & set(val_bl_pnl[val_bl_pnl > 0].index)
    print(f"\nLeaders positive on BOTH val and test: {len(both_bl_pos)}")

# ─── Test with leader concentration cap ───
print("\n" + "-" * 70)
print("LEADER CONCENTRATION CAP: max 15% of total per leader")
print("-" * 70)

if not imp_test.empty:
    total = imp_test["copyable_pnl"].sum()
    cap = total * 0.15
    # Find leaders that would be capped
    leader_totals = imp_test.groupby("leader_wallet")["copyable_pnl"].sum()
    capped_leaders = set(leader_totals[leader_totals > cap].index)
    print(f"Leaders to cap: {len(capped_leaders)}")

    # For capped leaders, keep only a subset of their implied trades
    if capped_leaders:
        parts = []
        for leader, grp in imp_test.groupby("leader_wallet"):
            if leader in capped_leaders:
                n_keep = max(1, int(len(grp) * cap / grp["copyable_pnl"].sum()))
                parts.append(grp.nlargest(n_keep, "copyable_pnl"))
            else:
                parts.append(grp)
        imp_capped = pd.concat(parts, ignore_index=True)
        print(f"  Capped PnL: ${imp_capped['copyable_pnl'].sum():,.2f}")
    else:
        print("  No leaders need capping")

# ─── Final recommended config ───
print("\n" + "=" * 80)
print("FINAL RECOMMENDED CONFIG")
print("=" * 80)
print("  Follower selection: min_copyable_roi=0.30, min_num_buckets=30, max_market_pnl_hhi=0.3")
print("  Leader selection: stable (2/3 train splits profitable), min_trade_count=20")
print("  Pair filter: min_observations=3 on training data")
print("  Time window: 30 minutes")
print("  No iterative refinement (causes overfitting)")

r_final = eval_full("FINAL: fw(0.30)+stable_bl+pair(3)+tw30", fw_t, bl_t, sl_t, tw=30)

# Also show the 60min variant
buy_imp60 = detect_implied_buys(df_train, hq_fw, stable_bl_ws, time_window_minutes=60, leader_side="BUY")
sell_imp60 = detect_implied_buys(df_train, hq_fw, stable_sl_ws, time_window_minutes=60, leader_side="SELL")
buy_f60 = filter_pairs_by_frequency(buy_imp60, min_observations=3)
sell_f60 = filter_pairs_by_frequency(sell_imp60, min_observations=3)
bl_60 = set(buy_f60["leader_wallet"]) if not buy_f60.empty else set()
sl_60 = set(sell_f60["leader_wallet"]) if not sell_f60.empty else set()
fw_60 = (set(buy_f60["follower_wallet"]) if not buy_f60.empty else set()) | (set(sell_f60["follower_wallet"]) if not sell_f60.empty else set())

r_60 = eval_full("60min variant: fw(0.30)+stable_bl+pair(3)+tw60", fw_60, bl_60, sl_60, tw=60)
