# Oversell Position & Valued-Position Signals — Most Promising

This document describes the oversell signal module (notebook cell 37) after cleanup,
focusing on the signals that passed selection and the rationale for keeping them.

## What the module does

An "overseller" is a wallet that is net-negative on its SELLs yet profitable overall
(train-only selection from `wallet_vol`). The module computes, for every candidate BUY
trade at time `t`, the **exact aggregate state of an overseller set's open positions on
the same token at that moment**, and expresses it in four families:

| family | column(s) | definition |
|---|---|---|
| `pos`   | `sig_os_opp_*` | aggregate position (quantity) |
| `val`   | `sig_os_val_{own,opp,total}_*` | aggregate **value-at-cost** (USDC) of the open position, using average-cost accounting |
| `avgc`  | `sig_os_avgc_own_*` | `val / pos / price − 1` — position-weighted entry premium vs the candidate's current price |

Signals exist for the set's position on the **own** outcome, the **opposite** outcome,
and (for `val`) the market total. The exact aggregate is computed with a two-pass
cumsum over post-trade checkpoints plus `merge_asof` (`A(t) − B(t)`), validated against
a per-wallet brute force. Value-at-cost is execution-order-aware (checkpoints ordered by
`(wallet, condition, outcome, dt, −position)`).

## Overseller sets (after cleanup)

| set | mask (train) | n wallets |
|---|---|---|
| `os_pnl`  | `sell_pnl < 0 & total_pnl > 0` | 865 |
| `os_deep` | `os_pnl & sell_roi < −0.1` | 216 |
| `os_thin` | `os_pnl & buy_pnl < 50` | 202 |

Note: the original module also defined `os_roi` and `os_buyprof`. On this dataset they
resolve to the **identical 865-wallet set** as `os_pnl` (every profitable overseller
profits on the buy side), so they were removed. `os_active` (`sell_notional > 500`)
produced no selected signal and was removed.

## Evaluation protocol

- Splits: train ≤ 2026-05-21, val ≤ 2026-06-23, test > 2026-06-23 (chronological).
- Selection uses **train + val only**: sign-consistent IC on `copyable_roi`,
  `|IC| ≥ IC_MIN` (0.005) on both, presence ≥ 0.005.
- Test ICs are **diagnostics only** (no leakage into selection).

## The six selected signals

| signal | kind | pres. | IC_train | IC_val | IC_test |
|---|---|---|---|---|---|
| `sig_os_val_total_os_thin`  | val-total | 0.633 | **−0.0214** | **−0.0085** | −0.0005 |
| `sig_os_avgc_own_os_pnl`    | avgc-own  | 0.291 | −0.0144 | −0.0180 | +0.0147 |
| `sig_os_avgc_own_os_thin`   | avgc-own  | 0.155 | −0.0106 | −0.0052 | +0.0036 |
| `sig_os_val_own_os_deep`    | val-own   | 0.684 | −0.0089 | −0.0073 | −0.0024 |
| `sig_os_val_opp_os_thin`    | val-opp   | 0.537 | +0.0075 | **+0.0114** | −0.0026 |
| `sig_os_opp_os_deep`        | pos-opp   | 0.723 | −0.0051 | −0.0092 | +0.0072 |

### `sig_os_val_total_os_thin` — the strongest, most robust signal
Total value-at-cost held by thin oversellers (buy-side pnl < $50, i.e. sellers first)
on **both** outcomes of the candidate token. It has the highest train IC of any selected
signal and is the only one sign-consistent across **all three** splits
(−0.0214 → −0.0085 → −0.0005, monotonically decaying). Interpretation: when money sits
at-cost in this token's hands of net sellers, the candidate BUY underperforms.

### `sig_os_avgc_own_os_pnl` — the cost-basis concept
Position-weighted entry premium of the full overseller set vs current price on the
candidate outcome. The **most economically novel** family: when oversellers are deeply
underwater on their average cost basis (val ≫ pos·price), the candidate BUY is bad
(IC −0.0144/−0.0180). Val-consistent; flips on test.

### `sig_os_val_own_os_deep` — sign-consistent across all splits
Value-at-cost held on the candidate outcome by deep oversellers (sell ROI < −10%).
−0.0089/−0.0073/−0.0024: monotone and never changes sign.

### `sig_os_val_opp_os_thin` — opposite-side money
Value-at-cost held by thin oversellers on the **opposite** outcome. Positive IC that
*strengthens* on val (+0.0075 → +0.0114): when money is parked on the other side, the
candidate BUY does better. Flipped on test (−0.0026).

### `sig_os_avgc_own_os_thin` and `sig_os_opp_os_deep`
Weaker but val-consistent: high cost-basis premium in thin-overseller hands (avgc) and
large opposite-outcome positions held by deep oversellers (`pos`-opp) both point to
underperforming candidate BUYs.

## Near misses worth watching

| signal | IC_train | IC_val | IC_test | why it missed |
|---|---|---|---|---|
| `sig_os_avgc_own_os_deep` | **−0.0234** (strongest in report) | +0.0043 | +0.0064 | val flips sign |
| `sig_os_uwl_opp_os_active` | +0.0111 | +0.0044 | +0.0026 | sign-consistent all splits but `|IC_val|` just under the 0.005 bar |

## Related finding: overseller counterparty group (cell 52)

Buying the other side of overseller SELLs, restricted to the 115 wallets profitable on
those matched BUYs (train `copyable_roi > 0`, n ≥ 10, val sign-consistent):
train ROI **0.115**, val ROI **0.130**, test ROI 0.022. The train→val stability is the
best standalone result in the pipeline; the edge decays ~5x on test.

## Caveats

- All ICs are small (|IC| ≤ 0.024) — weak signal territory.
- Most signals flip sign on test; only `val_total_os_thin` and `val_own_os_deep` hold.
- `os_pnl`/`os_roi`/`os_buyprof` were identical sets; keep sets contrastive in future.
- The headline test ROI (0.089 vs 0.047 control) comes from the val-chosen threshold
  1.0 (top ~3.6% of composite) — treat as diagnostic, not robust.
- Presence is low for thin/deep sets (hold positions on only 15–18% of candidates).
