# Idea: What Profitable Copyable Wallets Do Differently (Wallet-Quality Gradient)

## Summary

The copy universe (`COPY_DEFAULT`) is already a profitable slice of the population,
but it is heterogeneous. If *within* the copy universe there is a quality gradient —
candidates from higher-ROI copy wallets are better trades — we can weight or subset
candidates by the copy wallet's own train-period quality, sharpening the edge for free.

## Hypothesis

Candidate `roi_res` increases with the copy wallet's train-period quality metrics
(`buy_roi`, `copyable_roi`, `copyable_pnl`, `opening_roi`). If the gradient is flat,
the copy thresholds already saturate quality and the per-trade signals (Ideas 1-4) are
the only lever.

## Candidate Trades

- Base universe: existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

Per-candidate copy-wallet features from `wallet_metrics` (train-period):

- `sig_wal_buy_roi` — wallet's train buy ROI.
- `sig_wal_copyable_roi` — ROI weighted by copyable-pnl share.
- `sig_wal_copyable_pnl` — absolute copyable PnL.
- `sig_wal_opening_roi` — ROI of opening (non-exit) trades.
- `sig_wal_trade_count` / `sig_wal_num_markets` — activity scale.

## Expected Outcome

If positive and val/test-consistent, copy-wallet quality is a usable upward weight.
Caveat: `wallet_metrics` is train-period, so **train IC is inflated** (a train
candidate's own outcome feeds its wallet's train metric); the honest read is val/test.

## Findings

**Status: Weak but real gradient, secondary to Ideas 1-2.** Within the copy universe
there is a small quality gradient in candidates' residualized ROI. Full-data IC vs
`roi_res` (train / val / test):

| signal | IC_train | IC_val | IC_test | verdict |
|---|---|---|---|---|
| `sig_wal_copyable_roi` | +0.038 | +0.037 | +0.017 | consistent (+), weak |
| `sig_wal_num_markets` | +0.035 | +0.064 | +0.038 | consistent (+), weak |
| `sig_wal_copyable_pnl` | -0.007 | +0.035 | +0.073 | train ≈ 0, sign flip |
| `sig_wal_trade_count` | +0.005 | +0.034 | +0.066 | train ≈ 0 |
| `sig_wal_buy_roi` | +0.003 | -0.020 | -0.002 | ≈ 0 |
| `sig_wal_opening_roi` | -0.048 | -0.045 | -0.047 | consistent (−), strongest |

- The train-inflation caveat does not drive the result: `copyable_roi` stays positive
  on val/test, so the gradient is real (leakage alone would inflate only train).
- Counter-intuitive: wallets whose *opening* trades were most profitable are the worst
  to copy (negative). High-scope wallets (`num_markets`) and higher `copyable_roi` are
  modestly better.
- Magnitudes (~|0.03-0.05|) are far below the sell-flow (-0.20) and copy-crowding
  (-0.14) signals. Usable only as a secondary upward/downward weight, if at all.

No param tuning needed (single-shot feature set).
