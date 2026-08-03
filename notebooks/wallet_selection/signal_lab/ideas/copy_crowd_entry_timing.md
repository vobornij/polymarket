# Idea: Copy-Crowd Entry Timing

## Summary

Test whether a candidate copy-trade is better when the copy-set wallet is an **early
entrant** into the market — i.e. when few other copy-set wallets have already bought the
same condition — and worse when the trade is a late follower into an already-crowded
market.

## Hypothesis

The copyable edge is heavily concentrated (top 1% of candidate BUYs produce >1.8x total
PnL). Working backwards from profitable copyable trades, the number of **distinct
copy-set wallets that already bought this condition before time t** (`cum_distinct_before`)
is a strong negative predictor of residualized forward ROI. The first copy wallet into a
market (0 predecessors) should outperform followers.

This is a *flow / entry-timing* family — the aggregate `val_opp` crowding family measures
*opposite-outcome* value at cost; this measures *own-set* herding on the same market.

## Candidate Trades

- Base universe: the existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- `sig_ccw_n_cond` — # distinct copy-set wallets that bought this condition before t (lifetime).
- `sig_ccw_n_co` — same, but per (condition, outcome).
- `sig_ccw_recent_6h_co` / `sig_ccw_recent_24h_co` — distinct copy-set wallets that bought
  the same (condition, outcome) in the last 6h / 24h.
- `sig_ccw_first` — binary first-mover flag (`n_cond == 0`).

Higher = worse (expected negative IC). The copy-filter direction is the negation.

## Expected Outcome

Negative IC against `roi_res`, strongest for the lifetime condition-level count; fresh
recency variants test whether *recent* herding is the operative mechanism vs lifetime
crowding.

## Findings

**Status:** Confirmed — all signals highly significant, consistent sign across
train/val/test, on the full Weather candidate universe (140,802 candidate BUYs,
26,907 markets).

Full-data IC vs `roi_res` (train / val / test):

| signal | IC_train | IC_val | IC_test |
|---|---|---|---|
| `sig_ccw_n_cond` (# distinct copy wallets bought condition before) | -0.139 | -0.113 | -0.068 |
| `sig_ccw_first` (first-mover binary) | +0.132 | +0.105 | +0.061 |
| `sig_ccw_n_co` (per condition+outcome) | -0.124 | -0.097 | -0.045 |
| `sig_ccw_recent_24h_co` | -0.106 | -0.107 | -0.050 |
| `sig_ccw_recent_6h_co` | -0.093 | -0.103 | -0.046 |

- All bootstrap CIs exclude zero (pooled train+val); presence on train ≥ 0.39.
- Lifetime condition-level crowding (`n_cond`) is the strongest; recency variants add
  modest independent signal — 24h dominates 6h, so only `taus_h=[24.0]` is retained.
- Test decay is expected (crowding edge erodes as a market resolves), but sign holds.

Copy direction: these are **inverse** signals — copy BUY only when `n_cond` / `n_co` /
recency counts are low, or when `first == 1`.

**Next steps if pursued:** combine with `FreshOppositeCrowdingFilter` (val_opp side) for
an independent mechanism; check overlap/correlation before merging.
