# Idea: Fade Reactive Sell Flow

## Summary

Fade candidate copy BUYs that enter right after a burst of **same-outcome sells** by
reactive wallet groups. A candidate entering into an active sell-off on the same
condition+outcome should be a worse trade.

## Hypothesis

The strongest single predictor seen in grounding exploration is **same-side reactive
sell flow in the prior 30 minutes** — count of SELL trades by BOTH_SIDES /
OVERSELLER / RETAIL wallets on the same (condition, outcome) shortly before the
candidate BUY:

- `both_sides_sell_30m` IC -0.196 / -0.168 / -0.135 (train/val/test)
- `overseller_sell_30m`  IC -0.171 / -0.134 / -0.113
- `retail_sell_30m`      IC -0.091 / -0.100 / -0.076

The effect was independent of first-mover status in the grounding exploration.

These are *flow* signals (no position engine needed): count recent SELLs by the target
wallet group on the same (condition, outcome) before time t.

## Candidate Trades

- Base universe: existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- `sig_fsf_sell_30m_<set>` — SELL-trade count on same (condition, outcome) in last 30m, per set.
- `sig_fsf_sell_1h_<set>` / `sig_fsf_sell_6h_<set>` — longer windows (best retained after sweep).
- `sig_fsf_sell_qty_30m_<set>` — quantity-weighted variant (distinct from count).
- `sig_fsf_sell_distinct_30m_<set>` — distinct-wallet variant.

Sets: BOTH_SIDES, OVERSELLER, RETAIL.

Higher = worse (expected negative IC). The copy-filter direction is the negation.

## Expected Outcome

Negative IC against `roi_res`, strongest for BOTH_SIDES / OVERSELLER on the 30m window;
longer windows test persistence of the effect.

## Findings

**Status:** Confirmed — the strongest signal family so far. All signals highly
significant with consistent sign across train/val/test on the full Weather candidate
universe (140,802 candidate BUYs).

Full-data IC vs `roi_res` (train / val / test), SELL-trade count on same
(condition, outcome) in the prior window:

| signal | IC_train | IC_val | IC_test |
|---|---|---|---|
| `sig_fsf_sell_0.5h_both_sides` | -0.196 | -0.168 | -0.135 |
| `sig_fsf_sell_1h_both_sides` | -0.197 | -0.180 | -0.144 |
| `sig_fsf_sell_6h_both_sides` | -0.195 | -0.186 | -0.138 |
| `sig_fsf_sell_0.5h_overseller` | -0.171 | -0.134 | -0.113 |
| `sig_fsf_sell_6h_overseller` | -0.188 | -0.161 | -0.123 |
| `sig_fsf_sell_0.5h_retail` | -0.091 | -0.100 | -0.076 |
| `sig_fsf_sell_6h_retail` | -0.113 | -0.129 | -0.097 |

- BOTH_SIDES is flat across taus; OVERSELLER and RETAIL are best at 6h — so
  `taus_h=[6.0]` is retained (3x less compute than 0.5h).
- `sell_distinct` (distinct-wallet) variant ≈ `sell` (rank corr ~1); count only kept.
- Presence on train ≥ 0.15 for retail, ≥ 0.5 for overseller, ≥ 0.86 for both_sides.
- Test decay is mild and sign holds — this is a robust fade signal.

Copy direction: inverse signal — avoid/underweight candidate BUYs with recent
same-outcome sells by BOTH_SIDES / OVERSELLER / RETAIL.

**Next steps if pursued:** strongest candidate to combine with
`CopyCrowdEntryTiming` (independent mechanism); check correlation before merging.
