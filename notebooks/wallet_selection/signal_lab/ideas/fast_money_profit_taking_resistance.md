# Idea: Fast-Money Profit-Taking Resistance

## Summary

Test whether a high "entry premium" (profitability) for fast-money archetypes on the own side acts as a ceiling on price.

## Hypothesis

Archetypes like `flipper`, `scalper`, and `retail` tend to take profits quickly. If they hold large positions on our `own` side that are currently highly profitable (their average cost is much lower than the current price), they are likely to dump their bags soon, creating downward price resistance. Candidate BUYs into this setup might underperform.

## Candidate Trades

- Base universe: the existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- `sig_avgc_own_flipper`
- `sig_avgc_own_scalper`
- `sig_avgc_own_retail`

## Expected Outcome

Negative IC against `roi_res`. The more in-the-money these archetypes are, the worse our candidate BUY performs due to impending sell pressure.

## Findings

**Status:** Failed / Rejected.

The signals demonstrated massive instability across chronological splits:
- `sig_avgc_own_retail`: Train IC +0.094, Val IC -0.004, Test IC -0.031
- `sig_avgc_own_scalper`: Train IC +0.071, Val IC +0.010, Test IC -0.039
- `sig_avgc_own_flipper`: Train IC +0.053, Val IC +0.000, Test IC -0.049

While the train period suggested strong momentum (positive IC, contradicting the resistance hypothesis entirely), the effect evaporated or reversed in validation and test periods. The signal lacks any robust predictive power out of sample.
