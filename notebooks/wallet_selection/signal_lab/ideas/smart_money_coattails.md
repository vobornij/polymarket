# Idea: Smart Money Coattails

## Summary

Test whether a candidate BUY is more profitable if a "smart" archetype just bought the exact same outcome within the last few minutes.

## Hypothesis

Aggregate position sizes (`pos`, `val`) can sometimes represent stale capital. Immediate trade *flow* might be a stronger signal. Using the `SetProximityEngine`, we can detect if a highly profitable archetype bought the same outcome moments before our candidate trade, validating the near-term flow.

## Candidate Trades

- Base universe: the existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- Proximity engine matching `BUY` events on the `own` outcome.
- Archetypes: `consistent`, `overseller_deep`.
- Taus / Tolerance: 15 minutes, 60 minutes.

## Expected Outcome

Positive IC for proximity to smart money buys against `roi_res`.

## Findings

**Status:** Failed / Rejected.

The evaluation showed that immediate proximity to smart money buys is not a positive predictor:
- `prox_consistent_15m` and `60m`: Showed slightly negative IC on Train and Val (-0.02) and weak positive on Test. Bootstrap CIs crossed 0. Not robust.
- `prox_overseller_deep_15m` and `60m`: Showed a significant *negative* relationship (Train -0.02, Val -0.05, Test -0.03). This means buying right after an overseller buys is actively detrimental.

The flow hypothesis failed; aggregate position sizing remains a much better proxy than near-term event proximity for these archetypes.
