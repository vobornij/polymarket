# Idea: Whale vs. Consistent Divergence

## Summary

Test the performance of candidate BUYs when the biggest capital (`whale`) disagrees with the smartest capital (`consistent`).

## Hypothesis

If `consistent` (smart money) has a strong position on the `own` side, but `whale` (massive capital) holds heavily on the `opp` side, the market might be mispriced due to whale weight artificially suppressing our side. Trading alongside `consistent` and against the `whale` might yield outsized returns.

## Candidate Trades

- Base universe: the existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- Interaction/spread: `sig_pos_own_consistent` and `sig_pos_opp_whale`.
- Or simply combining both signals and checking if the combined state improves hit rate.

## Expected Outcome

Positive IC / high hit rate on the combined interaction against `roi_res`.

## Findings

**Status:** Failed / Rejected.

Evaluating the product/interaction of the two features (`pos_own_consistent * pos_opp_whale`) showed an unstable and overall negative relationship:
- Train IC -0.035, Val IC -0.055, Test IC +0.009.

This suggests that when whales oppose the smart money, the whales actually tend to win (negative relationship with our candidate BUY), but even this effect reverses out of sample in the test set. The interaction divergence is not robust.
