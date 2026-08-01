# Idea: Gambler Capitulation / Dumb Money Squeeze

## Summary

Test whether a massive underwater position by retail/gamblers on the *opposite* side predicts strong forward returns for the *own* side.

## Hypothesis

Gamblers and retail traders are highly prone to the disposition effect (holding onto losers). When they accumulate a massive underwater position (`uwl`) on the opposite side, they eventually face exhaustion or capitulation. Candidate BUYs on our outcome may capture the momentum as the opposite side is squeezed.

## Candidate Trades

- Base universe: the existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- `sig_uwl_opp_gambler`
- `sig_uwl_opp_retail`
- *Optional:* `sig_fuwl_opp_gambler` with `fresh_tau_ns` (e.g., 24h) to see if *recently* trapped gamblers are a stronger signal.

## Expected Outcome

Positive IC against `roi_res`. The more underwater they are on the opposite side, the better our side performs.

## Findings

**Status:** Promising!

Initial evaluation showed robust, positive, and significant results across splits for `uwl_opp` against `roi_res`.

* `sig_uwl_opp_retail`: Train IC +0.018, Val IC +0.062, Test IC +0.099.
* `sig_uwl_opp_gambler`: Train IC +0.017, Val IC +0.048, Test IC +0.068.

The signs are stable and strengthen in the validation and test splits. The more underwater retail and gambler wallets are on the opposite side, the better the candidate copy-trade performs. Both standard and `fresh` (24h tau) versions were tested, yielding very similar results (the fresh version did not significantly improve upon the standard lifetime underwater amount).

This is a strong candidate for inclusion in the overall copy-trade signal ensemble.
