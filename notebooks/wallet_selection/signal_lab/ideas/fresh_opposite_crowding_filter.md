# Idea: Fresh Opposite Crowding Filter

## Summary

Filter copy-trades so we copy a candidate BUY only when the opposite outcome is not heavily and recently crowded by reactive wallet groups.

This is a copy-trade strategy filter, not an opposite-side strategy.

## Hypothesis

The current stage-1 work suggests that large aggregate positions on the opposite side are often a negative sign for the candidate BUY after controlling for price.

A plausible refinement is that recent opposite-side crowding is more informative than stale crowding. If flippers, scalpers, both-side traders, or oversellers have built the opposite side recently, that may indicate near-term crowding, noisy flow, or short-horizon exhaustion. Candidate BUYs into that setup may underperform.

If true, the best copy-trades may be the candidate BUYs where recent opposite-side crowding is low.

## Candidate Trades

- Base universe: the existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

Start narrow. Test only a few recency-weighted signals:

- `sig_fval_opp_flipper`
- `sig_fval_opp_both_sides`
- `sig_fval_opp_overseller`
- `sig_fpos_opp_flipper`

Recommended first taus:

- 1 hour
- 6 hours
- 24 hours

The framework parameter is `fresh_tau_ns`.

## Minimal Evaluation Plan

1. Build the stage-1 workspace.
2. Attach only the small set of archetypes above.
3. Evaluate `fval_opp` and `fpos_opp` on `roi_res`.
4. Keep only sign-consistent train/validation results.
5. If one or two signals look promising, compare them against the non-fresh baseline:
   - `sig_val_opp_flipper`
   - `sig_val_opp_both_sides`
6. Only if fresh beats plain crowding on validation, try a simple thresholded composite.

## Success Criteria

This idea is promising if:

- fresh opposite crowding is more negative than the non-fresh baseline on validation
- the sign is stable on train and validation
- trade count does not collapse too much after filtering
- the rough test read still points in the same direction

## Likely Failure Modes

- Fresh signals may mostly repackage the same information as plain `val_opp`
- Very short taus may create sparse or unstable signals
- Any raw effect may still be partly price-linked if not checked on `roi_res`

## Recommendation For First Pass

Do not test every archetype or every family.

Start with:

- archetypes: `flipper`, `both_sides`, `overseller`
- families: `fval_opp`, `fpos_opp`
- taus: 1h and 6h

If that is weak, stop.

If that is promising, add:

- tau 24h
- `max_dd`
- a small composite of the best 1-2 signals

## Comments (2026-08-01)

- Applicability: this signal is directly applicable to stage-1 candidate BUY trades (`candidate_splits`) and can be used as a copy-filter score threshold.
- Runtime note: most runtime comes from rebuilding per-set position tables; caching `(set_name, fresh_tau_ns)` tables made repeated runs much faster.
- Flipper: `sig_fval_opp_flipper` is consistently slightly weaker than baseline `sig_val_opp_flipper` at 1h and 6h.
- Both-sides: fresh is close but generally not better than baseline; at 6h it is near-tied.
- Overseller: fresh shows the best relative behavior; `sig_fval_opp_overseller` is slightly stronger than baseline on validation at 6h/24h.
- Max-dd: mixed; fresh underperforms baseline at 1h and is only marginally better at 6h.
- Thresholding caveat: single-signal thresholds can look weak out-of-sample; selectivity matters because very low thresholds may keep most trades and behave like mild reweighting.
- Composite note: a simple `val_opp + fval_opp` composite for overseller at 6h looked stronger than single-signal thresholding in the quick pass.
