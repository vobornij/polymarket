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

## Expected Outcome

A strong negative IC. The heavier the recent crowding on the opposite side, the worse the expected outcome of the candidate trade.

## Findings

**Status:** Highly Promising!

Testing `val_opp` against various fresh taus (`1h`, `6h`, `24h`) for `both_sides`, `overseller`, and `flipper` archetypes yielded intensely negative, stable, and highly significant ICs across the board.

For instance, `sig_fval_opp_24h_both_sides` had:
- Train IC: -0.203
- Val IC: -0.195
- Test IC: -0.128

These results are remarkably strong and survive into the test split beautifully.

Interestingly, while the **fresh** versions (especially the 24-hour and 6-hour taus) are incredibly strong signals, they do not appear to drastically outperform the **baseline** `val_opp` (lifetime value at cost). The baseline `sig_val_opp_both_sides` was nearly identical (Train: -0.205, Val: -0.194, Test: -0.127).

**Conclusion:** The presence of large counter-positions by these reactive archetypes is a definitive negative filter for a copy-trade. The "freshness" of the crowding is perfectly valid, but raw crowding `val_opp` captures almost exactly the same edge without requiring tau tuning. Either one is a top-tier filter.

## Comments (2026-08-01)

- Applicability: this signal is directly applicable to stage-1 candidate BUY trades (`candidate_splits`) and can be used as a copy-filter score threshold.
- Runtime note: most runtime comes from rebuilding per-set position tables; caching `(set_name, fresh_tau_ns)` tables made repeated runs much faster.
- Flipper: `sig_fval_opp_flipper` is consistently slightly weaker than baseline `sig_val_opp_flipper` at 1h and 6h.
- Both-sides: fresh is close but generally not better than baseline; at 6h it is near-tied.
- Overseller: fresh shows the best relative behavior; `sig_fval_opp_overseller` is slightly stronger than baseline on validation at 6h/24h.
- Max-dd: mixed; fresh underperforms baseline at 1h and is only marginally better at 6h.
- Thresholding caveat: single-signal thresholds can look weak out-of-sample; selectivity matters because very low thresholds may keep most trades and behave like mild reweighting.
- Composite note: a simple `val_opp + fval_opp` composite for overseller at 6h looked stronger than single-signal thresholding in the quick pass.

## Re-evaluation on the fixed pipeline (2026-08-03)

Status changed: **IC real, edge NOT monetizable as a copy overlay. Do not deploy as a filter/sizing signal.**

Context: the `signal_lib._rankdata` bug (rank vs inv) was fixed; all ICs below are from the fixed pipeline with train-only rank normalization, Weather-only data. Full protocol and tables:

- `reevaluate_crowding.py` — signal panel, decile gate, firing-rate test
- `crowding_overlay.py` — capital-constrained ($10k) walk-forward sizing overlay; Sharpe selected on train (fold A) / train+val (fold B), reported on val / test
- `crowding_reeval_{panel,deciles,firing}.csv`, `crowding_overlay_results.csv`, `crowding_overlay_sharpe_ci.csv`

### What is real

- The crowding IC is confirmed on the fixed pipeline. `sig_fval_opp_24h_both_sides` vs `roi_res`: train −0.203 / val −0.195 / test −0.128; within-price-bin IC −0.08…−0.15, sign-consistent, bootstrap excludes zero.
- A blended copy score (negated, rank-normalized) has *consistent positive* IC on raw targets: `copy_blend` vs `copyable_roi` +0.084/+0.108/+0.098 and vs `copyable_pnl` +0.14/+0.157/+0.148; `copy_overseller` and `copy_max_dd` are strongest on raw `copyable_roi` (+0.17…+0.23).

### Why it does not monetize

- The score ranks `roi_res` monotonically across all splits (decile mean −0.30→+0.43 train) **but dollar PnL concentrates in the middle deciles**, and the low-crowding (high-score) tail carries little PnL.
- Firing-rate selection (top 10/20/30/50/70/90%) is below copy-all PnL on test in every case (top-10% ≈ +106 vs 9,605 copy-all); the high-crowding deciles are the big PnL contributors ("buy cheap = more upside" effect dominates dollars).
- Capital-constrained sizing confirms this at the Sharpe level. Deployment fold (select train+val → test):
  - best crowding variant `copy_blend@10%` Sharpe **0.63** but deploys only ~$3.6k of the $10k (pnl_per_peak 0.03)
  - `copy_max_dd` 0.48, `copy_overseller` 0.34
  - **copy-all 0.81, price-favorite sizing 0.99** — both beat every crowding overlay
  - the fold-A winner (`copy_overseller@25%`, val Sharpe 0.32 vs copy-all 0.14) flips to **−0.02 on test**
- Block-bootstrap (7-day) Sharpe CIs on test are wide and overlap zero for both overlay and copy-all → daily Sharpe here is dominated by a few resolution days, not a stable edge.

### Verdict

- The roi_res IC is a real but **residualization-only** effect; it does not survive as deployable alpha in a capital-constrained copy overlay.
- The raw-PnL edge ("copy cheap") is variance under the cap, best captured by copy-all / price-favorite sizing, not by crowding scoring.
- **Recommendation:** do not ship a crowding-filtered or crowding-sized copy. If deploying anything from this thread, it is the plain capital-constrained copy-all (test Sharpe ~0.8, pnl/peak ~1.1), which is a separate claim from this idea.


