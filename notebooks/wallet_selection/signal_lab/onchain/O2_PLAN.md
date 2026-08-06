# O2 — Re-test Finance/Politics with corrected selection

**Status**: Plan only. The next agent should run the diagnosis-driven
re-test described below and report the result.

## Motivation (from O1)

`o1_diagnose.py --all-tags` produced `o1_diagnosis.json`. The two
relevant findings:

- **Finance**: selected-wallet `buy_roi` mean is **0.29**, but
  rejected-wallet `buy_roi` mean is **0.40** (selected is *worse*).
  Only 32 wallets pass selection. The `lead_h→outcome` IC is 0.20 —
  high — meaning trades close to resolution are much more accurate.
- **Politics**: rejected-wallet `buy_roi` mean is **1.5** (!). The
  selection is filtering out the high-ROI wallets. Market
  concentration is 49x (a few markets dominate) and one-off market
  share is 49%.

## Plan: one-shot re-test

For each of `Finance` and `Politics`, build an alternative wallet
selection and re-run the existing `signal_lab` evaluation pipeline.
The goal is to answer the question: with a confounder-aware selection
does the in-sample copy edge survive out-of-sample?

### Selection candidates (one of each pair is enough)

For **Finance**:
- `selection_v2a`: drop the `min_buckets` and `min_trade_count` floors
  (use the same `min_buy_roi` and `min_copyable_roi`).
- `selection_v2b`: same as v2a, plus a soft cap on max drawdown
  (`max_drawdown_to_pnl <= 0.8` instead of 0.6), to admit wait-and-see
  wallets that the strict drawdown filter excluded.

For **Politics**:
- `selection_v2a`: drop `max_drawdown_to_pnl` constraint entirely,
  add an outlier trim (drop wallets with single-market PnL > 50% of
  total PnL).
- `selection_v2b`: same as v2a, plus require the wallet to trade on
  ≥ 3 markets (mitigate one-off-market overfitting).

### Evaluation

For each (tag, selection):
1. Run `load_stage1_data()` with `DEFAULT_TAGS={tag}`.
2. Apply the new selection, get candidate BUY trades.
3. Run the existing `evaluate_composite.py` to compute the
   confirmed-signal IC and capital-constrained PnL, with the same
   chronological train/val/test split as the Weather run.
4. Report the train / val / test ICs and PnL on test.

### Pass criteria (per `signal_lab/AGENTS.md`)

- train and validation IC same sign
- pooled train+val bootstrap CI excludes zero
- effect survives within-price-bin check
- test directionally consistent

If even one of the four (tag, selection) combinations passes, we
keep the tag for deeper work. If all four fail, the track is closed
with a one-paragraph "no edge under corrected selection" note in
PROGRESS.md.

## Files

- Inputs: `o1_diagnosis.json`, `lib.py`, `evaluate_composite.py`,
  `sizing.py`, `signal_engines.py`, `signal_lib.py`, `filters.py`.
- Outputs (one set per selection): `o2_{tag}_{selection}_summary.json`,
  `o2_{tag}_{selection}_pnl.csv`.
- Verifier: `verify_o2.py` — schema checks + a single boolean
  `any_passed` written to `o2_pass_summary.json`.

## Estimated runtime

Loading one tag's data takes ~30s. Running `evaluate_composite.py`
per tag/selection takes 1-2 minutes. Total ≈ 15 minutes. Within
budget.
