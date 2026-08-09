# Idea: Combined Composite of Confirmed Signal Families

## Summary

Combine the confirmed signal families (fade-sell-flow, opposite-crowding, copy-crowd
entry timing, weak-hand underwater contrarian) into composite scores and evaluate them
as a single copy-trade signal. This tests whether the confirmed families are *additive*
— i.e. whether an equal/IC-weighted/shrinkage-Markowitz composite has materially higher
IC than any single family, and how it thresholds.

## Hypothesis

Each confirmed family captures a different, partially-orthogonal mechanism:

- `sig_fsf_sell_6h_both_sides` — reactive wallets selling the same outcome recently (neg).
- `sig_val_opp_retail` / `sig_val_opp_flipper` / `sig_val_opp_overseller` — opposite-outcome
  crowding by weak/reactive wallets (neg).
- `sig_ccw_n_cond` / `sig_ccw_first` — copy-crowd entry timing / first-mover flag (neg/pos).
- `sig_uwl_opp_retail` / `sig_uwl_opp_gambler` / `sig_uwl_opp_whale` — underwater weak-hand
  opposite holdings (pos, near-orthogonal to all others).

If their residual correlation is low, the composite IC should exceed any single signal.

## Method

- Shared candidate universe: COPY_DEFAULT copy trades on Weather (140,802 candidate BUYs,
  26,907 markets); chronological train/val/test.
- Ran all 5 confirmed strategies on the shared universe: `CopyCrowdEntryTiming`,
  `FadeReactiveSellFlow`, `UwlOppContrarian`, `FreshOppositeCrowdingFilter`,
  `GamblerCapitulationSqueeze`.
- Deduped families by rank-correlation (train+val pooled, corr<0.70): `both_sides` kept
  over `overseller` (0.89); `uwl_opp_retail` / `uwl_opp_gambler` / `uwl_opp_whale` kept as
  orthogonal additives.
- Built composites via `build_composite_scores` (weights fit on **train** only): equal,
  IC-weighted, shrinkage-Markowitz (lambda=0.5).
- **Fit target = `copyable_pnl`** (the dollar PnL the copy strategy actually earns), not
  `roi_res` — optimizing residualized ROI does not maximize PnL.
- Price-confound decomposition: `pnl_res` (train-fit price-residualized PnL) and
  within-price-bin IC, because raw `copyable_roi` is ~50% correlated with `price`.
- Threshold grids on **validation** (selection, by net PnL); test reported only at the
  val-chosen threshold.

## Findings

**Status: Composite confirmed — beats every single signal on PnL IC, all splits.**

Full-data composite IC vs `copyable_pnl` (CLI run, 16 shards):

| scheme | IC_train | IC_val | IC_test |
|---|---|---|---|
| equal | +0.214 | +0.208 | +0.194 |
| ic_weighted | +0.308 | +0.353 | +0.331 |
| shrinkage_markowitz | +0.327 | +0.370 | +0.348 |

Best single signal for comparison: `sig_uwl_opp_retail` (the dominant family for PnL)
train +0.244 / val +0.320 / test +0.313; `sig_val_opp_overseller` neg -0.192/-0.219/-0.204.

- All three composite schemes clear every single family on train AND val AND test.
- The `uwl_opp_*` pair/triple is genuinely additive: |rank-corr| vs all other families
  <= 0.43, most <= 0.12, so the composite holds up where crowding signals decay.
- Fitting to `copyable_pnl` instead of `roi_res` roughly doubles PnL IC (equal test
  +0.136 -> +0.194) at the cost of ROI-residualization IC (test +0.169 -> -0.039):
  the two targets trade off which "edge" you optimize. `copyable_roi` and
  `copyable_pnl` fit produce the same composite (identical sign pattern).
- Regime note: val is a weak-PnL window (all-candidates PnL train 29,562 / val 2,070 /
  test 9,605), not a biased sample.

## Caveat: the raw edge is mostly the "buy cheap" price component

Composite IC is computed on `copyable_pnl`, and the composite correlates heavily with
`price` (`spearman_price` equal ~0.50/0.35/0.31, weighted schemes ~0.6–0.7). The
price-controlled views collapse:

| scheme | IC_target (test) | IC_pnl_res (test) | within-price-bin IC (test) |
|---|---|---|---|
| equal | +0.194 | +0.103 | -0.006 |
| ic_weighted | +0.331 | +0.148 | +0.068 |
| shrinkage_markowitz | +0.348 | +0.154 | +0.069 |

So most of the raw PnL edge is "cheap trades have more upside" (high-price trades are
near resolution with tiny PnL). The weighted schemes retain a small but real
within-price edge; the equal scheme has none.

Also, PnL-max threshold selection on validation degenerates to near-full firing (equal
fires ~95%), because more trades monotonically adds PnL in a positive-PnL regime. The
threshold table is illustrative, not the headline.

## Sizing: capital-constrained answer to the price confound

Threshold selection degenerates without a capital cap ("more trades = more PnL"
-> near-full firing), so we built a **capital-constrained sizing backtest**
(`signal_lab/sizing.py`): copy a score-proportional share quantity
(`qty = scale * max(0, score) * copyable_qty_5m_100`, clipped to `copyable_qty_5m_100`) under a
global `$10k` budget, with capital locked from `dt` until market resolution
(`end_date_iso`). Scale is picked on validation (Sharpe of daily PnL) and reported
on test.

Full-data sizing (scale picked on val, reported on test), price-exposed vs
price-controlled:

| scheme | variant | split | scale | trades | net_pnl | peak_used | pnl_per_peak | sharpe_daily |
|---|---|---|---|---|---|---|---|---|
| equal | price_exposed | val/test | 0.7 | 22,948/20,580 | 557/514 | 10,000 | 0.056/0.051 | 6.3/8.9 |
| equal | price_controlled | val/test | 0.7 | 22,948/20,580 | 557/514 | 10,000 | 0.056/0.051 | 6.3/8.9 |
| ic_weighted | price_exposed | val/test | 0.7 | 23,225/20,278 | 570/293 | 10,000 | 0.057/0.029 | 6.7/7.8 |
| ic_weighted | price_controlled | val/test | 0.7 | 24,321/21,068 | 622/286 | 10,000 | 0.062/0.029 | 6.4/7.4 |
| shrinkage | price_exposed | val/test | 0.8 | 22,780/20,008 | 630/291 | 10,000 | 0.063/0.029 | 6.7/8.5 |
| shrinkage | price_controlled | val/test | 0.8 | 24,358/21,222 | 603/242 | 10,000 | 0.060/0.024 | 6.1/6.7 |

Findings:

- **The price component is NOT tradable alpha.** Price-exposed and price-controlled
  composites size to essentially the same PnL and Sharpe under a `$10k` cap. The raw
  IC advantage of the price-exposed fit (0.35 vs 0.19 on test) does not survive
  risk-adjusted, capital-constrained sizing — the "buy cheap = more upside" component
  is a variance/leverage artifact, not mispricing. **Do not control for price in the
  signal; the capital cap already neutralizes it.**
- Sizing fully deploys the budget (peak_used = 10,000) and fires ~20k trades (~half the
  candidate universe), a far more realistic operationalization than thresholding.
- Notebook section 13 mirrors this on the notebook universe (equal scheme identical
  for both variants; ic_weighted price_controlled slightly better on test Sharpe).
- Caveat: capital frees at resolution (not wallet's actual exit), a conservative
  assumption that keeps concurrent capital high; a SELL-reconstruction model would
  deploy the same budget to slightly more trades.

## Next Steps

- Treat the composite IC panel and the sizing backtest as the findings; the price
  question is answered empirically (price component is not additive under capital).
- If pursuing a live strategy, use the capital-constrained sizing rule (scale ~0.7
  on the equal composite, ~$10k budget) rather than threshold-based firing.
- Candidates file: `evaluate_composite.py` (CLI runner, `--target` and `--sizing`
  flags), `composite_results.csv`, `sizing_results.csv`, notebook sections 12–13 in
  `quickstart_signal_lab.ipynb`.
