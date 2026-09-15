# Idea: Copy-Wallet Exposure Threshold (Capture Fixed % of Wallet PnL)

## Summary

For each copied wallet, copy a candidate trade only when the wallet's
**post-trade exposure** is large enough to matter. Per-wallet exposure is
`position * price` ($ value of the wallet's book on that `(condition, outcome)`
after the trade) — both inputs are precomputed on trade data. The threshold is
chosen so the retained trades capture a fixed fraction of the wallet's PnL
(100 / 80 / 50 / 20 %).

## Hypothesis

A wallet only builds a large book when it has edge, so **exposure is a
conviction/edge signal**: high-exposure candidate trades should carry better
copyable PnL per dollar copied. Filtering below-threshold trades should improve
per-dollar edge over just scaling wallets when capital is limited.

## Methodology (v2, wallet-relative)

Cross-sectional exposure is dominated by **wallet size** (whale bankroll vs
retail bankroll), so v2 added a wallet-relative axis for the IC panel, all
train-fit, no leakage:

- `sig_exp_wt_rank` — within-wallet exposure rank in [0, 1].
- `sig_exp_wt_minmax` — within-wallet min-max in [0, 1] (diagnostic).
- `sig_exp_marg` / `sig_exp_marg_wt` — marginal exposure `quantity * price`
  and its wallet-relative rank (averaging-down contrast).
- `sig_exp_thr_{0.8,0.5,0.2}` — per-wallet PnL-capture thresholds.
- `sig_exp_top_{0.8,0.5,0.2}` — per-wallet percentile thresholds at the train
  exposure quantile `1 - t` (top 80/50/20% of the wallet's own book).

New confound check: **within-price-bin IC** (`wb_ic_roi`) — pooled IC vs
`roi_res` inside price quantiles, isolating residual edge after the price
component is held constant.

## Findings (2-shard sample, Politics tag)

**Status: not promising — fail-fast gate fired.** The positive exposure signal
is the *price component*, which prior work (`combined_composite`) already showed
is not tradable alpha under a capital cap.

IC vs `copyable_pnl` (train / val / test):

| signal | IC_train | IC_val | IC_test | spearman_price |
|---|---|---|---|---|
| `sig_exp_usdc` (= log) | +0.322 | +0.177 | -0.007 | 0.45 |
| `sig_exp_pctile` | +0.298 | +0.176 | -0.005 | 0.44 |
| `sig_exp_thr_0.8` | +0.227 | +0.164 | -0.031 | 0.36 |
| `sig_exp_pos` | +0.193 | +0.027 | -0.116 | 0.10 |

- The train/val positive dollar IC is heavily price-correlated
  (`spearman_price ≈ 0.45` for USDC/log/pctile) — exposure ≈ "wallet bought a
  big cheap book," and cheap trades carry more upside per share by construction.
- On the price-residualized target `roi_res`, the edge is **negative or zero on
  every split** (`usdc` val -0.061, test -0.174; `thr_0.5` val -0.033, test
  -0.174). High-exposure trades do not have better residualized ROI.

Sizing vs scaling (10bps, $10k, same universe, 2 shards): **the exposure overlay
was monotonically worse than copy-all on the (net-negative) test window**
(-0.97 → -2.14 Sharpe as frac tightened vs copy_all -0.60).

## Findings (full data, 16 shards, Politics tag)

Thumbnail IC panel (full data):

| signal | IC_train | IC_val | IC_test | spearman_price | wb_ic_roi |
|---|---|---|---|---|---|
| `sig_exp_usdc` | +0.324 | +0.157 | +0.064 | 0.42 | **-0.079** |
| `sig_exp_wt_rank` | +0.313 | +0.139 | +0.068 | 0.43 | **-0.070** |
| `sig_exp_wt_minmax` | +0.328 | +0.150 | +0.069 | 0.43 | **-0.058** |
| `sig_exp_marg` | +0.244 | +0.114 | +0.039 | 0.45 | **-0.108** |
| `sig_exp_thr_0.8` | +0.256 | +0.199 | +0.132 | 0.33 | +0.019 |
| `sig_exp_top_0.2` | +0.284 | +0.203 | +0.109 | 0.35 | -0.013 |

- Wallet-relative re-scaling (`wt_rank`) preserves the price correlation
  (`spearman_price ≈ 0.43`) and leaves **no residual IC within price bins**
  (`wb_ic_roi ≈ -0.07`). The size/conviction axis has no independent edge.
- The binary PnL-capture threshold `thr_0.8` is the only signal with a mildly
  positive honest view (`roi_res` train +0.02 / val +0.07 / test +0.03,
  `wb_ic_roi` +0.019) — but note the 2-shard sample showed **negative** val/test
  for the same signal (-0.03 / -0.19), so it is not stable.

Capture realization on full test is well-behaved: realized capture ≈ frac
(0.94 / 1.02 / 0.83 / 0.43 for f = 1.0 / 0.8 / 0.5 / 0.2).

Sizing (10bps, $10k, same universe, full data):

| design | val roi_w | val Sharpe | test roi_w | test Sharpe |
|---|---|---|---|---|
| copy_all | 0.231 | 0.882 | 0.131 | 0.479 |
| tier4@2-0 (best val scaling) | 0.521 | 1.184 | -0.212 | -1.217 |
| uniform@0.5 | 0.223 | 0.940 | 0.077 | 0.299 |
| exposure_0.8 | 0.046 | 0.520 | **0.419** | **1.783** |
| exposure_0.5 | 0.090 | 0.986 | 0.330 | 1.396 |
| exposure_top50 | 0.134 | 0.819 | 0.106 | 0.683 |
| exposure_top20 | 0.020 | 0.229 | 0.085 | 1.237 |

## Weather replication (2-shard sample; full Weather load is >15 min, skipped per guardrail)

The same panel on Weather contracts (2 shards, ~1.8M trades):

| signal | IC_train | IC_val | IC_test | spearman_price | wb_ic_roi |
|---|---|---|---|---|---|
| `sig_exp_usdc` | +0.195 | +0.080 | +0.030 | 0.57 | +0.047 |
| `sig_exp_wt_rank` | +0.090 | +0.020 | +0.038 | 0.45 | -0.039 |
| `sig_exp_wt_minmax` | +0.160 | +0.019 | -0.008 | 0.56 | -0.011 |
| `sig_exp_marg` | +0.183 | +0.070 | -0.001 | 0.70 | -0.008 |
| `sig_exp_thr_0.8` | +0.153 | +0.035 | +0.009 | 0.46 | -0.010 |

- Dollar IC decays train→val→test (0.19 → 0.08 → 0.03) and is even *more*
  price-correlated than on Politics (`spearman_price` 0.57–0.70).
- `roi_res` IC is ~0 on every split (e.g. `usdc` val -0.005 / test -0.102,
  `thr_0.8` val -0.012 / test -0.034), and `wb_ic_roi ≈ 0`.
- The one mildly positive Politics read (`thr_0.8`, val +0.074 / test +0.033)
  does **not** replicate on Weather (val -0.012 / test -0.034).

Cross-tag verdict: under the pooled-IC/skip-race lenses this exposure overlay
looked like a **leverage dial on the copy book** (it amplifies whatever regime
the test window is in) rather than a selector with independent edge — no signal
has a consistent positive `roi_res` IC across tags, splits, or price bins.
That framing is what the unrestricted-sim rerun below re-examines: the dial
impression was largely a budget-skip artifact, while the *conditional*
(bin/wallet) read survives.

## Conditional bin diagnostic (Phase C, full Politics + 2-shard Weather)

The user's challenge: with bankroll management, **higher relative exposure should
require higher edge (Sharpe)** — so pooled rank IC was the wrong lens
(conditioned means are what test this). A read-only conditional analysis on the
cached full-Politics splits confirmed this, so the `--bins` phase was built
(train-fit within-wallet rank01, applied identically to val/test).

For the **level** axis (`exposure = position * price`) the conditional effect is
real, small, and monotone across **every** split and **both** tags:

`mean roi_res` by within-wallet exposure-rank bin (full Politics):

| bin | train | val | test |
|---|---|---|---|
| 0-20 | -0.795 | -0.556 | -0.413 |
| 40-60 | -0.637 | -0.499 | -0.370 |
| 80-100 | **-0.416** | **-0.334** | **-0.277** |

Per-trade copyable Sharpe by bin (full Politics): train +0.06 → +0.50 (top),
val -0.01 → +0.12, test -0.01 → +0.07 — rise at the top on all splits.

**Within-price-quintile** (the confound that killed pooled IC): 80-100 minus
0-20 bin in `mean roi_res` is positive in **all 5 price quintiles on train and
val** (train +0.40/+0.42/+0.37/+0.20/+0.01; val +0.12/+0.03/+0.05/+0.14/+0.02);
test is 4/5 positive. The **marginal** axis does not survive this check — skip.

Wallet-level copyable-Sharpe, top-20% vs rest (Politics, bootstrap 90% CI):

| split | n_wallets | mean diff | CI |
|---|---|---|---|
| train | 83 | +0.713 | +0.53 … +0.93 |
| val | 49 | +0.862 | +0.45 … +1.33 |
| test | 31 | +0.644 | +0.07 … +1.32 |

Weather (2-shard): bin means monotone in `roi_res` on all splits; wallet-level
diff +0.15…+0.30, CI excludes 0 on train/test but **includes 0 on val**.

So: **the conditional signal exists but is small, and the sizing overlay does not
harness it.** The tail-conforming overlay is exactly the existing
`exposure_top80/50/20` (top-80/50/20% of the wallet's own book): on val,
Sharpe/roi_w barely move at top80 and then collapse (top20 val roi_w 0.02,
Sharpe 0.23); test improves Sharpe only by concentrating on fewer trades. The
wallet-level CIs include zero exactly where the decision must be made (val),
so per AGENTS.md fail-fast the idea stays **not promising as a sizing rule**,
though the within-wallet conditional effect (higher relative exposure → higher
residualized ROI) is a defensible, replicable behavior of these wallets.

## Sizing artifact check (unrestricted sim, budget=inf)

The constrained-$10k sim was **skewing the test**: capital is locked from
`dt` to resolution (median 27 days, p90 164) and the sim **skips** any trade
that can't fit the running budget (`fits = used + cost <= budget`); with test
candidate notional ≈ $2.1M vs a $10k budget, only ~0.5% of the book can ever
be funded, and the chronological skip-race selects *cheap* trades regardless of
signal. That race produced the val↔test sign flip. Re-run with
`--unrestricted` (`budget=inf`; every offered trade sized at `alpha * qty`,
depth-capped, proportional to the wallet's own sizing) — this is the honest
"proportional to wallet sizing" evaluation:

| design | val roi_w | val Sharpe | test roi_w | test Sharpe |
|---|---|---|---|---|
| copy_all | 0.0169 | 0.435 | 0.085 | 0.567 |
| exposure_0.8 | 0.031 | 1.539 | 0.156 | 1.598 |
| exposure_0.5 | 0.030 | 1.945 | 0.179 | 1.609 |
| exposure_0.2 | 0.015 | 1.179 | 0.187 | 3.328 |
| exposure_top20 | 0.024 | 1.119 | 0.207 | 2.145 |

**The flip is gone.** Under proportional sizing, tightening exposure no longer
destroys val: `exposure_0.5/0.8` and the `top20` tail beat `copy_all` on *both*
val and test in roi_w and Sharpe (roi_w val 0.03 vs 0.017, test 0.16-0.21 vs
0.085). The earlier "leverage dial" story was dominated by the budget skip-race,
not the signal. The effect is smaller on val than test (high-exposure tail
actually loses to copy_all at the *tightest* cut on val while winning on test),
so it remains partially regime-dependent — but the "don't copy low-exposure
trades" direction is consistent.

Weather (2-shard, unrestricted): same direction but much weaker — `top20` test
roi_w 0.048 (vs copy_all 0.028) and Sharpe 0.508 (vs 0.230), yet val is flat-to-
negative. Smaller n, fundamentals-anchored markets, and the full-Weather load is
>15 min so only the 2-shard sample is used. This matches the hypothesis that
news/opinion-driven Politics should carry more edge-loading per dollar than
fundamentals-driven Weather.

## Event-dominance & ROI diagnostics (recomputed in-kernel, notebook §16)

Two questions the earlier framing was ambiguous on, recomputed from the cached
splits (train-fit capture thresholds, real `capital_constrained_sim` with
`budget=inf`, library 7-day block bootstrap — sanity row matches
`exposure_ci_unrestricted.csv` exactly):

1. **Is the val↔test difference "bad data / few events" on val? No — the
   event-dominance is on *test*.** Top-1-day share of sim PnL: val 0.19-0.41 vs
   test 0.50-0.69; top-3 days: val 0.39-1.14 vs test 0.56-1.47. The
   **Iran/Hormuz cluster is 132% of test `copy_all` PnL** (the non-Iran book nets
   negative on test) vs 77% on val. Val is a well-spread panel — the residual
   val↔test flip is not a "few events on val" artifact.
2. **Does the exposure carve survive removing the dominant cluster? Yes.**
   Delete Iran/Hormuz from test entirely and `copy_all` goes negative
   (roi_w -0.10, Sharpe -0.66) while `exposure_0.5` / `exposure_0.2` stay
   positive (+0.024/+0.104 roi_w, Sharpe +0.61/+4.04). The tail carve is not a
   restatement of "Iran was good".
3. **Is exposure a monotone ROI ranking or a tail/risk carve? Tail/risk carve.**
   Within-wallet exposure-decile `roi_w` is non-monotone (middle deciles worst;
   val dec-0 "lottery" low-exposure inflates the bottom bucket), but **pct_pos
   climbs with decile** on every split (val 0.24 → 0.51; test 0.32 → 0.48) and
   mean `roi_res` is positive at the top on val/test. This is *why* continuous
   exposure IC vs `roi_res` is ~0/negative while the binary PnL-capture tail
   (`thr_0.8`, `top_0.2`) is positively significant on val: the message is
   "drop the value-destroying low-exposure mass", not "rank by exposure". The
   within-wallet construction is required (raw exposure skew ≈ 11-23), but
   `rank01(exposure)` alone is not the usable signal.

Authoritative Sharpe CIs (val, block bootstrap): `copy_all` [-2.40, 2.51];
`exposure_1` [1.57, 4.14]; `exposure_0.8` [1.22, 4.73]; `exposure_0.5`
[1.78, 5.55]; `exposure_0.2` [-1.71, 4.33] — the moderate floors are
significantly positive on val; the extreme cut and `copy_all` include zero.
Test CIs still span zero (the test book is a few big events → high daily
variance), consistent with the cluster analysis above.

## Volatility-neutral target check (`z_vol`, notebook §17)

The user asked whether the proper objective — **profit per unit volatility** —
changes the picture, and whether the ROI target should be de-volatilized on the
**raw** ROI (a studentized residual) rather than on the rank-residualized
`roi_res`. Recomputed in-kernel from the cached splits (train-fit per price
octile, nothing val/test touches the fit):

```
z_vol = (copyable_roi − μ_train(price_bin)) / sd_train(price_bin)     [raw, train-fit, per price octile]
```

1. **It de-confounds price on *both* axes.** `rho(price, ·)`: raw `copyable_roi`
   +0.18/+0.36 (train/val), `roi_res` +0.05/+0.10, `z_vol` +0.07/+0.12. The
   mean effect (`roi_res`) is removed and the scale effect (cheap-token vol,
   cheap-bin SD ≈ 9.5 vs expensive ≈ 0.18) is standardized out.
2. **The exposure carve survives — roughly *doubled*, not killed.** The
   high- vs low-exposure tail delta (top-20% vs bottom-20% within-wallet rank,
   within price octile) is the same sign and ~2x the `roi_res` magnitude on
   every split under `z_vol`, and the high tail beats the low tail in **6–8/8
   price octiles** (val 8/8). A preliminary MAD-scaled version briefly suggested
   the effect vanished; that was a collapsed-MAD artifact, not the data.
3. **The carve *raises* mean `z_vol` of the copied book vs copy_all.** test
   copy_all sits at −0.16 (negative profit-per-vol) and `exposure_{0.8,0.5,0.2}`
   lift it to +0.03…+0.05; on val copy_all is −0.18 and the carve brings it to
   ~0. Under "maximize profit per unit of price-typical risk", filtering out the
   low-exposure mass is the right direction.
4. **Raw vs residualized barely matters here** (same counts of positive bins,
   near-proportional deltas), because within a price bin `roi_res` is a
   near-linear rescale of raw ROI — the *choice* matters only for a **global**
   (cross-price) ranking/selection, where the centered `z_vol` is the honest
   target. Keep `z_vol` as the go target for risk-adjusted objectives;
   `copyable_pnl`/`roi_w` as-is for unlimited-capital total-PnL objectives.
5. **Wallet-selection vs carve in risk-neutral PnL:** under
   `risk_pnl = Σ(copyable_pnl / sd_train(bin))`, `exposure_1` (≈ copy everything
   that train-profitable wallets do — a wallet-quality effect) is best on val
   *and* test (235k/209k vs copy_all 217k/180k); the carve proper (`0.8/0.5`)
   improves test but not val — the doc's "keep the 100%-capture floor as the
   deployment range" caveat is unchanged.

## The 80/50/20/10 ladder (notebook §18)

Notebook cleanup + a dedicated ladder evaluation for the bankroll-discipline
view. Per-wallet capture thresholds fit on train at `frac ∈ {0.8, 0.5, 0.2,
0.1}` (copy only trades whose exposure covers that share of the wallet's train
edge), same unrestricted sim, reported with **wallet_pnl / copyable_pnl /
copyable_roi / roi_w / total traded notional / net PnL / Sharpe** on val and
test.

- **The 80/50 carve is the robust core.** Both beat `copy_all` on roi_w and
  Sharpe on val *and* test; val Sharpe CIs exclude zero (0.8: [1.22, 4.73];
  0.5: [1.78, 5.55]) and they roughly halve the deployed notional (test 2.12M →
  1.18M/0.84M).
- **The signal source confirms the carve — it is not a copy-side artifact.**
  The retained trades carry *higher wallet pnl per trade and higher copyable
  ROI* than copy_all on both splits (val copyable_roi 0.034 → 0.149/0.140;
  test 0.071 → 0.291/0.426). On test the 0.8 carve keeps *more* wallet_pnl and
  copyable_pnl than copy_all (312k/185k vs 300k/183k) while copying only 6,750
  of 48,653 trades — the high-exposure tail really is where the copied
  wallets' edge lives.
- **The 20/10 extreme cuts keep the val↔test asymmetry.** Test roi_w *keeps
  rising* (0.187 / 0.162 vs copy_all 0.085) and Sharpe triples; val roi_w
  collapses back toward `copy_all` (0.015 / 0.005) with CIs including zero.
  The source mirrors this: val copyable_roi at 0.2/0.1 falls back (0.068/0.009)
  while test source ROI stays high (0.405/0.174).
- **The 10% cut is *not* a test artifact — the Iran-excluded control is the
  kicker.** Excluding the dominant cluster, `copy_all` nets −0.101 roi_w
  (−0.66 Sharpe) while `exposure_0.2`/`exposure_0.1` stay **+0.104 / +0.098**
  with bootstrapped Sharpe CIs that *exclude zero* ([0.92, 3.34] and
  [1.25, 3.50]) — the only designs on test statistically distinct from zero.
  Caveat: under Iran exclusion the extreme-tail *per-trade* mean copyable_roi
  is negative (−0.071/−0.067) — a handful of large winning positions carry the
  notional-weighted PnL.
- Use it as a **carve, not a dial**: `exposure_{0.8,0.5}` is the defensible
  production cut; 0.2/0.1 remain research-view pending fresh out-of-sample data.
  The filter never reduces roi_w below `copy_all` on test; the risk is
  concentration (fewer positions, not dilution).

## Verdict (revised after unrestricted sizing + event-dominance diagnostics)

**Promising, now — as a tail/risk carve, not a ranking dial.** The pooled-IC
and constrained-$10k findings were each misleading in a diagnosable way:
(1) pooled rank IC was the wrong instrument for a *conditional* claim — the
within-wallet bin means of `roi_res` rise across every split and both tags
(higher relative exposure → higher residualized ROI); (2) the constrained
skip-race budget artifact caused the apparent "leverage dial" (val↔test flip),
which disappears under `--unrestricted` proportional-to-wallet sizing —
`exposure_0.5`, `exposure_0.8` and `exposure_top20` beat `copy_all` on val
*and* test. (3) The remaining "test much stronger than val" impression is an
event-dominance level effect (Iran/Hormuz = 132% of test copy_all PnL), not a
val data problem — and the carve survives deleting the cluster entirely.

Caveats that keep it from "done":
- The mechanism is a **tail/risk carve**: drop the value-destroying
  low-exposure mass (val dropped half: roi_w -0.10, -$61k) and concentrate on
  trades whose *win rate* rises with within-wallet exposure. It is **not** a
  monotone ROI ranking — continuous exposure IC vs `roi_res` is still ~0 or
  negative (`sig_exp_wt_rank`, `marg`); only the binary threshold/tail forms
  carry the positive residual read.
- The extreme cut (`exposure_0.2`, and now `exposure_0.1`) still flips on val —
  keep `exposure_{0.8,0.5}` / `top20` as the usable range; the §18 Iran-excluded
  control shows 0.2/0.1 are the only designs statistically distinct from zero
  on test, so the research-view is on the *extreme* tail, not the mid cuts.
  On the 2-shard Weather sample the effect is weak/flat on val, so it is
  still partially tag-dependent.
- Test Sharpe CIs include zero (book is a few big events) — treat test point
  estimates as illustrative, not a deployable-edge proof.

Next step if pursued: confirm `exposure_top20`-style tail sizing on
train+val single-pass with cost/depth, then a fresh out-of-sample window.

## Artefacts

- `strategies/exposure_threshold.py` — `ExposureSignals` (raw + wallet-relative
  + marginal + threshold + percentile-top signals).
- `explore_exposure_thresholds.py` — `--ic` / `--capture` / `--bins` CLI,
  `--unrestricted`, `--tags`, `--max-shards`, `--cached-splits-dir`.
- `verify_exposure_thresholds.py` — artefact schema + self-consistency checks.
- `exposure_ic_report_{pnl,roi}.csv`, `exposure_thresholds.csv`,
  `exposure_capture_curve.csv`, `exposure_capture_summary.csv`,
  `exposure_sim.csv`, `exposure_ci.csv`, `exposure_sim_unrestricted{,-Weather}.csv`,
  `exposure_ci_unrestricted{,-Weather}.csv`, `exposure_bins_{exposure,marg}{,-Weather}.csv`,
  `exposure_bins_{exposure,marg}_price{,-Weather}.csv`,
  `exposure_wallet_bins*.csv`, `exposure_wallet_summary*.csv`.
- Notebook section 14 (reproduces the IC panel incl. within-price-bin IC);
  section 15 (unrestricted-sim ladder, read-only from saved CSVs); section 16
  (in-kernel recompute: val+test Sharpe CIs, top-day concentration, Iran/Hormuz
  cluster share, Iran-exclusion control, within-wallet decile ROI); section 17
  (volatility-neutral `z_vol` target: price de-confounding, tail-carve survival,
  mean-`z_vol`/risk-neutral-PnL by design); section 18 (bankroll-discipline
  ladder 80/50/20/10: starts from `copy_all` and `exposure_{0.8,0.5,0.2,0.1}`
  with roi_w / total traded notional / net PnL / Sharpe, plus an Iran-excluded
  control on test).
  The notebook is now **exposure-only**: the earlier `sig_val_opp_flipper`
  walkthrough, composite, and capital-constrained sizing branches were removed
  as dead ends; cells 1-5 (setup + load + wallet metrics) + sections 14-18
  remain.
- Verdict section with the tail/risk-carve mechanism and event-dominance
  diagnostics.