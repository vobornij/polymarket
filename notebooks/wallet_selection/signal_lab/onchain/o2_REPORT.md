# O2 — Systematic exploration of Finance/Politics strategies (2026-08-06)

## TL;DR

**Politics**: two simple working strategies, both validated on full data.
**Finance**: a sign-consistent edge exists but the realised PnL is small.

| Tag | Best simple strategy | Test IC | Test PnL on $10k | Sharpe | Notes |
|---|---|---|---|---|---|
| Politics | `price_lt_0p1` (buy every opening BUY at price < 0.1) | +0.126 | **+$154,260** | 0.47 | High-return lottery, low Sharpe |
| Politics | Phase A composite (5 strategies on COPY_DEFAULT) | +0.156 (avg) | **+$8,746** | 2.00 | Disciplined, high Sharpe |
| Finance | `price_gt_0p9` | +0.053 | +$331 | 0.34 | Weak |

Both Politics strategies pass the loose pass bar (`|val IC| > 0.005` same sign on test). Finance has signal but is thin and the realised PnL is small; the strategy needs a different (e.g., external-data) edge to scale.

---

## What was done

Four phases, each gated. Per CLAUDE.md, every step was sample-tested (2 shards) before running on the full data. Verifier: `python -m signal_lab.onchain.verify_o2 --all`.

### Phase A — control (existing composite on COPY_DEFAULT)

Run the existing `evaluate_composite.py`-style pipeline (5 strategies, COPY_DEFAULT selection) per tag, no changes.

| Tag | Train | Val | Test | Gate |
|---|---|---|---|---|
| Finance | equal +0.18 / shrinkage +0.30 | equal −0.08 / shrinkage +0.10 | equal −0.03 / shrinkage +0.07 | Pass *only* for `shrinkage_markowitz` |
| Politics | equal +0.19 / shrinkage +0.35 | equal +0.08 / shrinkage +0.29 | equal +0.07 / shrinkage +0.28 | **Pass** (all schemes positive on test) |

**Finding**: on full data the existing stack **does** transfer to Politics. The O1 diagnosis about rejected-wallet ROI was right (selection is suboptimal in raw terms) but the *composite signal* on the selected sub-universe is positive across splits. The "distortion" matters for **wallet ranking** more than for **composite IC**.

### Phase B — non-copy-trade baseline (ALL_BUYERS mask)

Same 5 strategies, but `copy_mask = ALL_BUYERS` (no quality filter). Per-signal ICs:

| Tag | Top B signal | Train | Val | Test | Same-sign on val+test |
|---|---|---|---|---|---|
| Finance | `sig_uwl_opp_whale` | +0.14 | +0.14 | +0.05 | True |
| Finance | `sig_uwl_opp_retail` | +0.14 | +0.12 | +0.10 | True |
| Finance | `sig_uwl_opp_gambler` | +0.07 | +0.06 | +0.05 | True |
| Politics (2-shard only) | `sig_uwl_opp_retail` | +0.32 | +0.29 | +0.20 | True |
| Politics (2-shard only) | `sig_uwl_opp_whale` | +0.28 | +0.22 | +0.26 | True |

Politics Phase B on full data timed out (1.9M opening BUYs × 15 position signals is too slow for the 600s budget). The 2-shard sample shows the same UWL_OPP family dominates; the family transfers to Politics with high IC.

### Phase C — direct price / lead / market rules

Six simple boolean rules evaluated on the all-buyers frame, Spearman IC vs `roi_res`:

#### Finance (full data)

| Rule | Train | Val | Test | Same sign | Pass? |
|---|---|---|---|---|---|
| `price < 0.5 AND lead_h > 24` | +0.009 | +0.021 | +0.002 | True | yes |
| `price < 0.5 AND lead_h > 72` | +0.012 | +0.002 | −0.020 | False | no |
| `price < 0.1` | +0.033 | −0.038 | +0.112 | False | no |
| `price > 0.9` | +0.018 | +0.079 | +0.053 | True | **yes** |
| `0.3 ≤ price ≤ 0.7 AND lead_h > 24` | −0.007 | +0.047 | +0.009 | True | **yes** |

#### Politics (full data)

| Rule | Train | Val | Test | Same sign | Pass? |
|---|---|---|---|---|---|
| `price < 0.5 AND lead_h > 24` | −0.008 | −0.028 | −0.014 | True | no (negative) |
| `price < 0.5 AND lead_h > 72` | −0.016 | −0.018 | −0.029 | True | no (negative) |
| `price < 0.1` | +0.039 | +0.036 | **+0.126** | True | **yes** |
| `price > 0.9` | −0.023 | +0.006 | +0.038 | True | marginal |
| `0.3 ≤ price ≤ 0.7 AND lead_h > 24` | −0.013 | −0.015 | −0.077 | True | no (negative) |
| Recurring market (≥ 3 distinct trading days) | −0.031 | −0.017 | −0.054 | True | no (negative) |

Politics' `price < 0.1` (long-shot) is the strongest single rule. 5.7% of fired trades win and the asymmetric payout (winners pay ~30x) gives a positive expected value.

### Phase D — combine

Built sign-only composites from top Phase B + Phase C winners; capital-constrained sizing at $10k.

#### Finance (full data)

Composite components: `sig_uwl_opp_whale`, `sig_uwl_opp_retail` (signs +1) + `price_gt_0p9`, `price_mid_lead_gt_24h` (signs +1).

| Split | IC_target | IC_pnl_res | spearman_price | n |
|---|---|---|---|---|
| train | +0.234 | −0.002 | +0.584 | 333,237 |
| val | +0.260 | +0.045 | +0.594 | 522,324 |
| test | +0.184 | −0.047 | +0.527 | 286,454 |

Sizing (val/test): val $1,062 / test $70 / Sharpe 0.17. Composite is **mostly price-driven** (the `price_*` rules contribute the signal, the UWL signals add little). The realised PnL is small.

#### Politics (2-shard sample; full Phase B did not complete in budget)

Composite: `sig_uwl_opp_retail`, `sig_uwl_opp_whale`, `price_mid_lead_gt_24h` (signs +1) + `price_lt_0p5_lead_gt_24h` (sign −1).

| Split | IC_target | IC_pnl_res | spearman_price | n |
|---|---|---|---|---|
| train | +0.350 | +0.083 | +0.684 | 180,939 |
| val | +0.378 | +0.116 | +0.688 | 183,545 |
| test | +0.361 | **+0.126** | +0.638 | 110,507 |

Sizing (val/test): val −$987 / test **+$24,932** / Sharpe 1.79.

The 2-shrinking rule `price_lt_0p5_lead_gt_24h` at sign −1 is the "fade cheap long-shots" component — it's not a single dominant rule (sample-only), so this composite should be re-evaluated on full data before betting on it. The cleaner alternative is the Phase A composite (see below).

---

## Two simple deployable strategies for Politics

### Strategy P-1: "Long-shots on Politics" — `price < 0.1`

```python
# Pseudocode
for opening_buy in politics_markets:
    if opening_buy.price < 0.1:
        take_size_proportional_to_score(0 or 1)  # score = price < 0.1
```

- **Test IC** (Spearman of mask vs `copyable_pnl`): +0.126
- **Capital-constrained sizing on test** ($10k budget): $154,260 net PnL, 102k trades, peak used $10k, daily Sharpe 0.47
- **Why it works**: 5.7% win rate × ~30x payout − 94.3% × small loss = positive EV per share.
- **Risk**: low Sharpe (0.47) because PnL is concentrated in rare winners. Real money you'd allocate to a fraction of the $10k cap (e.g., $1k–$2k) to control variance.
- **Capacity**: with 100k fired trades over a year, a $2k budget would scale ~linearly.

### Strategy P-2: "Existing composite on Politics" — 5 strategies on COPY_DEFAULT

```python
# Same composite that works on Weather, applied to Politics tag.
splits = evaluate_composite(tag='Politics')  # shrinkage_markowitz
# score = signed sum of 5 strategies' val-fit weights
```

- **Test IC_target** (composite vs `copyable_pnl`): +0.07 to +0.28 across schemes; `shrinkage_markowitz` is best.
- **Test IC_pnl_res** (price-controlled): +0.16.
- **Capital-constrained sizing on test** ($10k budget, val-tuned scale 0.3): **$8,746 net PnL, 1,721 trades, daily Sharpe 2.00**.
- **Why it works**: same composite that works on Weather, with `sig_uwl_opp_*` (underwater weak-hand contrarian) being the dominant positive contributor; the existing pipeline's residualisation handles the price confounder.
- **Risk**: small number of trades → need scale or larger candidate universe; val period is the tuning window.

**Recommendation**: deploy P-2 (high Sharpe, controlled variance) as the primary; consider P-1 as a satellite allocation with a smaller budget cap.

---

## What didn't work

- **Finance Phase A**: equal/ic-weighted schemes fail on test; only `shrinkage_markowitz` passes (weakly). The O1 distortion finding is real for the simple selection in Finance.
- **Finance Phase D**: composite is mostly price-driven; realised PnL is small ($70 on test). No deployable strategy under the loose bar.
- **Politics `recurring_market` rule** (drop one-off markets): weakly negative IC on all splits.
- **Politics `price < 0.5 AND lead_h > 24`**: weakly negative on all splits (different from Finance where it was positive).

---

## Caveats and what next

1. **Politics Phase B on full data timed out**. The 2-shard Phase D composite is therefore sample-validated only. Re-running on full data (e.g., 1-hour budget, or restricting to top-K markets by volume) is a follow-up.
2. **Phase D sizing on Finance is small** — under the loose pass bar the edge is there, but the realised PnL is below the noise floor. Finance likely needs external data (intraday price feeds) to extract a deployable strategy.
3. **The `price < 0.1` long-shot strategy on Politics** is the most striking finding. It implies the market systematically over-prices long-shots in Politics markets. Whether this is a real, persistent inefficiency or a one-off pattern tied to the test period is a follow-up question.
4. **No O2 step tested on Weather** — that tag was already validated by the Weather composite (different track). The O2 framework is general; running on Weather is a 5-minute sanity check.
5. **No orthogonal-to-Weather test** — the O2 strategies (P-1, P-2) are uncorrelated with the Weather composite by construction (different tags, no shared trades), so they would be additive in a multi-tag portfolio. A combined-size test is a follow-up.

## Files

- `o2_runner.py` — phases A/B/C/D dispatcher.
- `o2_rules.py` — six direct price/lead/market rules.
- `filters.py` — added `ALL_BUYERS` (no-quality wallet filter, used by Phases B and C).
- `verify_o2.py` — per-phase verifier (`--all` runs everything for both tags).
- `o2_{phase}_{tag}_{artefact}.{csv,json}` — per-step outputs.
- `o2_REPORT.md` — this file.

## How to reproduce

```bash
cd notebooks/wallet_selection
../.venv/bin/python -m signal_lab.onchain.verify_o2 --all

# Re-run a phase (full data):
../.venv/bin/python -m signal_lab.onchain.o2_runner --tag Politics --phase a
../.venv/bin/python -m signal_lab.onchain.o2_runner --tag Finance  --phase a
../.venv/bin/python -m signal_lab.onchain.o2_runner --tag Finance  --phase b
../.venv/bin/python -m signal_lab.onchain.o2_runner --tag Finance  --phase c
../.venv/bin/python -m signal_lab.onchain.o2_runner --tag Politics --phase c
../.venv/bin/python -m signal_lab.onchain.o2_runner --tag Finance  --phase d
```
