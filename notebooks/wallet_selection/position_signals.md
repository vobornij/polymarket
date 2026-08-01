# Archetype Position Signals — Exploration Notes

Exploration of generic aggregate-position signals per wallet archetype, applied to the
pre-selected copy universe (Weather tag). Selection = sign-consistent IC on train AND val
with `|IC| >= 0.005`, presence `>= 0.005`; test IC is out-of-sample diagnostics only.

> **CRITICAL: 2026-08-01 — `_rankdata` bug fixed.** `signal_lib._rankdata` returned
> `rank[inv]` instead of `rank` (a wrong final permutation that scrambled ranks whenever
> ties exist — i.e. always, since `copyable_roi` is degenerate). Every IC computed through
> `signal_lib` before this fix is invalid: the position-signal selection below (old
> `position_report.csv`), the `val_opp` drill-down verdict, and the fresh-family sweep.
> After the fix, ICs match scipy. **The old "signals are weak / not robust" conclusion was
> wrong.** Raw corrected ICs are large, then turn out to be mostly a *candidate-price
> confounder*; after price residualization a robust negative "crowdedness" family remains.
> Details in the sections below.

## Signal families

For each archetype set, at each candidate BUY trade at time `t`, the **exact aggregate
open position** of the set on the candidate's token (`A(t) - B(t)` two-pass cumsum over
post-trade checkpoints, validated against brute force):

| family | columns | definition |
|---|---|---|
| `pos`   | `sig_pos_{own,opp,total}` | aggregate quantity held |
| `val`   | `sig_val_{own,opp,total}` | aggregate value-at-cost (USDC), average-cost accounting |
| `avgc`  | `sig_avgc_{own,opp}` | `val/pos/price - 1` — entry premium vs current price (own at `price`, opp at `1-price`) |
| `uwl`   | `sig_uwl_{own,opp}` | `val - pos*price` — USDC underwater on the position |
| `fpos/fval/favgc/fuwl` | same x `{own,opp}` | recency-weighted: `exp(-age/tau)` decay, so recently-entered positions count more (`fresh_tau_ns` build; tau in {1h,6h,24h}) |

Every family is always attached and evaluated on the candidate's OWN outcome and on the
OPPOSITE outcome (`_opp`) — the "always test the opposite outcome" rule.

## Archetype sets (train, min_trade_count=100)

whale 133, retail 213, gambler 82, overseller 437, overseller_deep 54, overseller_thin 25,
consistent 31, max_dd 579, both_sides 186, scalper 70, flipper 161.

## Step 1 — corrected raw ICs (before price control)

With the fixed rankdata, pooled ICs jump 10-50x and are sign-consistent across splits
(train/val/test), e.g.:

| signal | IC_train | IC_val | IC_test |
|---|---|---|---|
| `sig_uwl_own_flipper` | -0.253 | -0.281 | -0.280 |
| `sig_uwl_own_retail` | -0.260 | -0.291 | -0.267 |
| `sig_avgc_own_flipper` | -0.267 | -0.250 | -0.259 |
| `sig_avgc_opp_flipper` | +0.233 | +0.265 | +0.251 |
| `sig_uwl_opp_flipper` | +0.228 | +0.275 | +0.281 |
| `sig_val_own_flipper` | +0.270 | +0.265 | +0.258 |

Daily IR on these was 2-3 (e.g. `uwl_own_flipper` -2.84, negative **every** day). This
initially looked like a breakthrough.

## Step 2 — the confounder: candidate price

`IC(candidate BUY price, copyable_roi) ≈ +0.47 / +0.52 / +0.51` (train/val/test). The
strong signals correlate with price (`rho(sig, price)` 0.4-0.6), and pooled IC within 20
price quantile bins collapses to ~0 and flips sign across splits. The raw ICs were mostly
a price-level proxy (favorite effect: high-price candidates win more often).

## Step 3 — price-residualized evaluation (the method)

To measure a signal's information **beyond price** (price is freely observable), we
residualize forward ROI against price and re-measure IC on the residual:

1. Van der Waerden scores: fractional ranks of `copyable_roi` and `price` mapped to
   standard normals — robust to the mass-at--1.0 ROI and heavy outliers.
2. OLS `rank_roi ~ beta * rank_price + intercept`, fitted on **train only**.
3. The fixed `beta`/`intercept` are applied to val and test (no refit, no leakage).
   Residual `eps = rank_roi - beta*rank_price - intercept`.
4. Signal IC = Spearman(signal, eps); daily IR likewise.

Implemented in `signal_lib` (`rank_scores`, `fit_roi_residualizer`, `residualized_roi`)
and run for all 11 archetypes in `resid_eval.py`
(→ `/tmp/pos_explore_cache/position_report_resid.csv`, `resid_attached_*.parquet`).
Caveat: this removes only the linear-in-ranks price component; out-of-sample
`IC(price, eps)` is +0.08 (val) / +0.12 (test) — small but nonzero.

## Step 4 — residualized results: a robust negative "crowdedness" family

After removing price, the `pos`/`val` aggregate-position families are **robustly
negative** predictors of forward ROI across most active archetypes (both_sides, scalper,
flipper, overseller*, max_dd, retail, gambler). Top rows (all sign-consistent on all three
splits):

| signal | presence | IC_train | IC_val | IC_test |
|---|---|---|---|---|
| `sig_val_opp_both_sides` | 0.98 | -0.177 | -0.220 | -0.134 |
| `sig_val_total_retail` | 0.97 | -0.165 | -0.169 | -0.117 |
| `sig_pos_opp_both_sides` | 0.98 | -0.164 | -0.144 | -0.043 |
| `sig_pos_opp_max_dd` | 0.98 | -0.160 | -0.175 | -0.109 |
| `sig_val_opp_flipper` | 0.91 | -0.154 | -0.206 | -0.148 |
| `sig_val_opp_max_dd` | 0.98 | -0.153 | -0.207 | -0.163 |
| `sig_val_opp_overseller` | 0.98 | -0.147 | -0.206 | -0.161 |
| `sig_val_own_retail` | 0.91 | -0.147 | -0.133 | -0.056 |
| `sig_val_own_scalper` | 0.85 | -0.141 | -0.138 | -0.077 |
| `sig_val_total_flipper` | 0.96 | -0.140 | -0.186 | -0.108 |

Daily IR on residuals for these is -1.4 to -2.6 with ~95-100% of days negative. The sign
is *opposite* the residual price leakage (+0.08/+0.12), so this is genuine incremental
information, not leftover price.

**Interpretation**: conditional on price, tokens where active wallets hold large
aggregate positions/value-at-cost underperform — a "crowded / mean-reversion" effect.

**Within-token robustness**: demeaning signal and residual within `condition_id`
(fixed-effects Spearman) keeps the effect: `val_opp_both_sides` -0.143/-0.160/-0.078,
`val_opp_flipper` -0.088/-0.143/-0.104, `pos_opp_max_dd` -0.122/-0.136/-0.095,
`val_total_retail` -0.088/-0.130/-0.089. Not a between-token artifact.

**What did NOT survive**: the `uwl`/`avgc` families (underwater / entry premium) — the
price proxies. Their residualized ICs are ~0 and flip sign; `avgc_opp` retains only a
weak within-bin residual (-0.05..-0.08, sign-consistent but near noise).

**Fresh family** (`fpos/fval/...`, recency-weighted): was only ever evaluated with the
buggy rankdata → all numbers invalid. Has not yet been re-tested on residuals; the plain
`pos`/`val` families are already strong, so recency weighting is a follow-up, not a
priority.

## Honest read / next steps

- The real signal after price control is the **negative aggregate-position family**
  (`pos`/`val`, esp. `_opp` on both_sides/flipper/overseller/max_dd and `total` on
  retail/scalper). Candidate for combination with price (two orthogonal axes: favorite
  effect + crowding mean-reversion).
- Not yet checked: overlap/orthogonality among the selected `pos`/`val` signals (likely
  highly collinear — same underlying aggregate, different archetypes), strategy eval
  (does trading AGAINST crowding add PnL net of the price effect?), and the fresh family
  on residuals.

## Non-fit / not pursued (see also oversell_signals.md)

- Proximity signals (bad-leader, quality-wallet) removed: tested earlier, no stable edge
  (note: their ICs were also computed pre-bug-fix; revisit if reused).
- Wallet-set proximity grid sweep: unstable, removed in favor of the archetype sweep.
- Tier robustness tests and leave-one-out contribution: dropped from the clean notebook.

## Cache

- `/tmp/pos_explore_cache/position_report_resid.csv` — full residualized IC report (11 archetypes x 14 signal kinds).
- `/tmp/pos_explore_cache/resid_attached_{train,val,test}.parquet` — candidate frames with all signals + roi_res.
- `/tmp/pos_explore_cache/position_report.csv`, `valopp_attached_*.parquet` — **pre-bug-fix, invalid**.
