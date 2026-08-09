# PnL-gate re-evaluation of all implemented ideas (2026-08-03)

Objective changed to **raw pnl (dollars)**. Every coded strategy re-run on the fixed
pipeline and measured by the dollars it produces, in two ways:

1. **Firing gate** (unconstrained): train-fit quantile selection of the edge tail,
   pnl / roi_w vs copy-all on val and test.
2. **Sizing gate** (capital-constrained): walk-forward $10k sim; floor+scale selected
   on train (fold A) or train+val (fold B) by daily Sharpe, reported on val/test.

Benchmarks: **copy-all** and **price-favorite** sizing. Nothing tuned on test.
Source: `reevaluate_ideas_pnl.py`, outputs `ideas_pnl_{ic,firing,sizing,summary}.csv`.

## Verdict: copy-all wins on raw pnl; no signal idea increases dollars

Deployment fold (select train+val -> test), sizing gate:

| idea | best signal | test pnl | vs copy-all | sharpe | roi_w |
|---|---|---|---|---|---|
| copy-all (benchmark) | — | 18,388 | 1.00x | 0.45 | 5.7% |
| price-favorite (benchmark) | — | 18,937 | 1.03x | 0.50 | 5.9% |
| BigWinnerMarketChar | `sig_mkt_price` (= price) | 18,937 | 1.03x (circular) | 0.50 | 5.9% |
| GamblerCapitulation / UwlOppContrarian | `sig_uwl_opp_gambler` | 14,189 | 0.77x | 0.35 | 4.8% |
| FreshOppositeCrowding | `sig_val_opp_both_sides` | 9,174 | 0.50x | 0.23 | 3.3% |
| FadeReactiveSellFlow | `sig_fsf_sell_6h_both_sides` | 7,967 | 0.43x | 0.24 | 2.9% |
| CopyWalletQualitySignals | `sig_wal_trade_count` | 3,198* | 0.17x* | 0.81 | 8.8% |
| CopyCrowdEntryTiming | `sig_ccw_recent_24h_co` | 709 | 0.04x | 3.18 | 12.5% |

\* sizing gate used score-proportional sizing with a Sharpe-selected scale; the firing
gate below shows this signal's real (unconstrained) potential.

Caveats:
- BigWinner's 1.03x is **not an edge**: its "best signal" `sig_mkt_price` is literally
  the trade price, i.e. the price-favorite benchmark under another name (price confound
  again).
- High Sharpe + tiny pnl rows (CopyCrowdEntryTiming 3.18 Sharpe on 709 pnl; CopyWallet
  0.81 on 3,198) are the shrink-into-noise artifact: they win Sharpe by deploying almost
  nothing. Against a **raw-pnl** objective they are failures.

## The one lead: CopyWalletQualitySignals (`sig_wal_trade_count`)

Unconstrained firing gate (threshold = train quantile):

| split | fraction | pnl | roi_w | vs copy-all pnl |
|---|---|---|---|---|
| val | copy-all | −38,141 | −8.0% | 1.00x |
| val | top 25% | **+15,723** | +6.6% | — |
| val | top 10% | +14,131 | +10.2% | — |
| test | copy-all | +22,265 | +3.8% | 1.00x |
| test | top 25% | **+23,841** | +10.0% | **1.07x** |
| test | top 10% | +6,352 | +8.2% | 0.29x |

- Only idea with a **large, sign-consistent dollars edge on both val and test**
  (val flips a −38k loss to +15k; test beats copy-all pnl at 2.5x roi_w).
- Signal = copy-wallet's train-period activity (`wallet_metrics`, train-only, no leak).
  Interpretable rule: **copy only trades of the most active train-period copy wallets.**
- Second-best wallet signal `sig_wal_copyable_pnl` also positive (val +12.8k / test +10.6k).
- FadeReactiveSellFlow `sig_fsf_sell_6h_both_sides` showed 1.02x on test @50% but only
  +1.1k on val (marginal, threshold-fragile) — deprioritized.

## Status

- Dead (do not deploy): FreshOppositeCrowding, CopyCrowdEntryTiming, FadeReactiveSellFlow,
  BigWinnerMarketChar (price-only), GamblerCapitulation/UwlOppContrarian as dollars
  generators (all ≤ copy-all pnl, with val/test sign inconsistency).
- **Needs verification:** CopyWalletQualitySignals. Open questions before deployment:
  1. fraction selected honestly on train+val (0.25 was post-hoc here);
  2. does the edge survive the $10k budget under a **hard top-k full-qty** sizing rule
     (score-proportional sizing was the wrong rule for it);
  3. block-bootstrap Sharpe CI on test daily pnl;
  4. is it just the price/notional composition again (within-price-bin check).

## VERDICT (2026-08-03): CopyWalletQualitySignals is CLOSED — fails the raw-pnl gate

Verification (`pnl_lead_verify.py`, full data, hard top-k full-qty sizing at scale=1.0,
$10k budget, fraction selected on train+val by budget-sim dollars):

| design | val pnl | test pnl | test roi_w | test sharpe |
|---|---|---|---|---|
| copy-all | 2,070 | 9,605 | 4.7% | 0.81 |
| price-top-50 | 68 | 8,149 | 4.2% | 1.02 |
| `sig_wal_trade_count` @0.50 | 1,438 (0.69x) | 7,623 (**0.79x**) | 5.4% | 1.14 |
| `sig_wal_copyable_pnl` @0.50 | 749 (0.36x) | 7,860 (**0.82x**) | 5.8% | 1.22 |

- Honest fraction selection is **degenerate**: budget-sim dollars are monotonically
  increasing in fraction for both signals, both folds — the grid maximum is always 0.50
  (copy-most). The tighter the fraction the better the roi_w (peak ~0.05-0.15) but the
  smaller the deployable capital, so dollars lose. The 0.25/0.10 "edge" from the firing
  gate was post-hoc; at the honest optimum the filter **underperforms copy-all in dollars**.
- Post-hoc test curve confirms it is not a threshold artifact: **no fraction of either
  signal beats copy-all (9,605) in dollars** on test (best 0.82x at 0.50).
- roi_w/Sharpe look better (5.4-5.8% vs 4.7%; 1.14-1.22 vs 0.81) but the 7-day
  block-bootstrap Sharpe CI is [-5.6, +11.9] — statistically indistinguishable — and
  roi_w is not the objective.
- Within-price-bin check (test): no consistent per-bin win; the filter is not a
  price-favorite bet but does not rescue any bin's dollars either.
- `price-top-50` alone beats both wallet signals on test dollars (8,149 > 7,623/7,860).

**Bottom line:** under the raw-pnl objective nothing beats copy-all; the wallet-activity
filter only concentrates roi per dollar at the cost of fewer dollars. Deployable baseline
remains copy-all sizing (~9.6k test pnl on $10k) or price-favorite sizing (8.1k) — the
same conclusion as the crowding overlay. Outputs: `pnl_lead_verify_{results,fractions,
pricebins,ci}.csv`, log in `pnl_lead_verify_full.log`.

## VERDICT (2026-08-03): "drop a small bad tail" — CLOSED. The bad tail is dollar-positive

Tested the one filtering regime that *could* win dollars: instead of concentrating on a
top fraction (which always loses coverage), drop a small tail of candidates ranked by
**bad conditions** — underwater-add severity (`sig_ua_underwater_usdc` < 0, copy wallet
adding to its own losing position) x fresh copy-crowd on the opposite outcome
(`sig_fval_opp_6h_copy_default`). Hard full-qty $10k sim, floors from train (A) or
train+val (B), val + test. New strategy `strategies/underwater_add_crowding.py` (first
signal built from the candidate wallet's OWN pre-buy position, `signal_engines` per-wallet
checkpoints), diagnostic `tail_drop_diag.py`, outputs `tail_drop_{tails,sim,ci,pricebins}.csv`.

Test (deployment fold), drop = top X% removed:

| ranking | drop 2% | drop 5% | drop 10% | drop 20% | copy-all |
|---|---|---|---|---|---|
| underwater-add (`bad_ua`) | **9,428** (1.00x) | 9,407 | 9,220 | 8,127 | 9,400 |
| opposite-crowd (`bad_cc`) | 8,520 (0.91x) | 8,513 | 7,962 | 5,468 | 9,400 |
| composite | 9,314 | 9,233 | 8,695 | 7,478 | 9,400 |

Why it fails — the hypothesized "bad" tail carries **positive dollars** forward:
- Underwater-add tail: test tail pnl +7 to +1,337 (positive); its IC **flips sign**
  train −0.018 → val +0.048 / test +0.061. Underwater-adds were bad on train (why the
  hypothesis seemed plausible) but good forward — non-stationarity, not a cullable effect.
- Opposite-crowd tail: test tail pnl +882 to +3,988 with roi_w up to 64% at mean_price
  0.21–0.41 — the **cheap/deep-favorite tail again**. Stable negative roi_res IC
  (−0.09/−0.10/−0.07) yet positive dollars: the clearest demonstration yet that crowding
  IC is per-share, not monetizable in dollars.
- Val "wins" at drop=20% (composite 1.7x, bad_cc 1.55x copy-all) do **not** replicate on
  test (0.80x, 0.58x) — val-period noise (~4 weeks).
- The two hoped-for mechanisms (avoided losses, freed budget) never materialize: `mean_used`
  stays ~2.3k and the tail is not dollar-negative. Post-hoc best (bad_ua drop 2%) adds
  $28 (+0.3%) — noise; identical bootstrap CI as copy-all.

**Bottom line:** even the small-tail filter cannot beat copy-all in dollars, because the
trades copy wallets make under adverse conditions (underwater, crowded-opposite) ARE the
cheap-favorite dollar-makers. All implemented ideas are now closed. Deployable baseline
remains copy-all sizing (test 9.4k on $10k, 0.81 Sharpe) or price-favorite sizing (8.1k).

All implemented ideas are now closed as dollars-generators. The only positive carry-over
for future work: per-dollar-efficiency signals (wallet activity, crowding, gambler) do
reliably raise roi_w; if the objective ever becomes **efficiency under a capital cap**
rather than raw dollars, those filters become the right inputs to revisit.

## Per-wallet copy sizing (alpha_w): first selector to beat copy-all on BOTH folds (2026-08-03)

Objective: **Sharpe of the sim's daily resolution-pnl, fixed $10k budget** (per user).
Each copy-default wallet gets an `alpha_w` (0 = skip, can exceed 1, capped by the
reconstructed share-depth cap `bucket_avail_copy_qty`). Alphas fit on **train only**;
per-scheme hyperparameters (tier count, alpha_max, alpha floor, uniform k) selected on
**val by sim Sharpe**; test is a single pass. Source: `wallet_scaling.py`,
`wallet_scaling_diag.py`; preprocessing `depth_cap.py` (bucket-level `avail_copy_qty_5m_100`
rebuilt from enriched shards — the processed files only kept a summed, over-counted
`avail_copy_total_vol_5m_100`). Universe: 58 copy-default wallets, val/test ~52k/~42k candidate
BUYs.

Selected config per scheme (deployed, mean-1 normalized alphas):

| scheme | config | val pnl | val Sharpe | test pnl | test Sharpe | test roi_w |
|---|---|---|---|---|---|---|
| copy-all (benchmark) | alpha=1 | 1,853 | 0.138 | 9,400 | 0.814 | 0.0458 |
| shrunk max-Sharpe (Kelly) | `kelly@2` | 5,049 | 0.441 | **7,707** | 0.740 | 0.0439 |
| **tier 3@2-0** (alpha 1.93/0.97/**0**) | winner | **6,689** | **0.703** | **10,818** | **1.397** | **0.0653** |
| uniform k = 0.5 (leverage-only) | val-picked | 927 | 0.138 | 4,700 | 0.814 | 0.0458 |

- `tier3@2-0` = 3 equal tiers by **train per-event Sharpe proxy**: top tier ~2x size,
  middle ~1x, bottom **alpha=0 (dropped)**. Selected on val, wins on **both** folds:
  test +15% dollars (10,818 vs 9,400), Sharpe 0.81 → 1.40, roi_w 4.6% → 6.5%; val +261%
  dollars (1,853 → 6,689), Sharpe 0.14 → 0.70. Bootstrap CI (test, 7-day blocks) is wide
  (tier3@2-0 [−2.2, +8.8], copy-all [−6.3, +7.8]) — evidence is the **two-fold
  consistency + placebo**, not the CI.
- **Placebo control (decisive):** dropping a *random* 1/3 of wallets (5 seeds) gives val
  pnl 0.6–3.2k / Sharpe ≤0.27 and test pnl 2.2–8.8k / Sharpe ≤1.18. The train-selected
  drop (val 5.3k / 0.68, test 9.1k / 1.31) is far better on both folds → the selection,
  not budget relief, is the signal.
- **Decomposition:** ~90% of the win is the **drop of the bottom tier**, ~10% the 2x top
  tier. Drop-only vs copy-all: val 5,304 / 0.681 vs 1,853 / 0.138; test 9,121 / 1.306 vs
  9,400 / 0.814 (drop-only is a roi_w/Sharpe win; the +$ on test comes from the 2x scale:
  11,091 raw / 10,818 deployed).

Why it works (the mechanism, and why it's NOT the old failure modes):
- The dropped tier is the **high-activity, higher-price, near-zero-margin** group: ~60% of
  candidate trade count (test 24,343 of 42,247) but ~0% of pnl (test $340; val **−$3,357**),
  mean price ~0.51–0.55 vs ~0.41 for kept tiers, per-trade pnl ~0.01 (test) / −0.10 (val)
  vs 0.43–0.68 / 0.05–0.74. Dropping them cuts notional ~30–44% while holding pnl — the
  first intervention that beats the old "notional cuts faster than roi_w rises" curse.
- **Not the cheap-favorite artifact:** the dropped tier is the *higher*-priced group, and
  the kept top tier beats copy-all per-trade pnl in 6 of 8 mid-price deciles (price
  controlled) — unlike the crowding tail, this survives the price-bin check.
- **Leverage alone doesn't help:** uniform k>1 on the full universe *hurts* (val uniform@2
  Sharpe 0.019, uniform@4 −0.18; budget+costs dominate). The depth cap binds hard (median
  depth/copyable = 1.00; 48% of top-tier trades depth-capped), so scale>1 only matters on
  the own-limited headroom — the dollar gain is real but second-order.
- Kelly (continuous weights) is a **val-only** win (val 0.441 vs 0.138) that fails on test
  dollars (7,707 < 9,400) — the continuous scheme over-fits the sparse per-wallet series;
  the coarse tier discretization is the robust form.

Honest caveats: 58-wallet universe (bottom tier = 13–18 wallets); config val-selected
from a 6-outcome effective grid (all floor=0 configs beat copy-all on val: Sharpe
0.57–0.70, so it's a consistent pattern, not one lucky cell); wide CI; single test window.

**Verdict: PROMISING, not proven — first wallet-level selector to transfer to both folds.
Deployable candidate: tier by train per-event Sharpe proxy, drop bottom 1/3 (alpha=0),
~2x top 1/3, cap at share depth** (test 10.8k on $10k, 1.40 Sharpe, roi_w 6.5%). Next
step if pursuing: confirm on a longer/out-of-sample window and stress the wallet-set
stability before treating it as the new baseline.
