# Idea: Big-Winner Market-Condition Characterization

## Summary

Copy PnL is violently concentrated: the top 1% of candidate BUYs produce **1.83x**
total copy PnL, the top 50 markets produce **83%** of it, and 68% of candidate
trades have PnL <= 0. If we could cheaply identify the *kind of market* that yields
big copy winners, filtering candidates to those markets would dominate any per-trade
tweak. This idea characterizes the market-condition at the moment of the candidate
BUY with trade-frame features (no position engine needed).

## Hypothesis

Big-winner candidate trades (top-1% PnL) are a distinct population: they are **late**
(enter after the market is already active), **higher-priced** (~0.35 vs ~0.25 for
first-movers), and **crowded**. That is, the copy edge lives in already-"discovered"
markets, not fresh ones — the opposite of the first-mover edge from Idea 1 (which
acts per-trade within a market). Market-level state should predict candidate ROI
above and beyond the per-trade crowd signals.

## Candidate Trades

- Base universe: existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First (trade-frame, at candidate time t)

- `sig_mkt_age_h` — hours since this condition's first trade (young vs mature market).
- `sig_mkt_trades_before` — cumulative trade count on the condition before t.
- `sig_mkt_notional_before` — cumulative copyable notional on the condition before t.
- `sig_mkt_price` — current mid/price level of the bought outcome (levels ~0.3-0.5?).
- `sig_mkt_price_mom_6h` — price change of the outcome over the last 6h (momentum).
- `sig_mkt_n_outcomes` — number of distinct outcomes on the condition.
- `sig_mkt_uncert` — dispersion / entropy of outcome prices (uncertainty proxy).

## Expected Outcome

Mixed: age / activity should be **positive** predictors of ROI (big winners are late
entrants), momentum and price-level directions are exploratory. Any strong, stable
feature becomes a market-level gate for the copy filter.

## Findings

**Status: Not promising as standalone.** Market-state features at candidate time do
not yield a stable predictor of `roi_res`. Full-data IC vs `roi_res` (train / val /
test):

| signal | IC_train | IC_val | IC_test | verdict |
|---|---|---|---|---|
| `sig_mkt_age_h` | +0.184 | -0.041 | +0.103 | sign flip train/val — unstable |
| `sig_mkt_price` | -0.131 | -0.249 | +0.380 | sign flip on test — unstable |
| `sig_mkt_trades_before` | -0.109 | -0.182 | -0.044 | consistent but weak |
| `sig_mkt_pmom_6h` | -0.105 | -0.147 | +0.225 | sign flip on test — unstable |
| `sig_mkt_notional_before` | -0.077 | -0.022 | -0.093 | consistent but weak |
| `sig_mkt_n_outcomes` | NaN | -0.003 | NaN | no signal |

The grounding finding that big winners are *late* entrants does not translate into a
robust per-trade market-state edge. The only directionally-stable features
(`trades_before`, `notional_before`, both negative — more active market → worse
residualized ROI) essentially recapture the copy-crowding effect already covered by
`CopyCrowdEntryTiming` (`n_cond`), and are far weaker.

Stop: do not pursue as a separate family. The per-trade crowd / sell-flow signals
(Ideas 1-2) remain the operative mechanisms.
