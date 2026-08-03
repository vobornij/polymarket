# Idea: Wallet-Set Interaction Exploitation (UWL_OPP Contrarian)

## Summary

The same "opposite-side crowding" signal has **different signs depending on which
wallet group provides the flow**. Reactive groups (FLIPPER / BOTH_SIDES / OVERSELLER)
crowding the opposite outcome predict worse candidate ROI (negative IC — already
captured by `FreshOppositeCrowdingFilter` via `val_opp`). But opposite-side pain held
by **weak hands (GAMBLER / RETAIL)** — underwater holdings, `uwl_opp` — reportedly has
**positive** IC: gamblers/retail trapped on the other side are a contrarian edge.

This idea exploits that wallet-set interaction directly.

## Hypothesis

- `sig_uwl_opp_gambler` and `sig_uwl_opp_retail` > 0 (positive IC on `roi_res`).
- `uwl_opp` > `val_opp` for these sets: the *loss-weighted* opposite-side crowding is
  what carries the edge (underwater = capitulation risk, not just positioning).

## Candidate Trades

- Base universe: existing stage-1 candidate BUY trades from the copy-wallet universe.

## Signals To Test First

- `sig_uwl_opp_gambler`, `sig_uwl_opp_retail` — positive direction expected.
- `sig_val_opp_gambler`, `sig_val_opp_retail` — contrast (same sets, value-at-cost).

## Expected Outcome

Positive, sign-consistent IC for the `uwl_opp` pair; if it holds, this adds a
contrarian value leg that combines with the reactive-crowding fade (Idea 2).

## Findings

**Status: Confirmed.** The wallet-set interaction is real and exploitable: for the
same weak-hand sets, opposite-side crowding measured by value-at-cost (`val_opp`) is
negative while underwater opposite-side holdings (`uwl_opp`) are positive.

Full-data IC vs `roi_res` (train / val / test):

| signal | IC_train | IC_val | IC_test | direction |
|---|---|---|---|---|
| `sig_uwl_opp_retail` | +0.018 | +0.062 | +0.099 | contrarian (+), monotone ↑ |
| `sig_uwl_opp_gambler` | +0.017 | +0.048 | +0.068 | contrarian (+), monotone ↑ |
| `sig_val_opp_retail` | -0.156 | -0.161 | -0.137 | fade (-) |
| `sig_val_opp_gambler` | -0.093 | -0.043 | -0.041 | fade (-) |

- All four significant (pooled train+val bootstrap CIs exclude zero), sign-consistent.
- `uwl_opp` is weaker in magnitude than the sell-flow / crowd signals but is the only
  **positive** family found so far and strengthens across splits — good diversification.
- `val_opp` for these same sets is strongly negative, so raw opposite-side positioning
  is bad; only the *underwater* component flips sign. This is a distinct mechanism
  from `FreshOppositeCrowdingFilter` (which uses FLIPPER/BOTH_SIDES/OVERSELLER).

Copy direction: `uwl_opp` is a **positive** signal — candidate BUYs on outcomes where
GAMBLER / RETAIL hold underwater opposite-side positions are favored.
