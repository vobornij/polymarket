# Signal Lab

## Purpose

Use this folder for quick copy-trade signal research from public trade data.

Goal: decide whether an idea is promising enough for deeper work.

Not the goal:

- full optimization
- exhaustive sweeps
- heavy processing
- tuning on test

## Subprojects

- `strategies/` — copy-trade signal strategies (existing).
- `weather_fv/` — Track W: weather forecast fair value (see PLAN.md).
- `cross_market/` — Track X: cross-market consistency (see PLAN.md).
- `onchain/` — Track O: bounded Finance/Politics re-validation
  (see PLAN.md).

For new work, prefer the declarative `SignalStrategy` protocol in
`strategies/base.py` and the canonical helpers in `signal_lib.py`.
Each step ships a CLI script and a sibling `verify_<step>.py`; run
`python -m signal_lab.verify_all` to validate the workspace.

## Main Files

- `stage1.py`: functional pipeline — load data, build the candidate universe, run strategies, evaluate signals
- `filters.py`: typed wallet filters (`WalletFilter` objects, `WALLET_FILTERS` registry); strategies reference filter objects directly
- `signal_engines.py`: position-signal math as module functions; `SignalKind` constants and `signal_col_name` name columns
- `signal_lib.py`: IC, residualization, bootstrap, combination, threshold checks
- `example_solution.py`: minimal working example
- `quickstart_signal_lab.ipynb`: notebook entrypoint for manual exploration
- `explore_new_ideas.py`: sequential CLI script for testing signal strategies using `SignalStrategy` protocol
- `strategies/`: Folder containing modular `SignalStrategy` classes for different hypotheses.
- `ideas/*.md`: strategy ideas to evaluate

Useful context:

- `../position_signals.md`
- `../stage1_experimental.ipynb`

## Strategy Protocol

The `signal_lab` supports modular exploration of signal combinations. A strategy is **declarative**: it declares what it wants (copy-wallet filter, wallet-set filters, position kinds, fresh-signal taus) and computes its own signals in `calculate_signals`.

Define a class that inherits from `signal_lab.strategies.base.DeclarativeStrategy` (or implements the `SignalStrategy` protocol) and declare typed objects — **no magic strings**:

- `copy_mask`: a `WalletFilter` object (default `filters.COPY_DEFAULT`). `run_strategy` selects these wallets' BUY trades as the candidate universe.
- `signal_sets`: list of `WalletFilter` objects the signals are computed against (e.g. `filters.FLIPPER`, `filters.BOTH_SIDES`).
- `kinds`: list of `SignalKind` constants (e.g. `signal_engines.VAL_OPP`, `signal_engines.UWL_OPP`).
- `fresh_kinds`: optional list of `SignalKind` attached per tau (defaults to `kinds`); each becomes its `.fresh()` family.
- `taus_h`: fresh-signal decay taus in hours (empty list disables fresh signals).

Fresh column names bake the tau in: `signal_col_name(VAL_OPP.fresh(), 'flipper', tau_h=6)` -> `sig_fval_opp_6h_flipper`. See `strategies/fresh_opposite_crowding.py` and `strategies/gambler_capitulation.py` for examples.

## Workflow

When asked to evaluate `ideas/X.md`:

1. Read the idea and restate the hypothesis.
2. Identify candidate trades, signals to test, and obvious confounders.
3. Start from `example_solution.py`, `quickstart_signal_lab.ipynb`, or implement a new `DeclarativeStrategy` in `strategies/`.
4. Load data once with `load_stage1_data()`.
5. Run the strategy end-to-end with `run_strategy(df_full, wallet_metrics, hold_metrics, strategy)` (rebuilds the candidate universe from the strategy's `copy_mask`, re-splits chronologically, re-residualizes ROI, attaches the declared signal panel).
6. Run the narrowest plausible test first.
7. Use train/validation to decide whether to continue.
8. Only inspect test after something looks promising.

## Guardrails

- Prefer `roi_res` over raw `copyable_roi` unless the idea is explicitly about price.
- Do not attach every signal family if the idea only needs a few.
- Do not sweep many archetypes, taus, and thresholds at once.
- If the first pass is weak, stop quickly and report that.

## What Counts As Promising

- train and validation IC have the same sign
- pooled train+validation bootstrap CI excludes zero
- effect survives basic confound checks
- rough thresholding does not destroy trade count
- test remains directionally consistent

## Minimal Code Pattern

```python
from signal_lab.stage1 import load_stage1_data, run_strategy, evaluate_signal_panel
from signal_lab.strategies import FreshOppositeCrowdingFilter

df_full, df_train, df_val, df_test, wallet_metrics, hold_metrics = load_stage1_data()
strategy = FreshOppositeCrowdingFilter()
splits, cols = run_strategy(df_full, wallet_metrics, hold_metrics, strategy)  # cols include e.g. "sig_fval_opp_6h_flipper"
report, selected = evaluate_signal_panel(splits, cols, roi_col="roi_res")
```

For a single known signal without a strategy, `attach_position_signal_panel(trades, splits, [FLIPPER], kinds=[VAL_OPP], wallet_metrics=wallet_metrics, hold_metrics=hold_metrics)` still works.

## Prompt Template

> Evaluate potential of the strategy described in `ideas/X.md`.
> Use only `signal_lab`.
> Start from the working example or quickstart notebook.
> Run the cheapest plausible test first.
> Do not run exhaustive sweeps or heavy processing.
> Use train/validation to decide whether the idea is promising.
> Only inspect test after something looks real.
