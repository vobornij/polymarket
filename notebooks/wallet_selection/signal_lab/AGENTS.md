# Signal Lab

## Purpose

Use this folder for quick copy-trade signal research from public trade data.

Goal: decide whether an idea is promising enough for deeper work.

Not the goal:

- full optimization
- exhaustive sweeps
- heavy processing
- tuning on test

## Main Files

- `stage1.py`: build the copy-universe workspace and evaluate signals
- `signal_engines.py`: construct position-based signals
- `signal_lib.py`: IC, residualization, bootstrap, combination, threshold checks
- `example_solution.py`: minimal working example
- `quickstart_signal_lab.ipynb`: notebook entrypoint for manual exploration
- `ideas/*.md`: strategy ideas to evaluate

Useful context:

- `../position_signals.md`
- `../stage1_experimental.ipynb`

## Workflow

When asked to evaluate `ideas/X.md`:

1. Read the idea and restate the hypothesis.
2. Identify candidate trades, signals to test, and obvious confounders.
3. Start from `example_solution.py` or `quickstart_signal_lab.ipynb`.
4. Build the workspace once.
5. Run the narrowest plausible test first.
6. Use train/validation to decide whether to continue.
7. Only inspect test after something looks promising.

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
from signal_lab.stage1 import build_stage1_workspace, attach_position_signal_panel, evaluate_signal_panel

ws = build_stage1_workspace()
splits, cols = attach_position_signal_panel(
    ws,
    signal_sets={"flipper": ws.signal_sets["flipper"]},
    kinds=[("val", "opp")],
)
report, selected = evaluate_signal_panel(splits, cols)
```

## Prompt Template

> Evaluate potential of the strategy described in `ideas/X.md`.
> Use only `signal_lab`.
> Start from the working example or quickstart notebook.
> Run the cheapest plausible test first.
> Do not run exhaustive sweeps or heavy processing.
> Use train/validation to decide whether the idea is promising.
> Only inspect test after something looks real.
