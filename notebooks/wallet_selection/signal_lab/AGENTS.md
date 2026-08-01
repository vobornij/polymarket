# Signal Lab

## Motivation

This folder isolates the small part of the repo that matters for fast copy-trade signal research.

Use it when the task is:

- start from a strategy idea described in `ideas/*.md`
- evaluate whether the idea looks promising on public trade data
- do a quick, disciplined exploration without heavy processing or broad parameter sweeps

The goal is not to fully optimize or productionize a strategy here. The goal is to decide whether an idea is promising enough to deserve deeper work.

## What Lives Here

- `stage1.py`: reusable stage-1 workspace and evaluation helpers
- `signal_engines.py`: signal construction, especially archetype position signals
- `signal_lib.py`: IC, residualization, bootstrap, combination, threshold evaluation
- `ideas/*.md`: strategy hypotheses to evaluate

Useful adjacent context outside this folder:

- `../position_signals.md`: prior findings on crowding and price confounding
- `../stage1_experimental.ipynb`: current exploratory notebook based on the same framework

## Core Rules

1. Start from the idea file, not from the whole codebase.
2. Prefer cheap tests first.
3. Use train/validation to decide whether the idea is worth pursuing.
4. Only look at test after a direction seems promising.
5. Do not run exhaustive sweeps across every archetype, signal family, tau, threshold, and variant.
6. Do not fine-tune on test.
7. If the first quick evidence is weak, stop and report that clearly.

## Current Methodology

The current framework is a copy-trade filter on candidate BUY trades:

1. Build a candidate universe of copyable BUY trades from historically decent wallets.
2. Attach public-data signals to those candidate trades.
3. Evaluate signal strength on forward `copyable_roi`, but first remove the strong price-level confounder using `roi_res`.
4. Keep only signals that are directionally consistent on train and validation and have a pooled bootstrap CI away from zero.
5. If needed, combine a small number of promising signals and do a rough threshold check.

Important caveat:

- Raw ICs can be badly confounded by candidate price.
- Use residualized ROI unless the idea is explicitly about price itself.
- The stage-1 framework is for rough selection/filtering, not a full execution backtest.

## Recommended Workflow

When asked to evaluate `ideas/X.md`, do this:

1. Read `X.md` and restate the hypothesis in one paragraph.
2. Identify:
   - candidate trades
   - signal columns or signal family to test
   - likely confounders
3. Build or reuse the workspace:

```python
from signal_lab.stage1 import build_stage1_workspace, summarize_workspace

ws = build_stage1_workspace()
summarize_workspace(ws)
```

4. Start with the narrowest plausible test.

Example:

```python
from signal_lab.stage1 import attach_position_signal_panel, evaluate_signal_panel

splits, cols = attach_position_signal_panel(
    ws,
    signal_sets={"flipper": ws.signal_sets["flipper"]},
    kinds=[("val", "opp")],
)
report, selected = evaluate_signal_panel(splits, cols)
```

5. If the first pass is promising, test only a few nearby variants.

Examples of nearby variants:

- add one or two adjacent archetypes
- compare `pos` vs `val`
- test a small recency family (`fresh_tau_ns`) such as 1h, 6h, 24h
- compare own vs opposite outcome

6. If a small set of signals survives, optionally build a simple composite:

```python
from signal_lab.stage1 import build_composite_scores, evaluate_threshold_grid

scored_splits, weights, _ = build_composite_scores(splits, selected)
val_grid = evaluate_threshold_grid(
    scored_splits["val"],
    "composite_shrinkage_markowitz",
    cost_bps=0.0,
)
```

7. Only then inspect the test split for a rough holdout estimate.

## Guardrails On Cost And Runtime

- Avoid rebuilding the full workspace repeatedly.
- Avoid attaching all signal families if the idea only needs 1-3 of them.
- Avoid testing many tau values or many archetypes unless the base signal already works.
- Avoid heavy parameter searches.
- Prefer a rough estimate of potential over exhaustive tuning.

## What Counts As Promising

An idea is worth another iteration if most of these hold:

- train and validation IC have the same sign
- pooled train+validation bootstrap CI excludes zero
- the effect survives basic confound checks
- rough thresholding improves validation economics without collapsing trade count
- the result still looks directionally sensible on test

## What To Report

For each evaluated idea, report:

1. hypothesis
2. candidate trade definition
3. exact signals tested
4. quick train/validation evidence
5. rough test estimate if warranted
6. main confounders or failure modes
7. recommendation: stop, iterate narrowly, or escalate

## Prompt Template

Use a prompt like:

> Evaluate potential of the strategy described in `ideas/X.md`.
> Use the `signal_lab` framework only.
> Start with the cheapest plausible test.
> Do not run exhaustive sweeps or heavy processing.
> Use train/validation to decide if the idea is promising.
> Only inspect test after something looks real.
> If the first pass is weak, stop quickly.
> If it is promising, test only a small number of nearby variants and provide a rough estimate of potential.

## Notes On Prior Findings

- Opposite-token and total crowding signals have looked more promising than many raw entry-premium signals.
- Recency-weighted fresh-position signals were not fully re-evaluated after earlier signal-framework fixes, so they are a good candidate for focused follow-up.
