# Politics O2 — exploration report

This folder is the Politics-specific workspace for the O2 systematic
exploration of non-Weather strategies. The shared infrastructure
(`o2_runner.py`, `o2_rules.py`, `verify_o2.py`, `filters.py::ALL_BUYERS`)
lives in the parent `signal_lab/onchain/` directory.

## What's here

- `politics_o2.ipynb` — the entry-point notebook. Run the cells in
  order. Refreshes all data and produces the headline plots.
- `politics_d.py` — Phase D variant that combines the Phase A
  `shrinkage_markowitz` composite with the `price_lt_0p1` rule (avoids
  the slow full-universe `run_strategies` that timed out on Politics).
- `o2_a_politics_*.{csv,json}` — Phase A composite ICs and summary.
- `o2_b_politics_*.{csv,json}` — Phase B (all-buyers baseline, sample
  only).
- `o2_c_politics_*.{csv,json}` — Phase C rule ICs.
- `o2_d_politics_*.{csv,json}` — Phase D composite + rule ICs and
  sizing.
- `o2_d_politics_daily_pnl.csv` — daily PnL timeseries for the
  composite and the rule on val/test (used by the notebook's plots).

## Reproduce

```bash
cd notebooks/wallet_selection
../.venv/bin/python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.kernel_name=polymarket \
    signal_lab/onchain/politics/politics_o2.ipynb
```

Or run the phases individually:

```bash
../.venv/bin/python -m signal_lab.onchain.o2_runner \
    --tag Politics --phase a --out-dir signal_lab/onchain/politics
../.venv/bin/python -m signal_lab.onchain.o2_runner \
    --tag Politics --phase b --max-shards 2 \
    --out-dir signal_lab/onchain/politics
../.venv/bin/python -m signal_lab.onchain.o2_runner \
    --tag Politics --phase c --out-dir signal_lab/onchain/politics
../.venv/bin/python -m signal_lab.onchain.politics.politics_d
../.venv/bin/python -m signal_lab.onchain.verify_o2 \
    --tag Politics --all --out-dir signal_lab/onchain/politics
```

The Jupyter kernel is registered as `polymarket` (uses the project's
`.venv`).

## Headline

| Strategy | Test IC (vs `roi_res`) | Test PnL on $10k | Sharpe (daily) |
|---|---|---|---|
| **`price_lt_0p1`** (single rule) | +0.126 | **+$153,974** | 0.47 |
| Phase A composite (5 strategies on COPY_DEFAULT) | +0.16 (price-controlled) | +$8,746 | 2.00 |

Both pass the O2 loose gate (`|val IC| > 0.005` same sign on test).
The two strategies are uncorrelated with each other and with the
Weather composite (different tags, no shared trades), so they are
additive in a multi-tag portfolio.

### Notes

- The `copyable_pnl` IC for `price_lt_0p1` is **negative** because
  long-shots have small dollar PnL per share (most lose $0.01–0.05,
  a few win $5–100). The per-dollar-allocated return is positive
  (0.84 vs 0.08 for unfired), and the realized PnL is the source of
  truth. Use `roi_res` (price-residualized ROI) for the IC metric.
- The composite underperforms the rule alone in the test split. The
  Phase A `shrinkage_markowitz` score on Politics captures price-driven
  information that's already implicit in the rule's price filter, so
  adding it on top adds noise. P-1 is the headline; P-2 is the
  complementary, lower-variance strategy.
- See `signal_lab/onchain/o2_REPORT.md` for the cross-tag comparison
  (Finance was tested in the same loop but the realized PnL is small
  — external data is the missing piece for Finance).
