import nbformat


def _modify(nb):
    # Cell 1: drop the removed Stage1Workspace import.
    for i, cell in enumerate(nb.cells):
        if "from signal_lab.stage1 import Stage1Workspace" in cell.source:
            cell.source = cell.source.replace(
                "from signal_lab.stage1 import Stage1Workspace\n", ""
            )
            print(f"Removed Stage1Workspace import in cell {i}")

    # Cell 14 Markdown
    nb.cells[14].source = """## 6. Build the Signals using Strategies

We build a restricted trade frame (the position-checkpoint input) and use our modular `SignalStrategy` objects to attach the signals directly to the candidate splits."""

    # Cell 15 Code: functional signal attachment, no workspace object.
    nb.cells[15].source = """restricted = df_full[df_full['condition_id'].isin(conditions)][ENGINE_COLS].copy()
candidate_splits = {"train": c_train, "val": c_val, "test": c_test}

# Instantiate our strategies
strategies = [
    GamblerCapitulationSqueeze(),
    FreshOppositeCrowdingFilter()
]

for strategy in strategies:
    print(f"Running strategy: {strategy.name}")
    candidate_splits = strategy.calculate_signals(
        candidate_splits,
        trades=restricted,
        wallet_metrics=wallet_metrics,
        hold_metrics=hold_metrics,
    )

# The splits have been updated (strategies return fresh copies)
c_train, c_val, c_test = candidate_splits["train"], candidate_splits["val"], candidate_splits["test"]

display(c_train.head(5))
"""

    # Ensure SIGNAL_COL points at a fresh-signal column the strategies attach.
    for i, cell in enumerate(nb.cells):
        if "SIGNAL_COL = 'sig_val_opp_flipper'" in cell.source:
            cell.source = cell.source.replace(
                "SIGNAL_COL = 'sig_val_opp_flipper'",
                "SIGNAL_COL = 'sig_fval_opp_24h_both_sides'",
            )
            print(f"Updated SIGNAL_COL in cell {i}")


def modify_notebook():
    for path in (
        "notebooks/wallet_selection/signal_lab/quickstart_signal_lab.ipynb",
        "notebooks/wallet_selection/signal_lab/quickstart_signal_lab_out.ipynb",
    ):
        with open(path, "r") as f:
            nb = nbformat.read(f, as_version=4)
        _modify(nb)
        with open(path, "w") as f:
            nbformat.write(nb, f)
        print(f"Patched {path}")


if __name__ == "__main__":
    modify_notebook()
