"""
Strategy registry: save, load, and list persisted WalletSelectionStrategy objects.

Each strategy is stored in a workspace directory as two files:

* ``{strategy_id}.parquet``    — the wallet DataFrame
* ``{strategy_id}.meta.json``  — JSON sidecar with all other fields

Usage
-----
.. code-block:: python

    from polymarket_analysis.strategy.registry import (
        save_strategy, load_strategy, load_all_strategies
    )

    save_strategy(ws, workspace_dir)
    strategy = load_strategy("skill_prob_edge_top50", workspace_dir)
    all_strategies = load_all_strategies(workspace_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .definition import TriggerSpec, WalletSelectionStrategy


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_strategy(
    strategy: WalletSelectionStrategy,
    directory: Path,
) -> tuple[Path, Path]:
    """Persist a :class:`WalletSelectionStrategy` to *directory*.

    Creates:

    * ``{strategy_id}.parquet``   — wallet DataFrame
    * ``{strategy_id}.meta.json`` — JSON sidecar

    Returns
    -------
    (parquet_path, json_path)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    parquet_path = directory / f"{strategy.strategy_id}.parquet"
    json_path = directory / f"{strategy.strategy_id}.meta.json"

    strategy.wallets.to_parquet(parquet_path, index=False)

    sidecar = {
        "strategy_id": strategy.strategy_id,
        "name": strategy.name,
        "selection_method": strategy.selection_method,
        "trigger": strategy.trigger.to_dict(),
        "params": strategy.params,
        "metadata": strategy.metadata,
        "created_at": strategy.created_at,
        "num_wallets": len(strategy.wallets),
        "wallet_columns": list(strategy.wallets.columns),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, default=str))

    return parquet_path, json_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_strategy(strategy_id: str, directory: Path) -> WalletSelectionStrategy:
    """Load a :class:`WalletSelectionStrategy` from *directory* by *strategy_id*.

    Raises
    ------
    FileNotFoundError
        If either the parquet or the JSON sidecar is missing.
    """
    directory = Path(directory)
    parquet_path = directory / f"{strategy_id}.parquet"
    json_path = directory / f"{strategy_id}.meta.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Strategy parquet not found: {parquet_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Strategy JSON sidecar not found: {json_path}")

    wallets = pd.read_parquet(parquet_path)
    sidecar = json.loads(json_path.read_text())

    return WalletSelectionStrategy(
        strategy_id=sidecar["strategy_id"],
        name=sidecar.get("name", strategy_id),
        selection_method=sidecar.get("selection_method", ""),
        trigger=TriggerSpec.from_dict(sidecar["trigger"]),
        wallets=wallets,
        params=dict(sidecar.get("params", {})),
        metadata=dict(sidecar.get("metadata", {})),
        created_at=sidecar.get("created_at", ""),
    )


def load_all_strategies(directory: Path) -> dict[str, WalletSelectionStrategy]:
    """Load every strategy found in *directory*.

    Returns
    -------
    ``{strategy_id → WalletSelectionStrategy}`` ordered by strategy_id.
    """
    directory = Path(directory)
    strategy_ids = sorted(
        p.name.removesuffix(".meta.json")
        for p in directory.glob("*.meta.json")
    )
    return {sid: load_strategy(sid, directory) for sid in strategy_ids}


def strategy_exists(strategy_id: str, directory: Path) -> bool:
    """Return ``True`` if both files for *strategy_id* exist in *directory*."""
    directory = Path(directory)
    return (
        (directory / f"{strategy_id}.parquet").exists()
        and (directory / f"{strategy_id}.meta.json").exists()
    )
