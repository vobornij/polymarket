"""
Wallet set persistence: save and load named wallet cohorts to disk.

Each wallet set is stored as two files:

* ``{set_id}.parquet`` — the wallet DataFrame (wallet, wallet_quality, …)
* ``{set_id}.json``    — a metadata sidecar (id, description, creation time, …)

Usage
-----
.. code-block:: python

    from polymarket_analysis.wallet_selection.persistence import (
        WalletSet, save_wallet_set, load_wallet_set
    )

    ws = WalletSet(
        id="quality_core_v3",
        description="Top-100 by prob_edge_shrunk on full train",
        wallets=selected_wallets,
    )
    save_wallet_set(ws, workspace_dir)
    ws2 = load_wallet_set("quality_core_v3", workspace_dir)
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class WalletSet:
    """A named, versioned set of selected wallets.

    Attributes
    ----------
    id:
        Unique identifier string (used as file stem).
    wallets:
        DataFrame with at least a ``wallet`` column.  Typically also contains
        ``wallet_quality`` and metric columns.
    description:
        Human-readable description of how the set was built.
    metadata:
        Arbitrary key-value pairs stored in the JSON sidecar.
    created_at:
        UTC creation timestamp (set automatically on construction).
    """

    id: str
    wallets: pd.DataFrame
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_wallet_set(ws: WalletSet, directory: Path) -> tuple[Path, Path]:
    """Persist a :class:`WalletSet` to *directory*.

    Creates two files:

    * ``{ws.id}.parquet`` — wallet DataFrame
    * ``{ws.id}.json``    — metadata sidecar

    Parameters
    ----------
    ws:
        The wallet set to persist.
    directory:
        Target directory (created if it does not exist).

    Returns
    -------
    (parquet_path, json_path)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    parquet_path = directory / f"{ws.id}.parquet"
    json_path = directory / f"{ws.id}.json"

    ws.wallets.to_parquet(parquet_path, index=False)

    sidecar = {
        "id": ws.id,
        "description": ws.description,
        "created_at": ws.created_at,
        "num_wallets": len(ws.wallets),
        "columns": list(ws.wallets.columns),
        **ws.metadata,
    }
    json_path.write_text(json.dumps(sidecar, indent=2))

    return parquet_path, json_path


def load_wallet_set(set_id: str, directory: Path) -> WalletSet:
    """Load a :class:`WalletSet` from *directory* by its *set_id*.

    Raises
    ------
    FileNotFoundError
        If ``{set_id}.parquet`` or ``{set_id}.json`` does not exist.
    """
    directory = Path(directory)
    parquet_path = directory / f"{set_id}.parquet"
    json_path = directory / f"{set_id}.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Wallet set parquet not found: {parquet_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Wallet set JSON sidecar not found: {json_path}")

    wallets = pd.read_parquet(parquet_path)
    sidecar = json.loads(json_path.read_text())

    # Extract reserved fields; treat the rest as metadata
    reserved = {"id", "description", "created_at", "num_wallets", "columns"}
    metadata = {k: v for k, v in sidecar.items() if k not in reserved}

    return WalletSet(
        id=sidecar.get("id", set_id),
        wallets=wallets,
        description=sidecar.get("description", ""),
        metadata=metadata,
        created_at=sidecar.get("created_at", ""),
    )


def wallet_set_exists(set_id: str, directory: Path) -> bool:
    """Return ``True`` if both the parquet and JSON sidecar exist."""
    directory = Path(directory)
    return (
        (directory / f"{set_id}.parquet").exists()
        and (directory / f"{set_id}.json").exists()
    )
