"""
Copy-trading wallet group selection.

Provides four wallet groups for a copy-trading strategy:

* **openers** — strong copyable openers, always copied on BUY
* **leaders** — wallets that precede profitable follower BUYs
* **followers** — wallets with high copyable ROI liquidity
* **closers** — wallets with positive PnL, copy their SELLs

Quick start::

    from polymarket_analysis.copy_groups import build_wallet_groups
    groups = build_wallet_groups(dataset, end_date=train_end)
    openers, leaders, followers, closers = (
        groups["openers"], groups["leaders"],
        groups["followers"], groups["closers"],
    )
"""

from __future__ import annotations

from .leader_follower import (
    aggregate_leader_follower_pairs,
    detect_leader_follower_pairs,
    preselect_followers,
    rank_leaders,
)
from .wallet_groups import (
    build_wallet_groups,
    select_closers,
    select_openers,
)

__all__ = [
    "build_wallet_groups",
    "preselect_followers",
    "detect_leader_follower_pairs",
    "aggregate_leader_follower_pairs",
    "rank_leaders",
    "select_openers",
    "select_closers",
]
