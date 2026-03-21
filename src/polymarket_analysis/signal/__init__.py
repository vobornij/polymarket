"""Signal building and scoring package."""

from polymarket_analysis.signal.builder import (
    build_wallet_profiles,
    attach_consensus_features,
    build_signal_events,
    verify_partial_fill_grouping,
)
from polymarket_analysis.signal.scorer import (
    build_calibration_tables,
    apply_signal_score,
    select_signal_threshold,
)

__all__ = [
    # builder
    "build_wallet_profiles",
    "attach_consensus_features",
    "build_signal_events",
    "verify_partial_fill_grouping",
    # scorer
    "build_calibration_tables",
    "apply_signal_score",
    "select_signal_threshold",
]
