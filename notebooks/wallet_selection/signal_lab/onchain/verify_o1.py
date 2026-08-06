"""O1 verify — sanity checks for the finance/politics distortion diagnostic.

Reads onchain/o1_diagnosis.json and asserts:
- all three tags (Weather, Finance, Politics) have a row
- each row has the expected metric set
- candidate BUY counts are non-zero
- price_outcome_ic in (0.30, 0.80) (sanity bound)
- top1pct_wallet_share < 0.5 (no single wallet dominates)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE / "o1_diagnosis.json"

EXPECTED_TAGS = ["Weather", "Finance", "Politics"]
EXPECTED_KEYS = {
    "tag", "n_markets", "n_opening_buys", "n_wallets_total",
    "n_wallets_selected", "wallet_selection_rate",
    "selected_wallet_buy_roi_mean", "rejected_wallet_buy_roi_mean",
    "n_candidate_buys", "market_concentration_p99_over_med",
    "top1pct_wallet_share_of_candidates", "one_off_market_share",
    "near_resolution_price_share", "price_outcome_ic", "lead_h_outcome_ic",
}


def main() -> int:
    diag = json.loads(DIAG.read_text())
    for tag in EXPECTED_TAGS:
        assert tag in diag, f"missing tag in diagnosis: {tag}"
        row = diag[tag]
        assert EXPECTED_KEYS.issubset(row.keys()), (
            f"missing keys for {tag}: {EXPECTED_KEYS - set(row.keys())}"
        )
        assert row["n_candidate_buys"] > 100, (
            f"{tag}: too few candidate trades ({row['n_candidate_buys']})"
        )
        assert 0.30 < row["price_outcome_ic"] < 0.80, (
            f"{tag}: price_outcome_ic out of range ({row['price_outcome_ic']})"
        )
        assert row["top1pct_wallet_share_of_candidates"] < 0.5, (
            f"{tag}: top1pct_wallet_share too high ({row['top1pct_wallet_share_of_candidates']})"
        )
    print(json.dumps(diag, indent=2))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
