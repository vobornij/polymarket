"""Wallet-set interaction exploitation: contrarian opposite-side UWL_OPP.

Exploits the differential sign of opposite-side crowding by wallet archetype.
Reactive groups (FLIPPER / BOTH_SIDES / OVERSELLER) crowding the opposite outcome
predict worse ROI (negative, captured by ``FreshOppositeCrowdingFilter``), while
underwater opposite-side holdings by weak hands (GAMBLER / RETAIL) — ``uwl_opp`` —
are expected to predict *better* ROI (contrarian edge).

Signals use the position-signal engine: ``sig_uwl_opp_<set>`` (underwater USDC on the
opposite outcome) and ``sig_val_opp_<set>`` (value-at-cost on the opposite outcome,
for contrast).
"""

from __future__ import annotations

from signal_lab.filters import GAMBLER, RETAIL, COPY_DEFAULT, WalletFilter
from signal_lab.signal_engines import UWL_OPP, VAL_OPP
from signal_lab.strategies.base import DeclarativeStrategy


class UwlOppContrarian(DeclarativeStrategy):
    """Contrarian opposite-side underwater holdings by weak-hand wallets."""

    copy_mask: WalletFilter = COPY_DEFAULT
    signal_sets: list[WalletFilter] = [GAMBLER, RETAIL]
    kinds: list = [UWL_OPP, VAL_OPP]
