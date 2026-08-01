from signal_lab.filters import BOTH_SIDES, COPY_DEFAULT, FLIPPER, OVERSELLER
from signal_lab.signal_engines import POS_OPP, VAL_OPP
from signal_lab.strategies.base import DeclarativeStrategy


class FreshOppositeCrowdingFilter(DeclarativeStrategy):
    """Filter copy-trades so we copy a candidate BUY only when the opposite
    outcome is not heavily and recently crowded by reactive wallet groups.

    Baseline ``val_opp`` crowding plus fresh ``fval_opp`` / ``fpos_opp`` for
    taus in {1, 6, 24}h on the flipper / both_sides / overseller sets. The tau
    is baked into fresh column names (``sig_fval_opp_6h_flipper``).
    """

    copy_mask = COPY_DEFAULT
    signal_sets = [FLIPPER, BOTH_SIDES, OVERSELLER]
    kinds = [VAL_OPP]
    fresh_kinds = [VAL_OPP, POS_OPP]
    taus_h = [1, 6, 24]
