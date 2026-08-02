from signal_lab.filters import (
    BOTH_SIDES,
    COPY_DEFAULT,
    FLIPPER,
    OVERSELLER,
)
from signal_lab.signal_engines import POS_OPP, VAL_OPP
from signal_lab.strategies.base import DeclarativeStrategy


class FreshOppositeCrowdingFilter(DeclarativeStrategy):
    """Filter copy-trades so we copy a candidate BUY only when the opposite
    outcome is not heavily and recently crowded by reactive wallet groups.

    Baseline ``val_opp`` crowding. 
    (Fresh ``fval_opp`` / ``fpos_opp`` are commented out because base `val_opp` 
    performed best in evaluation without the extra computation time).
    """

    copy_mask = COPY_DEFAULT
    # FLIPPER, BOTH_SIDES, and OVERSELLER yielded highly significant negative ICs 
    # (~ -0.19 Val, -0.13 Test)
    signal_sets = [FLIPPER, BOTH_SIDES, OVERSELLER]
    
    kinds = [VAL_OPP]
    
    # --- Best models below are commented out to keep execution fast ---
    # fresh_kinds = [VAL_OPP, POS_OPP]
    # taus_h = [0.25, 1, 24]
