from signal_lab.filters import COPY_DEFAULT, GAMBLER, RETAIL, WHALE
from signal_lab.signal_engines import UWL_OPP
from signal_lab.strategies.base import DeclarativeStrategy


class GamblerCapitulationSqueeze(DeclarativeStrategy):
    """Test whether a massive underwater position by retail/gamblers on the
    opposite side predicts strong forward returns for the own side.

    Lifetime underwater amount (``uwl``). 
    (Fresh underwater ``fuwl`` commented out because the baseline lifetime amount
    yielded the same strong positive IC).
    """

    copy_mask = COPY_DEFAULT
    # GAMBLER and RETAIL yield strong positive ICs (+0.06 to +0.09 Val/Test).
    # WHALE is kept as a reference for negative correlation (whales underwater = bad).
    signal_sets = [GAMBLER, RETAIL, WHALE]
    
    kinds = [UWL_OPP]
    
    # --- Best models below are commented out to keep execution fast ---
    # fresh_kinds = [UWL_OPP]
    # taus_h = [1, 24, 72]
