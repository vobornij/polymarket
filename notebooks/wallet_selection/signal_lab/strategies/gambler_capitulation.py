from signal_lab.filters import COPY_DEFAULT, GAMBLER, RETAIL
from signal_lab.signal_engines import UWL_OPP
from signal_lab.strategies.base import DeclarativeStrategy


class GamblerCapitulationSqueeze(DeclarativeStrategy):
    """Test whether a massive underwater position by retail/gamblers on the
    opposite side predicts strong forward returns for the own side.

    Lifetime underwater amount (``uwl``) plus fresh underwater (``fuwl``) over
    a 24h decay for the gambler / retail sets.
    """

    copy_mask = COPY_DEFAULT
    signal_sets = [GAMBLER, RETAIL]
    kinds = [UWL_OPP]
    taus_h = [24]
