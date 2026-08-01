from .base import DeclarativeStrategy, SignalStrategy
from .gambler_capitulation import GamblerCapitulationSqueeze
from .fresh_opposite_crowding import FreshOppositeCrowdingFilter

__all__ = [
    "SignalStrategy",
    "DeclarativeStrategy",
    "GamblerCapitulationSqueeze",
    "FreshOppositeCrowdingFilter",
]