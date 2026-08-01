from typing import List, Protocol
import pandas as pd

from signal_lab.filters import WalletFilter
from signal_lab.signal_engines import SignalKind, signal_col_name
from signal_lab.stage1 import attach_position_signal_panel


class SignalStrategy(Protocol):
    """Declarative protocol for a signal strategy.

    A strategy declares *what* it wants -- the copy-wallet filter that defines
    the candidate universe, the wallet filters the signals are computed
    against, the position kinds, and the fresh-signal decay taus -- and
    *computes* its own signals in ``calculate_signals``. Everything is typed:
    wallet sets are ``WalletFilter`` objects, position kinds are ``SignalKind``
    constants (no magic strings).
    """

    @property
    def name(self) -> str:
        """Unique strategy name (used for reporting/logging)."""
        ...

    @property
    def copy_mask(self) -> WalletFilter:
        """Copy-wallet filter.

        ``run_strategy`` selects these wallets' BUY trades as the candidate
        universe, re-splits it chronologically, and re-residualizes ROI on the
        resulting training split before calling ``calculate_signals``.
        """
        ...

    @property
    def signal_sets(self) -> List[WalletFilter]:
        """Wallet filters the signals are computed against."""
        ...

    @property
    def kinds(self) -> List[SignalKind]:
        """Base position kinds (see ``signal_engines.ALL_POSITION_KINDS``)."""
        ...

    @property
    def fresh_kinds(self) -> List[SignalKind] | None:
        """Fresh families attached per tau; ``None`` defaults to ``kinds``.

        Each ``kind`` becomes its ``.fresh()`` family with the tau baked into
        the column name (``sig_fval_opp_6h_flipper``).
        """
        ...

    @property
    def taus_h(self) -> List[int]:
        """Fresh-signal decay taus in hours. Empty list disables fresh signals."""
        ...

    def get_signal_columns(self) -> List[str]:
        """Column names this strategy attaches, matching ``signal_col_name``."""
        ...

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Attach signals to the candidate splits and return the mutated dict.

        Strategies may also filter candidates (e.g. drop rows) in place.
        """
        ...


class DeclarativeStrategy:
    """Default implementation shared by concrete strategies.

    Subclasses declare ``copy_mask``, ``signal_sets``, ``kinds``,
    ``fresh_kinds`` and ``taus_h``; everything else is derived from those
    declarations.
    """

    copy_mask: WalletFilter
    signal_sets: List[WalletFilter] = []
    kinds: List[SignalKind] = []
    fresh_kinds: List[SignalKind] | None = None
    taus_h: List[int] = []

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_signal_columns(self) -> List[str]:
        fresh = self.fresh_kinds if self.fresh_kinds is not None else self.kinds
        cols: List[str] = []
        for flt in self.signal_sets:
            for kind in self.kinds:
                cols.append(signal_col_name(kind, flt.name))
            for tau_h in self.taus_h:
                for kind in fresh:
                    cols.append(signal_col_name(kind.fresh(), flt.name, tau_h=tau_h))
        return cols

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if not self.signal_sets:
            return splits
        return attach_position_signal_panel(
            trades,
            splits,
            self.signal_sets,
            kinds=self.kinds,
            fresh_kinds=self.fresh_kinds,
            taus_h=self.taus_h,
            wallet_metrics=wallet_metrics,
            hold_metrics=hold_metrics,
        )[0]
