"""
universe.py — Universe selection and management.

Controls which symbols are active at any given time.
Mirrors Lean's IUniverseSelectionModel.

Selectors:
  StaticUniverse        — Fixed symbol list, never changes
  FilteredUniverse      — Applies screening rules on each rebalance date
  MomentumRankedUniverse— Select top-N by momentum score
  SP500ProxyUniverse    — Curated mega-cap proxy (demo)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Callable, Dict, Iterable, List, Optional, Set

from ..engine.events import UniverseChangeEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class UniverseSelectorBase(ABC):
    """
    Interface for universe management.

    The engine calls select() once per bar; the selector decides
    whether this bar triggers a rebalance and, if so, returns a
    UniverseChangeEvent describing what to add / remove.
    """

    def __init__(self):
        self._symbols: Set[str] = set()

    @property
    def current_symbols(self) -> List[str]:
        return sorted(self._symbols)

    @abstractmethod
    def select(
        self,
        timestamp: datetime,
        data,          # DataHandlerBase (for filtering by metrics)
    ) -> Optional[UniverseChangeEvent]:
        """
        Called every bar.  Return a UniverseChangeEvent if the
        universe changed, None otherwise.
        """

    def apply_change(self, event: UniverseChangeEvent):
        """Called by engine after routing UniverseChangeEvent."""
        self._symbols.update(event.added)
        self._symbols.difference_update(event.removed)
        logger.info(
            f"Universe updated: +{len(event.added)} / -{len(event.removed)} "
            f"→ {len(self._symbols)} total symbols"
        )

    def _build_change_event(
        self, new_set: Set[str], timestamp: datetime
    ) -> Optional[UniverseChangeEvent]:
        added   = new_set - self._symbols
        removed = self._symbols - new_set
        if not added and not removed:
            return None
        return UniverseChangeEvent(
            added=sorted(added),
            removed=sorted(removed),
            timestamp=timestamp,
        )


# ──────────────────────────────────────────────
# 1. Static Universe
# ──────────────────────────────────────────────

class StaticUniverse(UniverseSelectorBase):
    """
    Fixed set of symbols.  Applied once on the first bar.
    No changes thereafter.
    """

    def __init__(self, symbols: Iterable[str]):
        super().__init__()
        self._target = set(symbols)
        self._initialised = False

    def select(self, timestamp, data) -> Optional[UniverseChangeEvent]:
        if not self._initialised:
            self._initialised = True
            return self._build_change_event(self._target, timestamp)
        return None


# ──────────────────────────────────────────────
# 2. Scheduled Filtered Universe
# ──────────────────────────────────────────────

class FilteredUniverse(UniverseSelectorBase):
    """
    Rebalances the universe on a schedule, applying a user-supplied
    filter function to a candidate pool.

    Args:
        candidate_pool : All symbols to consider
        filter_fn      : (symbol, data) → bool — return True to include
        rebalance_freq : 'daily' | 'weekly' | 'monthly'
    """

    VALID_FREQ = {"daily", "weekly", "monthly"}

    def __init__(
        self,
        candidate_pool: List[str],
        filter_fn: Callable[[str, object], bool],
        rebalance_freq: str = "monthly",
    ):
        super().__init__()
        if rebalance_freq not in self.VALID_FREQ:
            raise ValueError(f"rebalance_freq must be one of {self.VALID_FREQ}")

        self.candidate_pool  = candidate_pool
        self.filter_fn       = filter_fn
        self.rebalance_freq  = rebalance_freq
        self._last_rebalance: Optional[date] = None

    def select(self, timestamp, data):
        today = timestamp.date()

        if not self._should_rebalance(today):
            return None
        self._last_rebalance = today

        new_set = set(
            sym for sym in self.candidate_pool
            if self.filter_fn(sym, data)
        )
        return self._build_change_event(new_set, timestamp)

    def _should_rebalance(self, today: date) -> bool:
        if self._last_rebalance is None:
            return True
        if self.rebalance_freq == "daily":
            return today > self._last_rebalance
        if self.rebalance_freq == "weekly":
            return today.isocalendar()[1] != self._last_rebalance.isocalendar()[1]
        if self.rebalance_freq == "monthly":
            return (today.year, today.month) != (
                self._last_rebalance.year, self._last_rebalance.month
            )
        return False


# ──────────────────────────────────────────────
# 3. Momentum-Ranked Universe
# ──────────────────────────────────────────────

class MomentumRankedUniverse(UniverseSelectorBase):
    """
    Monthly rebalance: select top-N symbols by their (lookback - skip)
    momentum return from the candidate pool.
    """

    def __init__(
        self,
        candidate_pool: List[str],
        top_n:          int  = 50,
        lookback:       int  = 252,
        skip:           int  = 21,
        rebalance_freq: str  = "monthly",
    ):
        super().__init__()
        self.candidate_pool  = candidate_pool
        self.top_n           = top_n
        self.lookback        = lookback
        self.skip            = skip
        self.rebalance_freq  = rebalance_freq
        self._last_rebalance: Optional[date] = None

    def select(self, timestamp, data):
        today = timestamp.date()
        if not self._should_rebalance(today):
            return None
        self._last_rebalance = today

        scores: Dict[str, float] = {}
        for sym in self.candidate_pool:
            s = data.series(sym) if data else None
            if s is None or len(s) < self.lookback + self.skip:
                continue
            closes = s.closes(self.lookback + self.skip)
            if len(closes) >= self.lookback + self.skip:
                scores[sym] = closes[-(self.skip + 1)] / closes[0] - 1.0

        if not scores:
            return None

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        new_set = set(top)
        return self._build_change_event(new_set, timestamp)

    def _should_rebalance(self, today: date) -> bool:
        if self._last_rebalance is None:
            return True
        if self.rebalance_freq == "monthly":
            return (today.year, today.month) != (
                self._last_rebalance.year, self._last_rebalance.month
            )
        return today > self._last_rebalance


# ──────────────────────────────────────────────
# 4. Liquidity-Screened Universe
# ──────────────────────────────────────────────

class LiquidityScreenedUniverse(UniverseSelectorBase):
    """
    Keeps only symbols that meet a minimum average daily volume (ADV)
    and minimum price threshold.  Rebalances monthly.
    """

    def __init__(
        self,
        candidate_pool:  List[str],
        min_adv_dollars: float = 5_000_000,   # $5M avg daily value traded
        min_price:       float = 5.0,          # exclude penny stocks
        adv_lookback:    int   = 21,            # bars for ADV calc
        rebalance_freq:  str   = "monthly",
    ):
        super().__init__()
        self.candidate_pool  = candidate_pool
        self.min_adv_dollars = min_adv_dollars
        self.min_price       = min_price
        self.adv_lookback    = adv_lookback
        self.rebalance_freq  = rebalance_freq
        self._last_rebalance: Optional[date] = None

    def select(self, timestamp, data):
        today = timestamp.date()
        if not self._should_rebalance(today):
            return None
        self._last_rebalance = today

        new_set = set()
        for sym in self.candidate_pool:
            px = data.price(sym) if data else None
            if px is None or px < self.min_price:
                continue
            s = data.series(sym) if data else None
            if s is None or len(s) < self.adv_lookback:
                continue
            closes  = s.closes(self.adv_lookback)
            volumes = s.volumes(self.adv_lookback)
            adv     = float((closes * volumes).mean())
            if adv >= self.min_adv_dollars:
                new_set.add(sym)

        return self._build_change_event(new_set, timestamp)

    def _should_rebalance(self, today: date) -> bool:
        if self._last_rebalance is None:
            return True
        if self.rebalance_freq == "monthly":
            return (today.year, today.month) != (
                self._last_rebalance.year, self._last_rebalance.month
            )
        return today > self._last_rebalance
