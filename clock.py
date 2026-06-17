"""
clock.py — Time management.

BacktestClock iterates deterministically through a date range.
LiveClock ticks in real-time (production).

All modules call clock.now() instead of datetime.utcnow() so that
the same code runs in both backtest and live without modification.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Iterator, List, Optional


class ClockBase(ABC):
    """Abstract clock interface."""

    @abstractmethod
    def now(self) -> datetime:
        """Return current simulation / wall-clock time."""

    @abstractmethod
    def advance(self) -> bool:
        """
        Move to the next tick.
        Returns False when the clock is exhausted (backtest end).
        """

    @abstractmethod
    def today(self) -> date:
        return self.now().date()


# ──────────────────────────────────────────────
# Backtest Clock
# ──────────────────────────────────────────────

class BacktestClock(ClockBase):
    """
    Deterministic clock that iterates through a sorted list of datetimes.
    Typically driven by trading calendar dates.

    Usage:
        clock = BacktestClock.from_range("2018-01-01", "2023-12-31")
    """

    def __init__(self, ticks: List[datetime]):
        if not ticks:
            raise ValueError("BacktestClock requires at least one tick.")
        self._ticks: List[datetime] = sorted(ticks)
        self._index: int = -1

    # ── Constructors ────────────────────────────

    @classmethod
    def from_range(
        cls,
        start: str,
        end: str,
        freq: str = "1D",
        market_open: str = "09:30",
    ) -> "BacktestClock":
        """
        Build clock from ISO date strings.
        freq: '1D' (daily), '1H' (hourly), '1Min' (minutely).
        """
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end,   "%Y-%m-%d")
        open_h, open_m = map(int, market_open.split(":"))

        delta_map = {"1D": timedelta(days=1), "1H": timedelta(hours=1),
                     "1Min": timedelta(minutes=1)}
        delta = delta_map.get(freq, timedelta(days=1))

        ticks = []
        cur = s.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
        while cur <= e:
            # Skip weekends for daily/hourly
            if freq == "1Min" or cur.weekday() < 5:
                ticks.append(cur)
            cur += delta

        return cls(ticks)

    # ── ClockBase interface ─────────────────────

    def now(self) -> datetime:
        if self._index < 0:
            return self._ticks[0]
        return self._ticks[self._index]

    def today(self) -> date:
        return self.now().date()

    def advance(self) -> bool:
        self._index += 1
        return self._index < len(self._ticks)

    # ── Utility ─────────────────────────────────

    @property
    def total_ticks(self) -> int:
        return len(self._ticks)

    @property
    def progress(self) -> float:
        return self._index / max(1, len(self._ticks) - 1)

    def peek_next(self) -> Optional[datetime]:
        nxt = self._index + 1
        return self._ticks[nxt] if nxt < len(self._ticks) else None

    def __repr__(self) -> str:
        return (
            f"BacktestClock({self._ticks[0].date()} → "
            f"{self._ticks[-1].date()}, "
            f"{len(self._ticks)} ticks, "
            f"at={self.now().date()})"
        )


# ──────────────────────────────────────────────
# Live Clock
# ──────────────────────────────────────────────

class LiveClock(ClockBase):
    """
    Wall-clock time. advance() is a no-op — the engine's main loop
    drives time by sleeping until the next scheduled event.
    """

    def now(self) -> datetime:
        return datetime.utcnow()

    def today(self) -> date:
        return datetime.utcnow().date()

    def advance(self) -> bool:
        # In live mode, we never "run out" of time.
        return True

    def sleep_until(self, target: datetime):
        """Block until wall-clock reaches target."""
        delta = (target - self.now()).total_seconds()
        if delta > 0:
            time.sleep(delta)
