"""
data_handler.py — Historical & live data management.

Responsibilities:
  • Load / stream OHLCV data per symbol
  • Maintain rolling bar history (lookback window)
  • Serve bars to alpha models and execution engine
  • Abstract over CSV / database / vendor API sources

Design mirrors Lean's SubscriptionDataConfig + DataManager.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..engine.events import MarketEvent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Bar Container
# ──────────────────────────────────────────────

class BarSeries:
    """
    Fixed-length deque of MarketEvents for one symbol.
    Provides numpy array views for fast alpha computation.
    """

    def __init__(self, symbol: str, maxlen: int = 500):
        self.symbol = symbol
        self._bars: deque[MarketEvent] = deque(maxlen=maxlen)

    # ── Write ─────────────────────────────────

    def push(self, bar: MarketEvent):
        self._bars.append(bar)

    # ── Read ──────────────────────────────────

    def __len__(self) -> int:
        return len(self._bars)

    def latest(self) -> Optional[MarketEvent]:
        return self._bars[-1] if self._bars else None

    def get(self, n: int = 1) -> List[MarketEvent]:
        """Return last n bars, oldest first."""
        bars = list(self._bars)
        return bars[-n:]

    # ── Numpy helpers ─────────────────────────

    def closes(self, n: Optional[int] = None) -> np.ndarray:
        bars = self.get(n) if n else list(self._bars)
        return np.array([b.adj_close for b in bars])

    def opens(self, n: Optional[int] = None) -> np.ndarray:
        bars = self.get(n) if n else list(self._bars)
        return np.array([b.open for b in bars])

    def highs(self, n: Optional[int] = None) -> np.ndarray:
        bars = self.get(n) if n else list(self._bars)
        return np.array([b.high for b in bars])

    def lows(self, n: Optional[int] = None) -> np.ndarray:
        bars = self.get(n) if n else list(self._bars)
        return np.array([b.low for b in bars])

    def volumes(self, n: Optional[int] = None) -> np.ndarray:
        bars = self.get(n) if n else list(self._bars)
        return np.array([b.volume for b in bars])

    def returns(self, n: Optional[int] = None) -> np.ndarray:
        c = self.closes(n)
        return np.diff(c) / c[:-1] if len(c) > 1 else np.array([])

    def log_returns(self, n: Optional[int] = None) -> np.ndarray:
        c = self.closes(n)
        return np.diff(np.log(c)) if len(c) > 1 else np.array([])

    def sma(self, period: int) -> Optional[float]:
        c = self.closes(period)
        return float(np.mean(c)) if len(c) == period else None

    def ema(self, period: int) -> Optional[float]:
        c = self.closes(period)
        if len(c) < period:
            return None
        weights = np.exp(np.linspace(-1.0, 0.0, period))
        weights /= weights.sum()
        return float(np.dot(weights, c))

    def atr(self, period: int = 14) -> Optional[float]:
        bars = self.get(period + 1)
        if len(bars) < period + 1:
            return None
        trs = []
        for i in range(1, len(bars)):
            h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(trs[-period:]))

    def rsi(self, period: int = 14) -> Optional[float]:
        r = self.returns(period + 1)
        if len(r) < period:
            return None
        gains = np.where(r > 0, r, 0.0)
        losses = np.where(r < 0, -r, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def zscore(self, period: int = 20) -> Optional[float]:
        c = self.closes(period)
        if len(c) < period:
            return None
        std = np.std(c, ddof=1)
        return float((c[-1] - np.mean(c)) / std) if std > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "timestamp": b.timestamp,
                "open":      b.open,
                "high":      b.high,
                "low":       b.low,
                "close":     b.close,
                "adj_close": b.adj_close,
                "volume":    b.volume,
            }
            for b in self._bars
        ]
        return pd.DataFrame(rows).set_index("timestamp")


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class DataHandlerBase(ABC):
    """All data sources implement this interface."""

    def __init__(self, lookback: int = 500):
        self.lookback = lookback
        self._series: Dict[str, BarSeries] = {}

    # ── Abstract ──────────────────────────────

    @abstractmethod
    def get_latest_bars(
        self, symbols: Iterable[str], timestamp: datetime
    ) -> List[MarketEvent]:
        """Return next batch of bars for the given symbols at this timestamp."""

    # ── Shared implementation ─────────────────

    def update_bar(self, event: MarketEvent):
        """Called by the engine after dispatching a MarketEvent."""
        if event.symbol not in self._series:
            self._series[event.symbol] = BarSeries(event.symbol, self.lookback)
        self._series[event.symbol].push(event)

    def series(self, symbol: str) -> Optional[BarSeries]:
        return self._series.get(symbol)

    def bar(self, symbol: str) -> Optional[MarketEvent]:
        s = self._series.get(symbol)
        return s.latest() if s else None

    def price(self, symbol: str) -> Optional[float]:
        b = self.bar(symbol)
        return b.adj_close if b else None

    def has_bars(self, symbol: str, n: int) -> bool:
        s = self._series.get(symbol)
        return s is not None and len(s) >= n

    def symbols(self) -> List[str]:
        return list(self._series.keys())


# ──────────────────────────────────────────────
# CSV Data Handler
# ──────────────────────────────────────────────

class CSVDataHandler(DataHandlerBase):
    """
    Loads per-symbol CSV files (date-indexed OHLCV).
    Supports daily and intraday data.

    Expected CSV columns (case-insensitive):
        date/timestamp, open, high, low, close, adj_close, volume

    Usage:
        handler = CSVDataHandler("/data/equities/")
        handler.load_symbols(["AAPL", "MSFT", "GOOGL"])
    """

    COLUMN_MAP = {
        "date": "timestamp", "datetime": "timestamp", "time": "timestamp",
        "adj close": "adj_close", "adjusted close": "adj_close",
        "adj_close": "adj_close", "close": "close",
    }

    def __init__(
        self,
        data_dir: str,
        lookback: int = 500,
        date_col: str = "date",
        delimiter: str = ",",
        date_format: Optional[str] = None,
    ):
        super().__init__(lookback)
        self.data_dir    = data_dir
        self.date_col    = date_col
        self.delimiter   = delimiter
        self.date_format = date_format

        # symbol → sorted list of (datetime, row_dict)
        self._raw: Dict[str, List[Tuple[datetime, dict]]] = {}
        # current pointer per symbol
        self._cursor: Dict[str, int] = {}

    def load_symbols(self, symbols: Iterable[str]):
        """Pre-load CSV files for the given symbols."""
        for sym in symbols:
            path = os.path.join(self.data_dir, f"{sym}.csv")
            if not os.path.exists(path):
                logger.warning(f"No CSV found for {sym} at {path}")
                continue
            self._load_file(sym, path)
        logger.info(
            f"CSVDataHandler loaded {len(self._raw)} symbols "
            f"from {self.data_dir}"
        )

    def _load_file(self, symbol: str, path: str):
        df = pd.read_csv(path, delimiter=self.delimiter)
        df.columns = [c.strip().lower() for c in df.columns]

        # Normalise column names
        for alias, canon in self.COLUMN_MAP.items():
            if alias in df.columns and canon not in df.columns:
                df.rename(columns={alias: canon}, inplace=True)

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format=self.date_format, infer_datetime_format=True
        )
        df.sort_values("timestamp", inplace=True)

        # If adj_close not present, fall back to close
        if "adj_close" not in df.columns:
            df["adj_close"] = df.get("close", 0.0)

        rows = df.to_dict("records")
        self._raw[symbol] = [(r["timestamp"], r) for r in rows]
        self._cursor[symbol] = 0

    def get_latest_bars(
        self, symbols: Iterable[str], timestamp: datetime
    ) -> List[MarketEvent]:
        events = []
        for sym in symbols:
            if sym not in self._raw:
                continue
            bars = self._raw[sym]
            cur = self._cursor[sym]

            # Advance past rows before this timestamp
            while cur < len(bars) and bars[cur][0] <= timestamp:
                ts, row = bars[cur]
                if ts == timestamp or (
                    ts.date() == timestamp.date()
                ):
                    events.append(self._row_to_event(sym, ts, row))
                    cur += 1
                    break
                cur += 1
            self._cursor[sym] = cur
        return events

    @staticmethod
    def _row_to_event(symbol: str, ts: datetime, row: dict) -> MarketEvent:
        return MarketEvent(
            symbol    = symbol,
            timestamp = ts,
            open      = float(row.get("open",  0)),
            high      = float(row.get("high",  0)),
            low       = float(row.get("low",   0)),
            close     = float(row.get("close", 0)),
            adj_close = float(row.get("adj_close", row.get("close", 0))),
            volume    = float(row.get("volume", 0)),
        )


# ──────────────────────────────────────────────
# In-Memory Data Handler (for tests / synthetic)
# ──────────────────────────────────────────────

class SyntheticDataHandler(DataHandlerBase):
    """
    Feeds pre-built MarketEvent lists — useful for unit tests
    and synthetic data experiments.
    """

    def __init__(self, data: Dict[str, List[MarketEvent]], lookback: int = 500):
        super().__init__(lookback)
        # Sort by timestamp
        self._data: Dict[str, List[MarketEvent]] = {
            s: sorted(v, key=lambda e: e.timestamp) for s, v in data.items()
        }
        self._cursor: Dict[str, int] = {s: 0 for s in data}

    def get_latest_bars(
        self, symbols: Iterable[str], timestamp: datetime
    ) -> List[MarketEvent]:
        events = []
        for sym in symbols:
            bars = self._data.get(sym, [])
            cur = self._cursor.get(sym, 0)
            while cur < len(bars) and bars[cur].timestamp <= timestamp:
                if bars[cur].timestamp == timestamp:
                    events.append(bars[cur])
                cur += 1
            self._cursor[sym] = cur
        return events

    @classmethod
    def generate_random_walk(
        cls,
        symbols: List[str],
        start: datetime,
        n_bars: int,
        freq_hours: int = 24,
        seed: int = 42,
    ) -> "SyntheticDataHandler":
        rng = np.random.default_rng(seed)
        data: Dict[str, List[MarketEvent]] = {}

        for sym in symbols:
            s0 = rng.uniform(10.0, 500.0)
            bars = []
            price = s0
            ts = start
            for _ in range(n_bars):
                ret    = rng.normal(0.0002, 0.015)
                open_  = price
                close  = price * (1 + ret)
                high   = max(open_, close) * (1 + abs(rng.normal(0, 0.005)))
                low    = min(open_, close) * (1 - abs(rng.normal(0, 0.005)))
                vol    = abs(rng.normal(1_000_000, 300_000))
                bars.append(MarketEvent(
                    symbol=sym, timestamp=ts,
                    open=round(open_, 4), high=round(high, 4),
                    low=round(low, 4),   close=round(close, 4),
                    adj_close=round(close, 4), volume=round(vol),
                ))
                price = close
                ts = ts + pd.Timedelta(hours=freq_hours)
            data[sym] = bars
        return cls(data)
