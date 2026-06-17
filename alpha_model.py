"""
alpha_model.py — Signal generation.

Mirrors Lean's IAlphaModel interface.  Each model receives a MarketEvent
plus the full DataHandler, does its computation, and emits a list of
SignalEvents with continuous strength scores in [-1, +1].

Models:
  • MomentumAlpha        — Cross-sectional relative-strength momentum
  • MeanReversionAlpha   — Z-score mean reversion
  • TrendFollowingAlpha  — Dual EMA trend filter
  • VolatilityAlpha      — Low-vol anomaly
  • CompositeAlpha       — Weighted blend of multiple models
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..engine.events import SignalDirection, SignalEvent
from ..data.data_handler import DataHandlerBase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class AlphaModelBase(ABC):
    """
    Interface every alpha model must implement.

    generate_signals is called once per MarketEvent (one symbol / one bar).
    The model can look at any symbol in the data handler, not just the
    symbol that triggered the event.
    """

    name: str = "AlphaModelBase"

    @abstractmethod
    def generate_signals(
        self,
        event,           # MarketEvent
        data: DataHandlerBase,
        universe: List[str],
    ) -> List[SignalEvent]:
        """Return zero or more SignalEvents."""

    def _make_signal(
        self,
        symbol: str,
        strength: float,
        timestamp: datetime,
        metadata: Optional[dict] = None,
    ) -> SignalEvent:
        direction = (
            SignalDirection.LONG  if strength > 0.0 else
            SignalDirection.SHORT if strength < 0.0 else
            SignalDirection.FLAT
        )
        return SignalEvent(
            symbol=symbol,
            direction=direction,
            strength=round(strength, 6),
            alpha_model=self.name,
            timestamp=timestamp,
            metadata=metadata or {},
        )


# ──────────────────────────────────────────────
# 1. Cross-Sectional Momentum
# ──────────────────────────────────────────────

class MomentumAlpha(AlphaModelBase):
    """
    Classic 12-1 month cross-sectional momentum (Jegadeesh & Titman 1993).

    Ranks all assets by their (lookback - skip) return, then assigns
    normalised strength scores in [-1, +1].  Top decile = long,
    bottom decile = short, rest = flat.
    """

    name = "CrossSectionalMomentum"

    def __init__(
        self,
        lookback: int   = 252,   # bars (≈12 months daily)
        skip:     int   = 21,    # bars to skip before current (≈1 month)
        long_pct: float = 0.3,   # top fraction to go long
        short_pct:float = 0.3,   # bottom fraction to short
        rebalance_every: int = 21,  # bars between rebalances
    ):
        self.lookback         = lookback
        self.skip             = skip
        self.long_pct         = long_pct
        self.short_pct        = short_pct
        self.rebalance_every  = rebalance_every
        self._bars_since_rebal = 0
        self._last_signals: Dict[str, float] = {}

    def generate_signals(self, event, data, universe):
        self._bars_since_rebal += 1
        if self._bars_since_rebal < self.rebalance_every:
            # Re-emit cached signals (stale but still valid)
            return [
                self._make_signal(s, v, event.timestamp)
                for s, v in self._last_signals.items()
            ]
        self._bars_since_rebal = 0

        # Compute momentum returns for each eligible symbol
        scores: Dict[str, float] = {}
        for sym in universe:
            s = data.series(sym)
            if s is None or len(s) < self.lookback + 1:
                continue
            closes = s.closes(self.lookback + self.skip)
            if len(closes) < self.lookback + self.skip:
                continue
            # Lookback return excluding skip period
            ret = closes[-(self.skip + 1)] / closes[0] - 1.0
            scores[sym] = ret

        if not scores:
            return []

        # Cross-sectional rank → normalised strength
        syms   = list(scores.keys())
        vals   = np.array([scores[s] for s in syms])
        ranks  = vals.argsort().argsort().astype(float)  # 0..N-1
        n      = len(ranks)
        norm   = (ranks / (n - 1)) * 2 - 1   # rescale to [-1, +1]

        long_threshold  =  (1 - self.long_pct)  * 2 - 1
        short_threshold = -(1 - self.short_pct) * 2 + 1

        signals = []
        self._last_signals = {}
        for sym, score in zip(syms, norm):
            if   score >= long_threshold:
                strength = float(score)
            elif score <= short_threshold:
                strength = float(score)
            else:
                strength = 0.0

            self._last_signals[sym] = strength
            if strength != 0.0:
                signals.append(self._make_signal(
                    sym, strength, event.timestamp,
                    {"raw_return": float(scores[sym]), "rank_score": float(score)},
                ))

        logger.debug(
            f"[{self.name}] {len(signals)} signals at {event.timestamp.date()}"
        )
        return signals


# ──────────────────────────────────────────────
# 2. Mean Reversion (Z-Score)
# ──────────────────────────────────────────────

class MeanReversionAlpha(AlphaModelBase):
    """
    Z-score mean reversion.  When price deviates > entry_z std devs
    from its rolling mean, fade the move.
    """

    name = "MeanReversion"

    def __init__(
        self,
        lookback:  int   = 20,
        entry_z:   float = 2.0,   # enter when |z| > this
        exit_z:    float = 0.5,   # exit when |z| < this
        max_z:     float = 4.0,   # cap z-score signal
    ):
        self.lookback = lookback
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self.max_z    = max_z

    def generate_signals(self, event, data, universe):
        sym = event.symbol
        s = data.series(sym)
        if s is None or len(s) < self.lookback:
            return []

        z = s.zscore(self.lookback)
        if z is None:
            return []

        # Mean reversion: fade the deviation → sign is flipped
        if abs(z) > self.entry_z:
            clamped  = max(-self.max_z, min(self.max_z, z))
            strength = -clamped / self.max_z   # negative z → long, positive → short
            return [self._make_signal(sym, strength, event.timestamp, {"zscore": z})]

        if abs(z) < self.exit_z:
            # Flat signal to close any open mean-reversion position
            return [self._make_signal(sym, 0.0, event.timestamp, {"zscore": z})]

        return []


# ──────────────────────────────────────────────
# 3. Trend Following (Dual EMA)
# ──────────────────────────────────────────────

class TrendFollowingAlpha(AlphaModelBase):
    """
    Dual-EMA crossover with strength proportional to
    the spread between fast and slow EMA.
    """

    name = "TrendFollowing"

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, event, data, universe):
        sym = event.symbol
        s = data.series(sym)
        if s is None or len(s) < self.slow:
            return []

        ema_fast = s.ema(self.fast)
        ema_slow = s.ema(self.slow)
        if ema_fast is None or ema_slow is None:
            return []

        spread   = (ema_fast - ema_slow) / ema_slow  # fractional spread
        strength = np.tanh(spread * 10)               # squash to (-1,+1)

        return [self._make_signal(sym, float(strength), event.timestamp,
                                  {"ema_fast": ema_fast, "ema_slow": ema_slow})]


# ──────────────────────────────────────────────
# 4. Low Volatility Anomaly
# ──────────────────────────────────────────────

class LowVolatilityAlpha(AlphaModelBase):
    """
    Rank assets by realised volatility; prefer lower-vol assets.
    Strength = inverse vol rank (higher = lower vol = more long).
    """

    name = "LowVolatility"

    def __init__(self, lookback: int = 63, rebalance_every: int = 21):
        self.lookback         = lookback
        self.rebalance_every  = rebalance_every
        self._counter = 0

    def generate_signals(self, event, data, universe):
        self._counter += 1
        if self._counter % self.rebalance_every != 0:
            return []

        vols: Dict[str, float] = {}
        for sym in universe:
            s = data.series(sym)
            if s is None or len(s) < self.lookback:
                continue
            r = s.returns(self.lookback)
            if len(r) > 0:
                vols[sym] = float(np.std(r, ddof=1))

        if not vols:
            return []

        syms = list(vols.keys())
        vol_array = np.array([vols[s] for s in syms])
        # Inverse rank: lowest vol gets highest score
        ranks = vol_array.argsort().argsort().astype(float)
        n = len(ranks)
        strength_array = 1.0 - 2.0 * ranks / (n - 1) if n > 1 else np.zeros(n)

        return [
            self._make_signal(sym, float(strength_array[i]), event.timestamp,
                              {"realised_vol": float(vol_array[i])})
            for i, sym in enumerate(syms)
            if abs(strength_array[i]) > 0.1
        ]


# ──────────────────────────────────────────────
# 5. Composite Alpha (multi-factor blend)
# ──────────────────────────────────────────────

class CompositeAlpha(AlphaModelBase):
    """
    Combines multiple alpha models with configurable weights.
    The aggregate signal is a weighted average of individual signals.
    """

    name = "CompositeAlpha"

    def __init__(self, models: List[Tuple[AlphaModelBase, float]]):
        """
        Args:
            models: list of (model, weight) tuples.
                    Weights are normalised internally.
        """
        if not models:
            raise ValueError("CompositeAlpha requires at least one model.")
        self._models = models
        total = sum(w for _, w in models)
        self._weights = [w / total for _, w in models]

    def generate_signals(self, event, data, universe):
        # Collect signals per symbol from each sub-model
        agg: Dict[str, List[Tuple[float, float]]] = {}   # sym → [(weight, strength)]

        for (model, _), weight in zip(self._models, self._weights):
            sigs = model.generate_signals(event, data, universe)
            for sig in sigs:
                if sig.symbol not in agg:
                    agg[sig.symbol] = []
                agg[sig.symbol].append((weight, sig.strength))

        # Weighted average per symbol
        signals = []
        for sym, pairs in agg.items():
            total_w   = sum(w for w, _ in pairs)
            weighted  = sum(w * s for w, s in pairs)
            combined  = weighted / total_w if total_w > 0 else 0.0
            if abs(combined) > 1e-6:
                signals.append(self._make_signal(
                    sym, combined, event.timestamp,
                    {"n_models_contributing": len(pairs)},
                ))
        return signals
