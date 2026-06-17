"""
portfolio.py — Portfolio state, position management, and construction.

Responsibilities:
  • Track live positions and cash
  • Translate alpha signals into target weights
  • Compute required orders to reach target weights
  • Support equal-weight, signal-weighted, and mean-variance construction

Mirrors Lean's SecurityPortfolioManager + IPortfolioConstructionModel.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..engine.events import (
    FillEvent, MarketEvent, OrderEvent,
    OrderSide, OrderType, SignalEvent,
)
from ..data.data_handler import DataHandlerBase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Position
# ──────────────────────────────────────────────

@dataclass
class Position:
    symbol:     str
    quantity:   float = 0.0       # positive = long, negative = short
    avg_cost:   float = 0.0       # average fill price
    realized_pnl: float = 0.0
    last_price: float = 0.0

    @property
    def is_long(self)  -> bool: return self.quantity > 0
    @property
    def is_short(self) -> bool: return self.quantity < 0
    @property
    def is_flat(self)  -> bool: return self.quantity == 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.avg_cost)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def update_price(self, price: float):
        self.last_price = price

    def apply_fill(self, qty: float, price: float) -> float:
        """
        Update position after a fill.
        Returns realized PnL from any closing trades.
        """
        realized = 0.0
        if self.quantity == 0:
            self.avg_cost = price
            self.quantity = qty
        elif (self.quantity > 0 and qty > 0) or (self.quantity < 0 and qty < 0):
            # Adding to position → update avg cost
            total_cost = self.avg_cost * self.quantity + price * qty
            self.quantity += qty
            self.avg_cost = total_cost / self.quantity if self.quantity != 0 else 0.0
        else:
            # Reducing / reversing position
            close_qty = min(abs(qty), abs(self.quantity))
            realized  = close_qty * (price - self.avg_cost) * np.sign(self.quantity)
            self.realized_pnl += realized
            self.quantity += qty
            if abs(self.quantity) < 1e-8:
                self.quantity = 0.0
                self.avg_cost = 0.0
            elif self.quantity * (self.quantity + qty) < 0:
                # Reversed → new avg cost is fill price
                self.avg_cost = price
        return realized


# ──────────────────────────────────────────────
# Portfolio State
# ──────────────────────────────────────────────

class PortfolioState:
    """Tracks all positions, cash, and aggregate metrics."""

    def __init__(self, initial_cash: float = 10_000_000.0):
        self.initial_cash  = initial_cash
        self.cash          = initial_cash
        self.positions: Dict[str, Position] = {}
        self._nav_history: List[Tuple[datetime, float]] = []

    # ── Position access ───────────────────────

    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def active_positions(self) -> Dict[str, Position]:
        return {s: p for s, p in self.positions.items() if not p.is_flat}

    # ── Aggregates ────────────────────────────

    @property
    def long_value(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.is_long)

    @property
    def short_value(self) -> float:
        return abs(sum(p.market_value for p in self.positions.values() if p.is_short))

    @property
    def gross_exposure(self) -> float:
        return self.long_value + self.short_value

    @property
    def net_exposure(self) -> float:
        return self.long_value - self.short_value

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def nav(self) -> float:
        return self.equity

    @property
    def total_pnl(self) -> float:
        return self.nav - self.initial_cash

    @property
    def total_return(self) -> float:
        return self.total_pnl / self.initial_cash

    def update_prices(self, prices: Dict[str, float]):
        for sym, px in prices.items():
            if sym in self.positions:
                self.positions[sym].update_price(px)

    def record_nav(self, timestamp: datetime):
        self._nav_history.append((timestamp, self.nav))

    def nav_series(self) -> List[Tuple[datetime, float]]:
        return list(self._nav_history)

    def apply_fill(self, fill: FillEvent) -> float:
        """Update position and cash from a fill. Returns realized PnL."""
        sign = 1 if fill.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else -1
        qty  = sign * fill.filled_quantity

        pos     = self.position(fill.symbol)
        pos.update_price(fill.fill_price)
        realized = pos.apply_fill(qty, fill.fill_price)

        self.cash -= fill.net_cost  # net_cost handles sign & commission
        return realized

    def weight(self, symbol: str) -> float:
        nav = self.nav
        if nav == 0:
            return 0.0
        p = self.positions.get(symbol)
        return (p.market_value / nav) if p else 0.0

    def weights(self) -> Dict[str, float]:
        nav = self.nav
        if nav == 0:
            return {}
        return {s: p.market_value / nav for s, p in self.positions.items() if not p.is_flat}


# ──────────────────────────────────────────────
# Abstract Portfolio Constructor
# ──────────────────────────────────────────────

class PortfolioConstructorBase(ABC):
    """
    Translates SignalEvents into OrderEvents.
    Has read access to PortfolioState.
    """

    def __init__(self, max_position_weight: float = 0.10):
        self.max_position_weight = max_position_weight
        self.state = PortfolioState()

    def initialize(self, cash: float, clock):
        self.state = PortfolioState(initial_cash=cash)
        self._clock = clock

    @abstractmethod
    def construct_orders(
        self,
        signal: SignalEvent,
        data:   DataHandlerBase,
        clock,
    ) -> List[OrderEvent]:
        """Convert a signal into concrete orders."""

    def update_on_fill(self, fill: FillEvent):
        self.state.apply_fill(fill)

    def liquidate_symbols(
        self,
        symbols: Iterable[str],
        data: DataHandlerBase,
        clock,
    ) -> List[OrderEvent]:
        """Market-sell (or cover) every active position in the given symbols."""
        orders = []
        for sym in symbols:
            pos = self.state.positions.get(sym)
            if pos is None or pos.is_flat:
                continue
            side = OrderSide.SELL if pos.is_long else OrderSide.BUY_TO_COVER
            orders.append(OrderEvent(
                symbol=sym,
                order_type=OrderType.MARKET,
                side=side,
                quantity=abs(pos.quantity),
                timestamp=clock.now(),
                notes="liquidation",
            ))
        return orders

    # ── Helpers ───────────────────────────────

    def _target_qty(
        self,
        symbol: str,
        target_weight: float,
        data: DataHandlerBase,
    ) -> float:
        """
        Convert target portfolio weight to share quantity.
        Returns the DELTA quantity (positive = buy, negative = sell).
        """
        price = data.price(symbol)
        if not price or price <= 0:
            return 0.0

        nav        = self.state.nav
        target_val = nav * target_weight
        current_val = self.state.position(symbol).market_value
        delta_val  = target_val - current_val
        return delta_val / price

    def _delta_to_order(
        self,
        symbol:    str,
        delta_qty: float,
        timestamp: datetime,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        strategy_id: str = "",
    ) -> Optional[OrderEvent]:
        if abs(delta_qty) < 1.0:   # ignore dust
            return None
        if delta_qty > 0:
            side = OrderSide.BUY
        else:
            pos = self.state.positions.get(symbol)
            side = (
                OrderSide.SELL if (pos and pos.is_long)
                else OrderSide.SELL_SHORT
            )
        return OrderEvent(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=abs(delta_qty),
            limit_price=limit_price,
            timestamp=timestamp,
            strategy_id=strategy_id,
        )


# ──────────────────────────────────────────────
# Equal-Weight Constructor
# ──────────────────────────────────────────────

class EqualWeightConstructor(PortfolioConstructorBase):
    """
    All active (non-zero) signals receive equal dollar weight.
    Long signals get +1/N, short signals get -1/N.
    """

    def __init__(
        self,
        max_position_weight: float = 0.10,
        target_gross_leverage: float = 1.0,  # 1.0 = 100% gross exposure
        allow_short: bool = True,
    ):
        super().__init__(max_position_weight)
        self.target_gross_leverage = target_gross_leverage
        self.allow_short = allow_short
        self._signals: Dict[str, float] = {}   # sym → strength

    def construct_orders(self, signal, data, clock):
        # Accumulate signals (overwrite old ones)
        if signal.strength == 0.0:
            self._signals.pop(signal.symbol, None)
        else:
            if not self.allow_short and signal.strength < 0:
                return []
            self._signals[signal.symbol] = signal.strength

        n_long  = sum(1 for v in self._signals.values() if v > 0)
        n_short = sum(1 for v in self._signals.values() if v < 0)
        n_total = n_long + n_short
        if n_total == 0:
            return []

        # Equal-weight each side
        lev     = self.target_gross_leverage
        w_long  = (lev / 2) / n_long  if n_long  else 0
        w_short = (lev / 2) / n_short if n_short else 0

        orders = []
        for sym, strength in self._signals.items():
            if strength > 0:
                target_weight = min(w_long, self.max_position_weight)
            else:
                target_weight = -min(w_short, self.max_position_weight)

            delta = self._target_qty(sym, target_weight, data)
            order = self._delta_to_order(
                sym, delta, clock.now(), strategy_id="EW"
            )
            if order:
                orders.append(order)
        return orders


# ──────────────────────────────────────────────
# Signal-Weighted Constructor
# ──────────────────────────────────────────────

class SignalWeightedConstructor(PortfolioConstructorBase):
    """
    Weights proportional to signal strength (softmax-normalised).
    """

    def __init__(
        self,
        max_position_weight: float = 0.15,
        target_gross_leverage: float = 1.0,
        temperature: float = 1.0,   # softmax temperature
        allow_short: bool = True,
    ):
        super().__init__(max_position_weight)
        self.target_gross_leverage = target_gross_leverage
        self.temperature = temperature
        self.allow_short = allow_short
        self._signals: Dict[str, float] = {}

    def construct_orders(self, signal, data, clock):
        if signal.strength == 0.0:
            self._signals.pop(signal.symbol, None)
        else:
            self._signals[signal.symbol] = signal.strength

        if not self._signals:
            return []

        syms      = list(self._signals.keys())
        strengths = np.array([self._signals[s] for s in syms])

        long_mask  = strengths > 0
        short_mask = strengths < 0

        weights = np.zeros(len(syms))
        lev     = self.target_gross_leverage

        if long_mask.any():
            ls = strengths[long_mask]
            sm = np.exp(ls / self.temperature)
            sm /= sm.sum()
            weights[long_mask] = sm * (lev / 2)

        if self.allow_short and short_mask.any():
            ss = -strengths[short_mask]
            sm = np.exp(ss / self.temperature)
            sm /= sm.sum()
            weights[short_mask] = -sm * (lev / 2)

        # Cap
        weights = np.clip(weights, -self.max_position_weight, self.max_position_weight)

        orders = []
        for sym, w in zip(syms, weights):
            delta = self._target_qty(sym, float(w), data)
            order = self._delta_to_order(sym, delta, clock.now(), strategy_id="SW")
            if order:
                orders.append(order)
        return orders


# ──────────────────────────────────────────────
# Mean-Variance Optimizer
# ──────────────────────────────────────────────

class MeanVarianceConstructor(PortfolioConstructorBase):
    """
    Markowitz mean-variance optimisation using historical returns.

    Solves for the portfolio that maximises expected Sharpe (via
    max-return-per-unit-risk) subject to long-only or long-short
    constraints.  Uses numpy's quadratic-form solver (no external
    solver dependency).
    """

    def __init__(
        self,
        lookback:             int   = 63,     # bars for covariance estimation
        risk_aversion:        float = 2.0,    # λ in U = μ - λ/2 σ²
        max_position_weight:  float = 0.10,
        allow_short:          bool  = False,
        rebalance_every:      int   = 21,     # bars
        min_weight:           float = 0.0,
    ):
        super().__init__(max_position_weight)
        self.lookback            = lookback
        self.risk_aversion       = risk_aversion
        self.allow_short         = allow_short
        self.rebalance_every     = rebalance_every
        self.min_weight          = min_weight
        self._counter            = 0
        self._target_weights: Dict[str, float] = {}

    def construct_orders(self, signal, data, clock):
        self._counter += 1
        if self._counter % self.rebalance_every != 0:
            return []

        # Gather return matrix
        eligible = [
            sym for sym in self._get_universe(data)
            if data.has_bars(sym, self.lookback)
        ]
        if len(eligible) < 2:
            return []

        ret_matrix = np.column_stack([
            data.series(sym).returns(self.lookback) for sym in eligible
        ])  # shape (T, N)

        mu    = ret_matrix.mean(axis=0)          # (N,)
        cov   = np.cov(ret_matrix.T)             # (N, N)

        # Analytical MV solution: w ∝ Σ⁻¹ μ  (unconstrained)
        try:
            cov_inv = np.linalg.pinv(cov)
            w_raw   = cov_inv @ mu
        except np.linalg.LinAlgError:
            return []

        if not self.allow_short:
            w_raw = np.maximum(w_raw, 0)

        total = np.sum(np.abs(w_raw))
        if total < 1e-10:
            return []

        w_norm = w_raw / total   # normalise to unit gross exposure

        # Apply position cap
        w_norm = np.clip(
            w_norm,
            -self.max_position_weight if self.allow_short else self.min_weight,
            self.max_position_weight,
        )

        orders = []
        self._target_weights = {}
        for sym, w in zip(eligible, w_norm):
            self._target_weights[sym] = float(w)
            delta = self._target_qty(sym, float(w), data)
            order = self._delta_to_order(sym, delta, clock.now(), strategy_id="MV")
            if order:
                orders.append(order)
        return orders

    def _get_universe(self, data: DataHandlerBase) -> List[str]:
        return data.symbols()
