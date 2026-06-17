"""
execution.py — Order execution and fill simulation.

Mirrors Lean's IExecutionModel + fill models.

Components:
  SlippageModel   — How much market impact / spread do we pay?
  CommissionModel — Brokerage fees per trade
  ExecutionHandler — Combines both to produce FillEvents

Slippage models:
  FixedBpsSlippage    — Fixed basis points regardless of size
  VolumeSlippage      — Square-root market impact  (Almgren-Chriss)
  SpreadSlippage      — Half-spread only

Commission models:
  ZeroCommission
  PerShareCommission  — e.g. IB tiered: $0.005/share
  PerTradePctCommission — e.g. 0.05% of notional

Execution handlers:
  SimulatedExecutionHandler — Backtest / paper trading
  (IBKRExecutionHandler stub for live trading)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..engine.events import FillEvent, OrderEvent, OrderSide, OrderType
from ..data.data_handler import DataHandlerBase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Slippage Models
# ──────────────────────────────────────────────

class SlippageModelBase(ABC):
    @abstractmethod
    def apply(
        self,
        order: OrderEvent,
        fill_price: float,
        volume: float,
    ) -> float:
        """Return adjusted fill price after slippage."""


class FixedBpsSlippage(SlippageModelBase):
    """
    Fixed basis points slippage.
    Buys pay more, sells receive less.
    Default: 5 bps (0.05%).
    """
    def __init__(self, bps: float = 5.0):
        self.fraction = bps / 10_000

    def apply(self, order, fill_price, volume):
        direction = 1 if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else -1
        return fill_price * (1 + direction * self.fraction)


class VolumeSlippage(SlippageModelBase):
    """
    Almgren-Chriss square-root market impact model.

    impact = η * σ * sqrt(Q / V)

    where:
        η = impact coefficient (default 0.1)
        σ = daily volatility (estimated from ATR)
        Q = order size in shares
        V = daily volume
    """
    def __init__(self, eta: float = 0.1, min_bps: float = 1.0, max_bps: float = 200.0):
        self.eta     = eta
        self.min_bps = min_bps / 10_000
        self.max_bps = max_bps / 10_000

    def apply(self, order, fill_price, volume):
        if volume <= 0:
            return fill_price
        participation = order.quantity / volume
        impact = self.eta * math.sqrt(participation)
        impact = max(self.min_bps, min(self.max_bps, impact))
        direction = 1 if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else -1
        return fill_price * (1 + direction * impact)


class SpreadSlippage(SlippageModelBase):
    """
    Half-spread model.  Uses a fixed spread estimate in dollars.
    """
    def __init__(self, spread_pct: float = 0.02):  # 2 cents per $100 = 2 bps
        self.half_spread = spread_pct / 200

    def apply(self, order, fill_price, volume):
        direction = 1 if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else -1
        return fill_price * (1 + direction * self.half_spread)


# ──────────────────────────────────────────────
# Commission Models
# ──────────────────────────────────────────────

class CommissionModelBase(ABC):
    @abstractmethod
    def compute(self, order: OrderEvent, fill_price: float) -> float:
        """Return commission in base currency."""


class ZeroCommission(CommissionModelBase):
    def compute(self, order, fill_price): return 0.0


class PerShareCommission(CommissionModelBase):
    """
    Interactive Brokers tiered equity model (simplified).
    $0.005/share, min $1.00, max 1% of trade value.
    """
    def __init__(
        self,
        rate_per_share: float = 0.005,
        minimum:        float = 1.00,
        max_pct:        float = 0.01,
    ):
        self.rate     = rate_per_share
        self.minimum  = minimum
        self.max_pct  = max_pct

    def compute(self, order, fill_price):
        gross      = order.quantity * fill_price
        commission = max(self.minimum, order.quantity * self.rate)
        return min(commission, gross * self.max_pct)


class PerTradePctCommission(CommissionModelBase):
    """
    Percentage of notional (e.g. prime brokerage: 0.05% each way).
    """
    def __init__(self, pct: float = 0.0005):
        self.pct = pct

    def compute(self, order, fill_price):
        return order.quantity * fill_price * self.pct


# ──────────────────────────────────────────────
# Abstract Execution Handler
# ──────────────────────────────────────────────

class ExecutionHandlerBase(ABC):
    @abstractmethod
    def execute_order(
        self,
        order: OrderEvent,
        data:  DataHandlerBase,
        clock,
    ) -> Optional[FillEvent]:
        """Submit order and return FillEvent (or None if not filled)."""


# ──────────────────────────────────────────────
# Simulated Execution Handler (Backtest)
# ──────────────────────────────────────────────

class SimulatedExecutionHandler(ExecutionHandlerBase):
    """
    Simulated execution for backtesting.

    Fill logic:
      MARKET  → fills at next bar's open (default) or current close
      LIMIT   → fills if limit_price crosses bar range
      STOP    → fills if stop_price crosses bar range
      MOC     → fills at close price (market-on-close)

    Applies both a slippage model and a commission model.
    """

    def __init__(
        self,
        slippage:     Optional[SlippageModelBase]    = None,
        commission:   Optional[CommissionModelBase]  = None,
        fill_on_open: bool = True,   # True = next open; False = current close
        max_volume_pct: float = 0.10,  # max % of bar volume we'll fill
    ):
        self.slippage       = slippage   or FixedBpsSlippage(bps=5)
        self.commission     = commission or PerShareCommission()
        self.fill_on_open   = fill_on_open
        self.max_volume_pct = max_volume_pct

    def execute_order(self, order, data, clock) -> Optional[FillEvent]:
        bar = data.bar(order.symbol)
        if bar is None:
            logger.warning(f"No bar for {order.symbol}, order not filled.")
            return None

        # Determine reference price by order type
        if order.order_type == OrderType.MARKET:
            ref_price = bar.open if self.fill_on_open else bar.close
            filled_qty = order.quantity

        elif order.order_type == OrderType.MOC:
            ref_price = bar.close
            filled_qty = order.quantity

        elif order.order_type == OrderType.LIMIT:
            ref_price, filled_qty = self._simulate_limit(order, bar)
            if filled_qty == 0:
                return None

        elif order.order_type == OrderType.STOP:
            ref_price, filled_qty = self._simulate_stop(order, bar)
            if filled_qty == 0:
                return None

        else:
            ref_price  = bar.open
            filled_qty = order.quantity

        # Volume cap
        if bar.volume > 0:
            max_qty = bar.volume * self.max_volume_pct
            filled_qty = min(filled_qty, max_qty)

        if filled_qty < 1.0:
            return None

        # Apply slippage
        fill_price = self.slippage.apply(order, ref_price, bar.volume)
        slippage_cost = abs(fill_price - ref_price) * filled_qty

        # Compute commission
        commission = self.commission.compute(
            order=_clone_with_qty(order, filled_qty),
            fill_price=fill_price,
        )

        fill = FillEvent(
            order_id        = order.order_id,
            symbol          = order.symbol,
            side            = order.side,
            filled_quantity = filled_qty,
            fill_price      = fill_price,
            commission      = commission,
            slippage        = slippage_cost,
            timestamp       = clock.now(),
            exchange        = "SIM",
        )

        logger.debug(
            f"FILL  {order.side.value:12s} {order.symbol:8s}  "
            f"qty={filled_qty:>8.0f}  px={fill_price:>9.4f}  "
            f"slip={slippage_cost:>6.2f}  comm={commission:>6.2f}"
        )
        return fill

    # ── Limit / Stop fill logic ───────────────

    def _simulate_limit(self, order, bar):
        """Fill at limit price if bar range crosses it."""
        lp = order.limit_price
        if lp is None:
            return bar.open, order.quantity

        if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            # Buy limit: fill if ask (low) ≤ limit price
            if bar.low <= lp:
                return lp, order.quantity
        else:
            # Sell limit: fill if bid (high) ≥ limit price
            if bar.high >= lp:
                return lp, order.quantity

        return 0.0, 0.0

    def _simulate_stop(self, order, bar):
        """Fill at stop price if bar range crosses it."""
        sp = order.stop_price
        if sp is None:
            return bar.open, order.quantity

        if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            # Buy stop: fill if bar high ≥ stop price
            if bar.high >= sp:
                return max(sp, bar.open), order.quantity
        else:
            # Sell stop: fill if bar low ≤ stop price
            if bar.low <= sp:
                return min(sp, bar.open), order.quantity

        return 0.0, 0.0


# ──────────────────────────────────────────────
# Live Broker Stub (IBKR / FIX)
# ──────────────────────────────────────────────

class IBKRExecutionHandler(ExecutionHandlerBase):
    """
    Stub for live IBKR execution via ib_insync.

    In production, replace the stub with:
        import ib_insync as ib
        ...
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497):
        self.host = host
        self.port = port
        logger.info(f"IBKRExecutionHandler stub created ({host}:{port})")

    def execute_order(self, order, data, clock):
        # LIVE IMPLEMENTATION:
        #   1. Connect to TWS / IB Gateway
        #   2. Convert OrderEvent → ib_insync.Order
        #   3. Place via ib.placeOrder(contract, order)
        #   4. Wait for fill event (callback)
        #   5. Return FillEvent
        raise NotImplementedError(
            "IBKRExecutionHandler is a stub. "
            "Implement with ib_insync for live trading."
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _clone_with_qty(order: OrderEvent, qty: float) -> OrderEvent:
    from dataclasses import replace
    return replace(order, quantity=qty)
