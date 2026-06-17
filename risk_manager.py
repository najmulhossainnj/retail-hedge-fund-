"""
risk_manager.py — Pre-trade and post-trade risk controls.

Every OrderEvent passes through the risk manager before reaching the
execution layer.  The manager can:
  • Approve the order as-is
  • Reduce the quantity
  • Reject the order entirely
  • Emit a RiskBreachEvent that triggers position flattening

Limits enforced:
  ┌────────────────────────────────────────────┐
  │ 1. Max single-position weight              │
  │ 2. Max gross exposure                      │
  │ 3. Max net exposure                        │
  │ 4. Max sector / factor exposure            │
  │ 5. Portfolio drawdown stop                 │
  │ 6. Daily loss limit                        │
  │ 7. Parametric VaR ceiling                  │
  └────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..engine.events import (
    OrderEvent, OrderSide, OrderType,
    RiskBreachEvent, RiskBreachType,
)
from ..data.data_handler import DataHandlerBase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Risk Limits Config
# ──────────────────────────────────────────────

@dataclass
class RiskLimits:
    """
    All risk parameters in one place.
    None = limit disabled.
    """
    # Position-level
    max_position_weight:    float = 0.10     # fraction of NAV
    max_position_notional:  Optional[float] = None

    # Portfolio-level
    max_gross_leverage:     float = 2.0      # gross / NAV
    max_net_leverage:       float = 1.0      # |net| / NAV
    max_long_exposure:      float = 1.5
    max_short_exposure:     float = 0.75

    # Drawdown / loss
    max_drawdown:           float = 0.15     # from HWM
    daily_loss_limit:       float = 0.03     # fraction of NAV at day start

    # VaR  (parametric, 1-day, 99%)
    max_var_pct:            Optional[float] = 0.02

    # Sector / factor limits  {sector_id: max_weight}
    sector_limits:          Dict[str, float] = field(default_factory=dict)

    # Concentration
    max_single_name_pct:    float = 0.10     # same as max_position_weight

    # Minimum cash buffer
    min_cash_pct:           float = 0.05


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class RiskManagerBase(ABC):

    @abstractmethod
    def screen_orders(
        self,
        orders:    List[OrderEvent],
        portfolio,
        data:      DataHandlerBase,
    ) -> List[OrderEvent]:
        """Filter / adjust orders.  Return approved orders."""

    @abstractmethod
    def generate_remediation(
        self,
        breach,    # RiskBreachEvent
        portfolio,
        data:      DataHandlerBase,
    ) -> List[OrderEvent]:
        """Called when a RiskBreachEvent fires — return orders to fix it."""


# ──────────────────────────────────────────────
# Standard Risk Manager
# ──────────────────────────────────────────────

class StandardRiskManager(RiskManagerBase):

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        sector_map: Optional[Dict[str, str]] = None,  # symbol → sector
    ):
        self.limits     = limits or RiskLimits()
        self.sector_map = sector_map or {}

        self._hwm:          float = 0.0
        self._day_start_nav: float = 0.0
        self._current_date:  Optional[date] = None
        self._breach_queue:  List[RiskBreachEvent] = []

    # ──────────────────────────────────────────
    # Pre-trade screening
    # ──────────────────────────────────────────

    def screen_orders(self, orders, portfolio, data):
        state   = portfolio.state
        nav     = state.nav

        # Update high-water mark and daily loss baseline
        self._update_hwm_and_daily(state)

        # Check portfolio-level hard stops before touching any order
        breach = self._check_portfolio_limits(state, nav)
        if breach:
            logger.warning(f"Portfolio-level risk breach: {breach.message}")
            self._breach_queue.append(breach)
            return []   # block ALL new orders during breach

        approved = []
        for order in orders:
            ok, adjusted = self._screen_single(order, state, data, nav)
            if ok and adjusted:
                approved.append(adjusted)

        return approved

    def _screen_single(
        self,
        order: OrderEvent,
        state,
        data:  DataHandlerBase,
        nav:   float,
    ) -> Tuple[bool, Optional[OrderEvent]]:
        """
        Returns (approved: bool, order_or_None).
        May reduce quantity before approving.
        """
        lim = self.limits
        sym = order.symbol

        # Price needed for notional calculations
        price = data.price(sym)
        if price is None or price <= 0:
            logger.warning(f"No price for {sym}, rejecting order.")
            return False, None

        # 1. Position weight limit
        sign = 1 if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else -1
        pos  = state.positions.get(sym)
        current_qty  = pos.quantity if pos else 0.0
        new_qty      = current_qty + sign * order.quantity
        new_weight   = abs(new_qty * price / nav) if nav > 0 else 0.0

        if new_weight > lim.max_position_weight:
            max_qty = (lim.max_position_weight * nav / price)
            allowed = max(0.0, sign * (max_qty - abs(current_qty)))
            if allowed < 1.0:
                logger.debug(f"Rejected {sym}: position limit. weight={new_weight:.3f}")
                return False, None
            order = _resize_order(order, allowed)
            logger.debug(f"Reduced {sym} order to {allowed:.0f} shares (pos limit).")

        # 2. Notional cap
        if lim.max_position_notional:
            notional = order.quantity * price
            if notional > lim.max_position_notional:
                new_qty_cap = lim.max_position_notional / price
                order = _resize_order(order, new_qty_cap)

        # 3. Gross leverage
        proj_gross = state.gross_exposure + order.quantity * price
        if proj_gross / max(nav, 1) > lim.max_gross_leverage:
            logger.debug(f"Rejected {sym}: gross leverage limit.")
            return False, None

        # 4. Net exposure
        proj_net = state.net_exposure + sign * order.quantity * price
        if abs(proj_net) / max(nav, 1) > lim.max_net_leverage:
            logger.debug(f"Rejected {sym}: net leverage limit.")
            return False, None

        # 5. Cash buffer
        if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
            cost = order.quantity * price
            if (state.cash - cost) / max(nav, 1) < lim.min_cash_pct:
                logger.debug(f"Rejected {sym}: insufficient cash buffer.")
                return False, None

        return True, order

    # ──────────────────────────────────────────
    # Portfolio-level checks
    # ──────────────────────────────────────────

    def _check_portfolio_limits(
        self, state, nav: float
    ) -> Optional[RiskBreachEvent]:
        lim = self.limits

        # Drawdown stop
        if self._hwm > 0:
            dd = (self._hwm - nav) / self._hwm
            if dd > lim.max_drawdown:
                return RiskBreachEvent(
                    breach_type=RiskBreachType.MAX_DRAWDOWN,
                    current=dd, limit=lim.max_drawdown,
                    message=f"Max drawdown breached: {dd:.2%} > {lim.max_drawdown:.2%}",
                )

        # Daily loss limit
        if self._day_start_nav > 0:
            daily_loss = (self._day_start_nav - nav) / self._day_start_nav
            if daily_loss > lim.daily_loss_limit:
                return RiskBreachEvent(
                    breach_type=RiskBreachType.LOSS_LIMIT,
                    current=daily_loss, limit=lim.daily_loss_limit,
                    message=f"Daily loss limit: {daily_loss:.2%} > {lim.daily_loss_limit:.2%}",
                )

        # Gross exposure
        if lim.max_gross_leverage and nav > 0:
            gross_lev = state.gross_exposure / nav
            if gross_lev > lim.max_gross_leverage:
                return RiskBreachEvent(
                    breach_type=RiskBreachType.GROSS_EXPOSURE,
                    current=gross_lev, limit=lim.max_gross_leverage,
                    message=f"Gross leverage {gross_lev:.2f}x > {lim.max_gross_leverage:.2f}x",
                )

        return None

    # ──────────────────────────────────────────
    # Parametric VaR
    # ──────────────────────────────────────────

    def compute_var(
        self, state, data: DataHandlerBase, confidence: float = 0.99
    ) -> Optional[float]:
        """
        Parametric (Gaussian) 1-day VaR as a fraction of NAV.
        Returns None if insufficient data.
        """
        nav = state.nav
        if nav <= 0:
            return None

        weights, vols, corr_rows = [], [], []
        syms = []
        for sym, pos in state.active_positions().items():
            s = data.series(sym)
            if s is None or len(s) < 20:
                continue
            r    = s.returns(63)
            if len(r) < 5:
                continue
            syms.append(sym)
            weights.append(pos.market_value / nav)
            vols.append(float(np.std(r, ddof=1)))

        if not syms:
            return None

        n   = len(syms)
        w   = np.array(weights)
        s   = np.array(vols)

        # Use identity correlation if can't compute (simplification)
        cov = np.diag(s ** 2)
        portfolio_var = float(w @ cov @ w)
        portfolio_std = np.sqrt(portfolio_var)

        z = _normal_quantile(confidence)   # ~2.326 for 99%
        return float(z * portfolio_std)

    # ──────────────────────────────────────────
    # Remediation
    # ──────────────────────────────────────────

    def generate_remediation(self, breach, portfolio, data) -> List[OrderEvent]:
        """
        On a risk breach, return flatten orders for the whole book.
        Could be made more surgical (e.g. reduce only the offending leg).
        """
        logger.warning(f"Generating remediation for {breach.breach_type.value}")
        from datetime import datetime
        return portfolio.liquidate_symbols(
            symbols=list(portfolio.state.active_positions().keys()),
            data=data,
            clock=type("_c", (), {"now": lambda s: datetime.utcnow()})(),
        )

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _update_hwm_and_daily(self, state):
        nav = state.nav
        if nav > self._hwm:
            self._hwm = nav

        today = datetime.utcnow().date()
        if today != self._current_date:
            self._current_date   = today
            self._day_start_nav  = nav

    def pop_breaches(self) -> List[RiskBreachEvent]:
        """Allow engine to drain queued breach events."""
        b, self._breach_queue = self._breach_queue, []
        return b


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _resize_order(order: OrderEvent, new_qty: float) -> OrderEvent:
    """Return a copy of order with adjusted quantity."""
    from dataclasses import replace
    return replace(order, quantity=round(new_qty))


def _normal_quantile(p: float) -> float:
    """Approximate inverse normal CDF via Beasley-Springer-Moro."""
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    u = p - 0.5
    if abs(u) < 0.42:
        r = u * u
        return u * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / \
               ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1)
    r = np.log(-np.log(1 - p if u > 0 else p))
    x = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(c[5] +
        r*(c[6] + r*(c[7] + r*c[8])))))))
    return x if u >= 0 else -x
