"""
events.py — Central event bus types.

Every module communicates exclusively through events placed on the
shared queue.  No module holds a reference to another module.

Event flow:
  DataHandler  →  MarketEvent
  AlphaModel   →  SignalEvent
  Portfolio    →  OrderEvent
  Execution    →  FillEvent
  Risk         →  RiskBreachEvent
  Universe     →  UniverseChangeEvent
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class EventType(Enum):
    MARKET          = auto()
    SIGNAL          = auto()
    ORDER           = auto()
    FILL            = auto()
    RISK_BREACH     = auto()
    UNIVERSE_CHANGE = auto()
    PORTFOLIO_UPDATE= auto()


class OrderType(Enum):
    MARKET     = "MARKET"
    LIMIT      = "LIMIT"
    STOP       = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MOC        = "MOC"   # Market-on-Close
    LOC        = "LOC"   # Limit-on-Close


class OrderSide(Enum):
    BUY           = "BUY"
    SELL          = "SELL"
    SELL_SHORT    = "SELL_SHORT"
    BUY_TO_COVER  = "BUY_TO_COVER"


class SignalDirection(Enum):
    LONG  =  1
    SHORT = -1
    FLAT  =  0


class RiskBreachType(Enum):
    MAX_DRAWDOWN    = "MAX_DRAWDOWN"
    POSITION_LIMIT  = "POSITION_LIMIT"
    GROSS_EXPOSURE  = "GROSS_EXPOSURE"
    NET_EXPOSURE    = "NET_EXPOSURE"
    SECTOR_LIMIT    = "SECTOR_LIMIT"
    VAR_LIMIT       = "VAR_LIMIT"
    LOSS_LIMIT      = "LOSS_LIMIT"


# ──────────────────────────────────────────────
# Base Event
# ──────────────────────────────────────────────

@dataclass
class Event:
    """All events inherit from this base."""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ──────────────────────────────────────────────
# Market Events
# ──────────────────────────────────────────────

@dataclass
class MarketEvent(Event):
    """
    Single OHLCV bar for one symbol.
    Emitted by DataHandler once per bar per symbol.
    """
    symbol:    str   = ""
    open:      float = 0.0
    high:      float = 0.0
    low:       float = 0.0
    close:     float = 0.0
    volume:    float = 0.0
    adj_close: float = 0.0
    vwap:      Optional[float] = None
    # Optional extended fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.type = EventType.MARKET

    @property
    def returns(self) -> float:
        return (self.adj_close / self.open) - 1.0 if self.open else 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


# ──────────────────────────────────────────────
# Signal Events
# ──────────────────────────────────────────────

@dataclass
class SignalEvent(Event):
    """
    Raw alpha signal from an AlphaModel.
    strength: continuous [-1.0, +1.0]
      +1.0 = maximum long conviction
      -1.0 = maximum short conviction
       0.0 = no view / flat
    """
    symbol:      str              = ""
    direction:   SignalDirection  = SignalDirection.FLAT
    strength:    float            = 0.0       # [-1, +1]
    alpha_model: str              = "unknown"
    horizon:     int              = 1         # bars
    metadata:    Dict[str, Any]   = field(default_factory=dict)

    def __post_init__(self):
        self.type = EventType.SIGNAL
        # Clamp strength
        self.strength = max(-1.0, min(1.0, self.strength))


# ──────────────────────────────────────────────
# Order Events
# ──────────────────────────────────────────────

@dataclass
class OrderEvent(Event):
    """
    Order instruction produced by PortfolioConstructor
    after passing through RiskManager.
    """
    symbol:       str       = ""
    order_type:   OrderType = OrderType.MARKET
    side:         OrderSide = OrderSide.BUY
    quantity:     float     = 0.0
    limit_price:  Optional[float] = None
    stop_price:   Optional[float] = None
    order_id:     str       = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id:  str       = ""
    parent_id:    Optional[str] = None   # for bracket / algo orders
    notes:        str       = ""

    def __post_init__(self):
        self.type = EventType.ORDER


# ──────────────────────────────────────────────
# Fill Events
# ──────────────────────────────────────────────

@dataclass
class FillEvent(Event):
    """
    Execution confirmation from broker / simulated engine.
    """
    order_id:        str       = ""
    symbol:          str       = ""
    side:            OrderSide = OrderSide.BUY
    filled_quantity: float     = 0.0
    fill_price:      float     = 0.0
    commission:      float     = 0.0
    slippage:        float     = 0.0
    exchange:        str       = "SIM"
    # Net cost: positive = cash out, negative = cash in
    @property
    def net_cost(self) -> float:
        sign = 1 if self.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER) else -1
        return sign * self.filled_quantity * self.fill_price + self.commission

    def __post_init__(self):
        self.type = EventType.FILL


# ──────────────────────────────────────────────
# Risk Breach Events
# ──────────────────────────────────────────────

@dataclass
class RiskBreachEvent(Event):
    breach_type: RiskBreachType = RiskBreachType.POSITION_LIMIT
    symbol:      Optional[str] = None
    current:     float = 0.0
    limit:       float = 0.0
    message:     str   = ""

    def __post_init__(self):
        self.type = EventType.RISK_BREACH


# ──────────────────────────────────────────────
# Universe Change Events
# ──────────────────────────────────────────────

@dataclass
class UniverseChangeEvent(Event):
    added:   List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.type = EventType.UNIVERSE_CHANGE


# ──────────────────────────────────────────────
# Portfolio Update Events
# ──────────────────────────────────────────────

@dataclass
class PortfolioUpdateEvent(Event):
    nav:          float = 0.0
    cash:         float = 0.0
    gross_exposure: float = 0.0
    net_exposure:   float = 0.0

    def __post_init__(self):
        self.type = EventType.PORTFOLIO_UPDATE
