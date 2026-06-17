"""
engine.py — Central event loop and pipeline orchestrator.

Architecture (mirrors Lean's AlgorithmManager):

  ┌─────────────────────────────────────────────┐
  │                   ENGINE                    │
  │                                             │
  │  Clock → DataHandler → AlphaModel           │
  │       → PortfolioConstructor → RiskManager  │
  │       → ExecutionHandler → Performance      │
  └─────────────────────────────────────────────┘

All inter-module communication happens via the shared event queue.
Modules never hold references to each other.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from .events import (
    Event, EventType,
    MarketEvent, SignalEvent, OrderEvent, FillEvent,
    RiskBreachEvent, UniverseChangeEvent,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Engine Config
# ──────────────────────────────────────────────

@dataclass
class EngineConfig:
    name:              str   = "HedgeFundEngine"
    max_bars:          Optional[int] = None   # None = unlimited
    log_level:         int   = logging.INFO
    warmup_bars:       int   = 0    # bars to skip before trading
    benchmark_symbol:  str   = "SPY"
    base_currency:     str   = "USD"
    initial_cash:      float = 10_000_000.0


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

class Engine:
    """
    Central event loop.

    Dependencies are injected at construction — the engine itself has
    zero business logic; it only routes events between the pipeline stages.
    """

    def __init__(
        self,
        clock,
        data_handler,
        universe_selector,
        alpha_model,
        portfolio_constructor,
        risk_manager,
        execution_handler,
        performance_tracker,
        config: EngineConfig = None,
    ):
        self.clock        = clock
        self.data         = data_handler
        self.universe     = universe_selector
        self.alpha        = alpha_model
        self.portfolio    = portfolio_constructor
        self.risk         = risk_manager
        self.execution    = execution_handler
        self.performance  = performance_tracker
        self.config       = config or EngineConfig()

        self._queue: queue.Queue[Event] = queue.Queue()
        self._running   = False
        self._bar_count = 0
        self._warmup    = self.config.warmup_bars

        # Custom event hooks (subscribe pattern)
        self._hooks: Dict[EventType, List[Callable[[Event], None]]] = {
            et: [] for et in EventType
        }

        logging.basicConfig(level=self.config.log_level)
        logger.info(f"[{self.config.name}] Engine initialised.")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def subscribe(self, event_type: EventType, fn: Callable[[Event], None]):
        """Register a callback for any event type (for plugins / monitoring)."""
        self._hooks[event_type].append(fn)

    def post(self, event: Event):
        """Allow external modules to inject events (e.g. corporate actions)."""
        self._queue.put(event)

    def run(self) -> Dict[str, Any]:
        """
        Main backtest / live loop.
        Returns a performance report dict when finished.
        """
        self._running = True
        start_wall = time.perf_counter()
        logger.info(
            f"[{self.config.name}] Starting at {self.clock.now()} | "
            f"cash={self.config.initial_cash:,.0f}"
        )

        self.portfolio.initialize(
            cash=self.config.initial_cash,
            clock=self.clock,
        )

        try:
            while self._running:
                # ── Advance clock ──────────────────────────
                if not self.clock.advance():
                    logger.info("Clock exhausted — backtest complete.")
                    break

                if (
                    self.config.max_bars is not None
                    and self._bar_count >= self.config.max_bars
                ):
                    logger.info(f"Max bars ({self.config.max_bars}) reached.")
                    break

                self._bar_count += 1
                ts = self.clock.now()

                # ── Universe rebalance (daily / weekly) ───
                universe_event = self.universe.select(ts, self.data)
                if universe_event:
                    self._dispatch(universe_event)

                # ── Generate market events for active symbols
                market_events = self.data.get_latest_bars(
                    self.universe.current_symbols, ts
                )
                for me in market_events:
                    self._dispatch(me)

                # ── Drain event queue ──────────────────────
                self._drain_queue()

        except KeyboardInterrupt:
            logger.warning("Engine interrupted by user.")
        finally:
            self._running = False

        elapsed = time.perf_counter() - start_wall
        logger.info(
            f"[{self.config.name}] Finished. "
            f"{self._bar_count} bars in {elapsed:.1f}s "
            f"({self._bar_count / max(elapsed, 0.001):.0f} bars/s)"
        )

        return self.performance.generate_report()

    def stop(self):
        self._running = False

    # ──────────────────────────────────────────
    # Event dispatching
    # ──────────────────────────────────────────

    def _drain_queue(self):
        """Process all events currently on the queue."""
        while not self._queue.empty():
            try:
                event = self._queue.get(block=False)
                self._route(event)
                # Fire external hooks
                for fn in self._hooks.get(event.type, []):
                    fn(event)
            except queue.Empty:
                break
            except Exception as exc:
                logger.error(f"Error routing {event}: {exc}", exc_info=True)

    def _dispatch(self, event: Event):
        """Put event on queue (also fires hooks immediately for market events)."""
        self._queue.put(event)

    def _route(self, event: Event):
        """Route a single event to the appropriate handler."""
        t = event.type
        if   t == EventType.MARKET:          self._on_market(event)
        elif t == EventType.SIGNAL:          self._on_signal(event)
        elif t == EventType.ORDER:           self._on_order(event)
        elif t == EventType.FILL:            self._on_fill(event)
        elif t == EventType.RISK_BREACH:     self._on_risk_breach(event)
        elif t == EventType.UNIVERSE_CHANGE: self._on_universe_change(event)

    # ──────────────────────────────────────────
    # Event handlers (pipeline stages)
    # ──────────────────────────────────────────

    def _on_market(self, event: MarketEvent):
        """Stage 1: Update bars → generate alpha signals."""
        self.data.update_bar(event)

        if self._bar_count <= self._warmup:
            return  # Skip trading during warmup

        signals = self.alpha.generate_signals(
            event=event,
            data=self.data,
            universe=self.universe.current_symbols,
        )
        for sig in signals:
            self._queue.put(sig)

    def _on_signal(self, event: SignalEvent):
        """Stage 2: Signals → target orders via portfolio construction."""
        orders = self.portfolio.construct_orders(
            signal=event,
            data=self.data,
            clock=self.clock,
        )
        # Run orders through risk filters before queueing
        approved = self.risk.screen_orders(
            orders=orders,
            portfolio=self.portfolio,
            data=self.data,
        )
        for order in approved:
            self._queue.put(order)

    def _on_order(self, event: OrderEvent):
        """Stage 3: Execute order → receive fill."""
        fill = self.execution.execute_order(
            order=event,
            data=self.data,
            clock=self.clock,
        )
        if fill:
            self._queue.put(fill)

    def _on_fill(self, event: FillEvent):
        """Stage 4: Update positions + performance on fill."""
        self.portfolio.update_on_fill(event)
        self.performance.record_fill(event, self.portfolio, self.clock)
        logger.debug(
            f"FILL  {event.side.value:12s} {event.symbol:8s} "
            f"qty={event.filled_quantity:>8.0f}  "
            f"px={event.fill_price:>10.4f}  "
            f"comm={event.commission:>7.2f}"
        )

    def _on_risk_breach(self, event: RiskBreachEvent):
        """Stage 5: Handle risk limit breach — flatten or reduce positions."""
        logger.warning(
            f"RISK BREACH [{event.breach_type.value}] "
            f"{event.symbol or 'PORTFOLIO'}  "
            f"current={event.current:.4f}  limit={event.limit:.4f}"
        )
        remediation_orders = self.risk.generate_remediation(
            breach=event,
            portfolio=self.portfolio,
            data=self.data,
        )
        for order in remediation_orders:
            self._queue.put(order)

    def _on_universe_change(self, event: UniverseChangeEvent):
        """Stage 0: Update active universe."""
        logger.info(
            f"UNIVERSE  +{len(event.added)} / -{len(event.removed)} symbols"
        )
        self.universe.apply_change(event)
        # Liquidate removed symbols
        liquidation_orders = self.portfolio.liquidate_symbols(
            symbols=event.removed, data=self.data, clock=self.clock
        )
        for order in liquidation_orders:
            self._queue.put(order)

    # ──────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────

    @property
    def bar_count(self) -> int:
        return self._bar_count

    def __repr__(self) -> str:
        return (
            f"Engine(name={self.config.name!r}, "
            f"bars={self._bar_count}, "
            f"running={self._running})"
        )
