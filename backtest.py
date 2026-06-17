"""
backtest.py — Backtest configuration and runner.

BacktestBuilder provides a fluent API for wiring together all
pipeline modules (Lean-style "algorithm" setup).

Usage:
    result = (
        BacktestBuilder("Momentum Strategy")
        .set_cash(10_000_000)
        .set_dates("2018-01-01", "2023-12-31")
        .set_data(CSVDataHandler("./data/"))
        .set_universe(StaticUniverse(["AAPL", "MSFT", ...]))
        .set_alpha(MomentumAlpha(lookback=252))
        .set_portfolio(EqualWeightConstructor(max_position_weight=0.10))
        .set_risk(StandardRiskManager())
        .set_execution(SimulatedExecutionHandler(slippage=FixedBpsSlippage(5)))
        .run()
    )
    result.print_report()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .engine.clock     import BacktestClock
from .engine.engine    import Engine, EngineConfig
from .performance.performance import PerformanceTracker

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Backtest Result wrapper
# ──────────────────────────────────────────────

class BacktestResult:
    """Wraps the report dict and provides convenience methods."""

    def __init__(self, report: Dict[str, Any], engine: Engine):
        self.report = report
        self.engine = engine

    def print_report(self):
        self.engine.performance.print_report(self.report)

    def get(self, key: str, default=None):
        return self.report.get(key, default)

    @property
    def sharpe(self) -> float:
        return self.report.get("risk", {}).get("sharpe", 0.0)

    @property
    def cagr(self) -> float:
        return self.report.get("summary", {}).get("cagr", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.report.get("drawdown", {}).get("max_drawdown", 0.0)

    @property
    def calmar(self) -> float:
        return self.report.get("drawdown", {}).get("calmar", 0.0)

    @property
    def total_return(self) -> float:
        return self.report.get("summary", {}).get("total_return", 0.0)

    def __repr__(self):
        return (
            f"BacktestResult("
            f"CAGR={self.cagr:.2%}, "
            f"Sharpe={self.sharpe:.2f}, "
            f"MaxDD={self.max_drawdown:.2%}, "
            f"Calmar={self.calmar:.2f})"
        )


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

class BacktestBuilder:
    """
    Fluent builder that assembles a complete backtest pipeline.
    Every setter returns self for chaining.

    Required setters (will raise if missing):
        set_data, set_universe, set_alpha, set_portfolio,
        set_risk, set_execution

    Optional setters:
        set_cash, set_dates, set_warmup, set_benchmark,
        set_performance, configure_engine
    """

    def __init__(self, name: str = "Backtest"):
        self._name          = name
        self._cash          = 10_000_000.0
        self._start         = "2018-01-01"
        self._end           = "2023-12-31"
        self._warmup        = 0
        self._benchmark     = "SPY"
        self._data          = None
        self._universe      = None
        self._alpha         = None
        self._portfolio     = None
        self._risk          = None
        self._execution     = None
        self._performance   = None
        self._engine_config = None
        self._max_bars      = None

    # ── Required ──────────────────────────────

    def set_data(self, handler) -> "BacktestBuilder":
        self._data = handler;   return self

    def set_universe(self, selector) -> "BacktestBuilder":
        self._universe = selector; return self

    def set_alpha(self, model) -> "BacktestBuilder":
        self._alpha = model;    return self

    def set_portfolio(self, constructor) -> "BacktestBuilder":
        self._portfolio = constructor; return self

    def set_risk(self, manager) -> "BacktestBuilder":
        self._risk = manager;   return self

    def set_execution(self, handler) -> "BacktestBuilder":
        self._execution = handler; return self

    # ── Optional ──────────────────────────────

    def set_cash(self, cash: float) -> "BacktestBuilder":
        self._cash = cash;      return self

    def set_dates(self, start: str, end: str) -> "BacktestBuilder":
        self._start = start;  self._end = end; return self

    def set_warmup(self, bars: int) -> "BacktestBuilder":
        self._warmup = bars;    return self

    def set_benchmark(self, symbol: str) -> "BacktestBuilder":
        self._benchmark = symbol; return self

    def set_performance(self, tracker) -> "BacktestBuilder":
        self._performance = tracker; return self

    def set_max_bars(self, n: int) -> "BacktestBuilder":
        self._max_bars = n;     return self

    # ── Run ───────────────────────────────────

    def run(self) -> BacktestResult:
        """Wire all modules together and execute the backtest."""
        self._validate()

        # Build clock
        clock = BacktestClock.from_range(self._start, self._end)

        # Default performance tracker if not supplied
        performance = self._performance or PerformanceTracker(
            benchmark_symbol=self._benchmark,
        )

        # Engine config
        cfg = self._engine_config or EngineConfig(
            name             = self._name,
            initial_cash     = self._cash,
            warmup_bars      = self._warmup,
            max_bars         = self._max_bars,
            benchmark_symbol = self._benchmark,
        )

        # Wire engine
        engine = Engine(
            clock               = clock,
            data_handler        = self._data,
            universe_selector   = self._universe,
            alpha_model         = self._alpha,
            portfolio_constructor = self._portfolio,
            risk_manager        = self._risk,
            execution_handler   = self._execution,
            performance_tracker = performance,
            config              = cfg,
        )

        # Hook: snapshot NAV every bar (market event hook)
        def _nav_snapshot(event):
            performance.snapshot_nav(self._portfolio, clock)

        from .engine.events import EventType
        engine.subscribe(EventType.MARKET, _nav_snapshot)

        # Pre-load symbols if data handler supports it
        if hasattr(self._data, "load_symbols"):
            all_syms = getattr(self._universe, "candidate_pool", None) \
                or getattr(self._universe, "_target", None) \
                or []
            if all_syms:
                self._data.load_symbols(all_syms)

        logger.info(
            f"\n{'='*54}\n"
            f"  BACKTEST: {self._name}\n"
            f"  Period  : {self._start} → {self._end}\n"
            f"  Cash    : ${self._cash:,.0f}\n"
            f"  Warmup  : {self._warmup} bars\n"
            f"{'='*54}"
        )

        t0     = time.perf_counter()
        report = engine.run()
        elapsed = time.perf_counter() - t0
        logger.info(f"Backtest complete in {elapsed:.1f}s")

        return BacktestResult(report, engine)

    def _validate(self):
        missing = [
            name for name, attr in [
                ("data",      self._data),
                ("universe",  self._universe),
                ("alpha",     self._alpha),
                ("portfolio", self._portfolio),
                ("risk",      self._risk),
                ("execution", self._execution),
            ] if attr is None
        ]
        if missing:
            raise ValueError(
                f"BacktestBuilder missing required modules: {missing}"
            )
