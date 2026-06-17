# Modular Hedge Fund Engine
### A QuantConnect Lean-inspired event-driven trading system in Python

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BacktestBuilder / Engine                    │
│                                                                     │
│  ┌───────────┐   MarketEvent   ┌───────────────┐   SignalEvent      │
│  │  Clock    │ ──────────────► │  AlphaModel   │ ──────────────►   │
│  │(Backtest/ │                 │               │                   │
│  │  Live)    │                 │ • Momentum    │                   │
│  └───────────┘                 │ • MeanRevert  │   ┌────────────┐  │
│        │                       │ • TrendFollow │   │ Portfolio  │  │
│        │ advance()             │ • LowVol      │──►│Constructor │  │
│        ▼                       │ • Composite   │   │            │  │
│  ┌───────────┐                 └───────────────┘   │ • EqualWt  │  │
│  │   Data    │                                     │ • SignalWt │  │
│  │  Handler  │◄────────────────────────────────────│ • MV Opt.  │  │
│  │           │                                     └─────┬──────┘  │
│  │ • CSV     │     OrderEvent                            │         │
│  │ • Synth.  │◄──────────────────────────────────────────┘         │
│  │ • Live    │        ┌─────────────┐                              │
│  └─────┬─────┘        │ RiskManager │  RiskBreachEvent             │
│        │              │             │──────────────────────────────►│
│        │              │ • Pos Limit │                              │
│        │              │ • DD Stop   │   OrderEvent (approved)      │
│        │              │ • VaR       │──────────────►               │
│        │              └─────────────┘  ┌────────────────────┐      │
│        │                               │ ExecutionHandler   │      │
│        │◄──────────────────────────────│                    │      │
│        │          FillEvent            │ • SimulatedExec    │      │
│        │                               │ • SlippageModel    │      │
│        │                               │ • CommissionModel  │      │
│        │                               └────────────────────┘      │
│        │                                        │                  │
│        │                               ┌────────┴───────┐          │
│        └──────────────────────────────►│  Performance   │          │
│                                        │  Tracker       │          │
│  ┌─────────────┐                       │                │          │
│  │  Universe   │                       │ • NAV History  │          │
│  │  Selector   │                       │ • Sharpe       │          │
│  │             │                       │ • Max DD       │          │
│  │ • Static    │ UniverseChangeEvent   │ • Calmar       │          │
│  │ • Filtered  │──────────────────────►│ • Attribution  │          │
│  │ • Momentum  │                       └────────────────┘          │
│  └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Map

| Module | File | Description |
|--------|------|-------------|
| **Events** | `engine/events.py` | All event types (Market, Signal, Order, Fill, RiskBreach, Universe) |
| **Clock** | `engine/clock.py` | BacktestClock (deterministic) + LiveClock |
| **Engine** | `engine/engine.py` | Central event loop, routes all events between modules |
| **DataHandler** | `data/data_handler.py` | CSV loader, BarSeries (with numpy helpers), SyntheticDataHandler |
| **AlphaModel** | `alpha/alpha_model.py` | Momentum, MeanReversion, TrendFollowing, LowVol, Composite |
| **Portfolio** | `portfolio/portfolio.py` | Position tracking + EqualWeight / SignalWeighted / MeanVariance constructors |
| **RiskManager** | `risk/risk_manager.py` | Pre-trade screening, drawdown stop, daily loss limit, VaR |
| **Execution** | `execution/execution.py` | Simulated fills with slippage + commission models |
| **Performance** | `performance/performance.py` | Full tearsheet: Sharpe, Calmar, Sortino, drawdown, attribution |
| **Universe** | `universe/universe.py` | Static, Filtered, MomentumRanked, LiquidityScreened |
| **Backtest** | `backtest.py` | Fluent BacktestBuilder that wires all modules |

---

## Event Flow

```
Clock.advance()
    └─► DataHandler.get_latest_bars()
            └─► [MarketEvent × N symbols] → Queue
                    └─► AlphaModel.generate_signals()
                            └─► [SignalEvent] → Queue
                                    └─► PortfolioConstructor.construct_orders()
                                            └─► RiskManager.screen_orders()
                                                    └─► [OrderEvent] → Queue
                                                            └─► ExecutionHandler.execute_order()
                                                                    └─► [FillEvent] → Queue
                                                                            └─► Portfolio.update_on_fill()
                                                                                    └─► PerformanceTracker.record_fill()
```

---

## Quick Start

```python
from hedge_fund.backtest import BacktestBuilder
from hedge_fund.data.data_handler   import CSVDataHandler
from hedge_fund.universe.universe   import StaticUniverse
from hedge_fund.alpha.alpha_model   import MomentumAlpha, CompositeAlpha, TrendFollowingAlpha
from hedge_fund.portfolio.portfolio import EqualWeightConstructor
from hedge_fund.risk.risk_manager   import StandardRiskManager, RiskLimits
from hedge_fund.execution.execution import (
    SimulatedExecutionHandler, FixedBpsSlippage, PerShareCommission
)

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "GS"]

result = (
    BacktestBuilder("My First Strategy")
    .set_dates("2019-01-01", "2023-12-31")
    .set_cash(10_000_000)
    .set_warmup(260)
    .set_data(CSVDataHandler("./data/daily/"))
    .set_universe(StaticUniverse(SYMBOLS))
    .set_alpha(MomentumAlpha(lookback=252, skip=21, long_pct=0.3))
    .set_portfolio(EqualWeightConstructor(max_position_weight=0.10))
    .set_risk(StandardRiskManager(RiskLimits(
        max_drawdown=0.15, daily_loss_limit=0.03
    )))
    .set_execution(SimulatedExecutionHandler(
        slippage   = FixedBpsSlippage(bps=5),
        commission = PerShareCommission(),
    ))
    .run()
)

result.print_report()
print(result)  # BacktestResult(CAGR=12.3%, Sharpe=0.87, MaxDD=-8.2%, Calmar=1.50)
```

---

## Adding a Custom Alpha Model

```python
from hedge_fund.alpha.alpha_model import AlphaModelBase
from hedge_fund.engine.events import SignalEvent

class MyAlpha(AlphaModelBase):
    name = "MyAlpha"

    def generate_signals(self, event, data, universe):
        s = data.series(event.symbol)
        if s is None or len(s) < 14:
            return []
        rsi = s.rsi(14)
        if rsi is None:
            return []
        # Contrarian: buy oversold, sell overbought
        strength = -(rsi - 50) / 50   # maps [0,100] → [+1,-1]
        return [self._make_signal(event.symbol, strength, event.timestamp)]
```

---

## Adding a Custom Execution Handler (Live)

```python
from hedge_fund.execution.execution import ExecutionHandlerBase

class MyBrokerExecution(ExecutionHandlerBase):
    def execute_order(self, order, data, clock):
        # Call your broker API here
        # Return a FillEvent on success
        ...
```

---

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
```

No external optimisation or broker libraries required for backtesting.

---

## File Tree

```
hedge_fund/
├── __init__.py
├── backtest.py                 ← BacktestBuilder (entry point)
├── engine/
│   ├── events.py               ← All event dataclasses
│   ├── clock.py                ← BacktestClock / LiveClock
│   └── engine.py               ← Main event loop
├── data/
│   └── data_handler.py         ← CSV, Synthetic, BarSeries
├── alpha/
│   └── alpha_model.py          ← 5 alpha models + Composite
├── portfolio/
│   └── portfolio.py            ← Position + 3 constructors + MV optimizer
├── risk/
│   └── risk_manager.py         ← 7 risk limits + VaR + remediation
├── execution/
│   └── execution.py            ← 3 slippage + 3 commission + SimExec
├── performance/
│   └── performance.py          ← Full tearsheet (12 metrics)
├── universe/
│   └── universe.py             ← 4 universe selectors
└── examples/
    └── strategies.py           ← 2 complete end-to-end strategies
```
