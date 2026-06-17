"""
examples/strategies.py — Two complete strategy examples.

Demonstrates the full backtest builder API.

Strategy 1: Cross-Sectional Momentum (12-1, long-short)
Strategy 2: Long-only low-volatility composite (blended alpha)
"""

from __future__ import annotations

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Local imports ─────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hedge_fund.backtest import BacktestBuilder

from hedge_fund.data.data_handler      import SyntheticDataHandler
from hedge_fund.universe.universe      import StaticUniverse, LiquidityScreenedUniverse
from hedge_fund.alpha.alpha_model      import (
    MomentumAlpha, MeanReversionAlpha,
    TrendFollowingAlpha, LowVolatilityAlpha, CompositeAlpha,
)
from hedge_fund.portfolio.portfolio    import (
    EqualWeightConstructor, SignalWeightedConstructor,
    MeanVarianceConstructor,
)
from hedge_fund.risk.risk_manager      import StandardRiskManager, RiskLimits
from hedge_fund.execution.execution    import (
    SimulatedExecutionHandler,
    FixedBpsSlippage, VolumeSlippage,
    PerShareCommission, PerTradePctCommission,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared test universe (synthetic random-walk data)
# ──────────────────────────────────────────────────────────────────────────────

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM",  "BAC",  "GS",
    "JNJ",  "PFE",  "UNH",  "XOM",  "CVX",
    "WMT",  "COST", "HD",   "DIS",  "NFLX",
]

def make_synthetic_data(seed: int = 42) -> SyntheticDataHandler:
    """Generate 1500 bars of synthetic OHLCV data for all symbols."""
    return SyntheticDataHandler.generate_random_walk(
        symbols   = SYMBOLS,
        start     = datetime(2018, 1, 2, 9, 30),
        n_bars    = 1500,
        freq_hours= 24,
        seed      = seed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1: Cross-Sectional Momentum (Long-Short)
# ──────────────────────────────────────────────────────────────────────────────

def run_momentum_strategy() -> None:
    """
    Classic 12-1 cross-sectional momentum.

    Pipeline:
      Universe   : Static 20 large-caps
      Alpha      : MomentumAlpha (12-month lookback, 1-month skip)
      Portfolio  : EqualWeightConstructor (long-short, 1.5× gross leverage)
      Risk       : StandardRiskManager (10% position limit, 15% max DD)
      Execution  : SimulatedExecution (5 bps slippage + IB commission)
    """

    data = make_synthetic_data(seed=1)

    result = (
        BacktestBuilder("Cross-Sectional Momentum (Long-Short)")
        .set_dates("2018-06-01", "2023-12-31")
        .set_cash(10_000_000)
        .set_warmup(260)    # 1 year warmup for momentum signal
        .set_data(data)
        .set_universe(StaticUniverse(SYMBOLS))
        .set_alpha(MomentumAlpha(
            lookback         = 252,
            skip             = 21,
            long_pct         = 0.3,
            short_pct        = 0.3,
            rebalance_every  = 21,
        ))
        .set_portfolio(EqualWeightConstructor(
            max_position_weight    = 0.10,
            target_gross_leverage  = 1.5,
            allow_short            = True,
        ))
        .set_risk(StandardRiskManager(RiskLimits(
            max_position_weight  = 0.10,
            max_gross_leverage   = 1.6,
            max_net_leverage     = 0.6,
            max_drawdown         = 0.15,
            daily_loss_limit     = 0.03,
        )))
        .set_execution(SimulatedExecutionHandler(
            slippage   = FixedBpsSlippage(bps=5),
            commission = PerShareCommission(rate_per_share=0.005),
        ))
        .run()
    )

    result.print_report()
    print(f"\n  >> Result summary: {result}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2: Multi-Factor Composite (Long-Only, Mean-Variance Weights)
# ──────────────────────────────────────────────────────────────────────────────

def run_composite_mv_strategy() -> None:
    """
    Composite alpha + mean-variance portfolio construction.

    Alpha blend:
      40% Trend-following (50/200 dual-EMA)
      30% Low-volatility anomaly
      30% Mean reversion (z-score)

    Portfolio: Mean-variance optimiser (long-only)
    Risk: Conservative drawdown and exposure limits
    """

    data = make_synthetic_data(seed=2)

    composite_alpha = CompositeAlpha(models=[
        (TrendFollowingAlpha(fast=50, slow=200),    0.40),
        (LowVolatilityAlpha(lookback=63),            0.30),
        (MeanReversionAlpha(lookback=20, entry_z=1.5), 0.30),
    ])

    result = (
        BacktestBuilder("Composite Multi-Factor Long-Only MV")
        .set_dates("2018-06-01", "2023-12-31")
        .set_cash(10_000_000)
        .set_warmup(210)
        .set_data(data)
        .set_universe(StaticUniverse(SYMBOLS))
        .set_alpha(composite_alpha)
        .set_portfolio(MeanVarianceConstructor(
            lookback             = 63,
            risk_aversion        = 3.0,
            max_position_weight  = 0.15,
            allow_short          = False,
            rebalance_every      = 21,
        ))
        .set_risk(StandardRiskManager(RiskLimits(
            max_position_weight = 0.15,
            max_gross_leverage  = 1.0,
            max_net_leverage    = 1.0,
            max_drawdown        = 0.12,
            daily_loss_limit    = 0.025,
            min_cash_pct        = 0.05,
        )))
        .set_execution(SimulatedExecutionHandler(
            slippage   = VolumeSlippage(eta=0.1),
            commission = PerTradePctCommission(pct=0.0005),
        ))
        .run()
    )

    result.print_report()
    print(f"\n  >> Result summary: {result}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 3: Signal-Weighted Trend (illustrating SignalWeightedConstructor)
# ──────────────────────────────────────────────────────────────────────────────

def run_trend_signal_weighted() -> None:
    """
    Trend-following with signal-proportional position sizing.
    Higher conviction trends get more capital.
    """

    data = make_synthetic_data(seed=3)

    result = (
        BacktestBuilder("Trend-Following (Signal-Weighted)")
        .set_dates("2019-01-01", "2023-12-31")
        .set_cash(5_000_000)
        .set_warmup(200)
        .set_data(data)
        .set_universe(StaticUniverse(SYMBOLS))
        .set_alpha(TrendFollowingAlpha(fast=50, slow=200))
        .set_portfolio(SignalWeightedConstructor(
            max_position_weight    = 0.12,
            target_gross_leverage  = 1.0,
            temperature            = 0.5,
            allow_short            = False,
        ))
        .set_risk(StandardRiskManager(RiskLimits(
            max_position_weight = 0.12,
            max_gross_leverage  = 1.1,
            max_drawdown        = 0.20,
        )))
        .set_execution(SimulatedExecutionHandler(
            slippage   = SpreadSlippage(spread_pct=0.02),
            commission = ZeroCommission(),
        ))
        .run()
    )

    result.print_report()
    print(f"\n  >> Result summary: {result}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*54)
    print("  STRATEGY 1: Cross-Sectional Momentum")
    print("═"*54)
    run_momentum_strategy()

    print("\n" + "═"*54)
    print("  STRATEGY 2: Composite Multi-Factor MV")
    print("═"*54)
    run_composite_mv_strategy()
