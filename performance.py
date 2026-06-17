"""
performance.py — Portfolio performance tracking and analytics.

Tracks every fill, computes returns, and produces a full tearsheet
report dict with all industry-standard metrics.

Metrics computed:
  ┌─────────────────────────────────────────────┐
  │ Returns    : Total, CAGR, annualised vol     │
  │ Risk-adj   : Sharpe, Sortino, Calmar, Info   │
  │ Drawdown   : Max DD, avg DD, max DD duration │
  │ Activity   : # trades, win-rate, avg PnL     │
  │ Exposure   : avg gross/net, turnover         │
  │ Attribution: PnL by symbol and by month      │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..engine.events import FillEvent, OrderSide

logger = logging.getLogger(__name__)

ANNUALISATION = 252.0   # trading days per year


# ──────────────────────────────────────────────
# Trade Record
# ──────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol:     str
    side:       str
    quantity:   float
    fill_price: float
    commission: float
    slippage:   float
    timestamp:  datetime
    nav_at_fill: float


# ──────────────────────────────────────────────
# Performance Tracker
# ──────────────────────────────────────────────

class PerformanceTracker:
    """
    Plugged into the engine's _on_fill and _on_market hooks.
    Call record_fill() every time a FillEvent fires,
    and snapshot_nav() once per bar.
    """

    def __init__(
        self,
        benchmark_symbol: str = "SPY",
        risk_free_rate:   float = 0.05,   # annual
        ann_factor:       float = ANNUALISATION,
    ):
        self.benchmark_symbol = benchmark_symbol
        self.rfr_annual       = risk_free_rate
        self.rfr_daily        = risk_free_rate / ann_factor
        self.ann_factor       = ann_factor

        # Time series
        self._nav:        List[Tuple[datetime, float]] = []
        self._bench_px:   List[Tuple[datetime, float]] = []

        # Trade log
        self._trades: List[TradeRecord] = []

        # Attribution by symbol
        self._pnl_by_symbol: Dict[str, float] = defaultdict(float)

    # ──────────────────────────────────────────
    # Data ingestion
    # ──────────────────────────────────────────

    def record_fill(self, fill: FillEvent, portfolio, clock):
        """Called by engine on every FillEvent."""
        nav = portfolio.state.nav
        self._trades.append(TradeRecord(
            symbol      = fill.symbol,
            side        = fill.side.value,
            quantity    = fill.filled_quantity,
            fill_price  = fill.fill_price,
            commission  = fill.commission,
            slippage    = fill.slippage,
            timestamp   = fill.timestamp,
            nav_at_fill = nav,
        ))

    def snapshot_nav(self, portfolio, clock):
        """Call once per bar to record NAV and benchmark price."""
        nav = portfolio.state.nav
        ts  = clock.now()
        self._nav.append((ts, nav))

        # Snapshot portfolio PnL by symbol
        for sym, pos in portfolio.state.active_positions().items():
            self._pnl_by_symbol[sym] += pos.unrealized_pnl

    def record_benchmark(self, timestamp: datetime, price: float):
        self._bench_px.append((timestamp, price))

    # Alias used by engine
    def update(self, fill: FillEvent, portfolio, clock):
        self.record_fill(fill, portfolio, clock)

    # ──────────────────────────────────────────
    # Report generation
    # ──────────────────────────────────────────

    def generate_report(self) -> Dict:
        """
        Build and return a full performance report dict.
        All monetary values in base currency; ratios dimensionless.
        """
        if len(self._nav) < 2:
            return {"error": "Insufficient data for performance report."}

        nav_df  = self._nav_dataframe()
        returns = self._daily_returns(nav_df)

        report = {
            "summary":     self._summary_stats(nav_df, returns),
            "risk":        self._risk_stats(returns),
            "drawdown":    self._drawdown_stats(nav_df),
            "trading":     self._trading_stats(),
            "monthly_pnl": self._monthly_pnl(nav_df),
            "pnl_by_symbol": dict(sorted(
                self._pnl_by_symbol.items(), key=lambda x: -abs(x[1])
            )[:20]),
        }

        if len(self._bench_px) > 2:
            report["benchmark"] = self._benchmark_stats(nav_df, returns)

        return report

    # ──────────────────────────────────────────
    # Internal: stat blocks
    # ──────────────────────────────────────────

    def _summary_stats(self, nav_df: pd.DataFrame, returns: np.ndarray) -> Dict:
        start_nav = nav_df["nav"].iloc[0]
        end_nav   = nav_df["nav"].iloc[-1]
        total_ret = (end_nav - start_nav) / start_nav

        n_years   = len(returns) / self.ann_factor
        cagr      = (end_nav / start_nav) ** (1.0 / max(n_years, 1e-6)) - 1

        return {
            "start_date":   str(nav_df.index[0].date()),
            "end_date":     str(nav_df.index[-1].date()),
            "start_nav":    round(start_nav, 2),
            "end_nav":      round(end_nav, 2),
            "total_return": round(total_ret, 6),
            "cagr":         round(cagr, 6),
            "n_days":       len(returns),
            "n_trades":     len(self._trades),
        }

    def _risk_stats(self, returns: np.ndarray) -> Dict:
        if len(returns) < 2:
            return {}

        ann = self.ann_factor
        mu  = float(np.mean(returns))
        vol = float(np.std(returns, ddof=1)) * np.sqrt(ann)

        # Sharpe
        excess = returns - self.rfr_daily
        sharpe = (np.mean(excess) / np.std(excess, ddof=1)
                  * np.sqrt(ann)) if np.std(excess, ddof=1) > 0 else 0.0

        # Sortino (downside std)
        neg = returns[returns < self.rfr_daily]
        down_vol = float(np.std(neg, ddof=1)) * np.sqrt(ann) if len(neg) > 1 else vol
        sortino  = (mu - self.rfr_daily) * ann / down_vol if down_vol > 0 else 0.0

        # Skew / Kurtosis
        skew = float(pd.Series(returns).skew())
        kurt = float(pd.Series(returns).kurtosis())

        # VaR / CVaR (historical, 95%)
        var_95  = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

        return {
            "annualised_vol":    round(vol, 6),
            "sharpe":            round(sharpe, 4),
            "sortino":           round(sortino, 4),
            "skewness":          round(skew, 4),
            "excess_kurtosis":   round(kurt, 4),
            "var_95_daily":      round(var_95, 6),
            "cvar_95_daily":     round(cvar_95, 6),
            "best_day":          round(float(np.max(returns)), 6),
            "worst_day":         round(float(np.min(returns)), 6),
            "positive_days_pct": round(float((returns > 0).mean()), 4),
        }

    def _drawdown_stats(self, nav_df: pd.DataFrame) -> Dict:
        nav = nav_df["nav"].values
        hwm = np.maximum.accumulate(nav)
        dd  = (nav - hwm) / hwm

        max_dd   = float(np.min(dd))
        avg_dd   = float(np.mean(dd[dd < 0])) if (dd < 0).any() else 0.0

        # Drawdown duration (bars)
        in_dd  = dd < 0
        max_dur = 0
        cur_dur = 0
        for flag in in_dd:
            if flag:
                cur_dur += 1
                max_dur  = max(max_dur, cur_dur)
            else:
                cur_dur  = 0

        # Calmar ratio
        cagr   = self._summary_stats(nav_df, self._daily_returns(nav_df))["cagr"]
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "max_drawdown":      round(max_dd, 6),
            "avg_drawdown":      round(avg_dd, 6),
            "max_dd_duration":   max_dur,
            "calmar":            round(calmar, 4),
            "underwater_pct":    round(float(in_dd.mean()), 4),
        }

    def _trading_stats(self) -> Dict:
        if not self._trades:
            return {}

        n = len(self._trades)
        commissions = sum(t.commission for t in self._trades)
        slippages   = sum(t.slippage   for t in self._trades)

        # Approximate PnL per trade via net cost
        pnls = []
        for t in self._trades:
            sign = 1 if t.side in ("BUY", "BUY_TO_COVER") else -1
            pnls.append(-sign * t.quantity * t.fill_price - t.commission)

        win_rate = float(sum(p > 0 for p in pnls) / n) if pnls else 0.0
        avg_pnl  = float(np.mean(pnls)) if pnls else 0.0

        return {
            "n_trades":         n,
            "total_commission": round(commissions, 2),
            "total_slippage":   round(slippages, 2),
            "win_rate":         round(win_rate, 4),
            "avg_pnl_per_trade":round(avg_pnl, 4),
        }

    def _monthly_pnl(self, nav_df: pd.DataFrame) -> Dict:
        monthly = nav_df["nav"].resample("ME").last()
        monthly_ret = monthly.pct_change().dropna()
        return {
            str(k.date())[:7]: round(float(v), 6)
            for k, v in monthly_ret.items()
        }

    def _benchmark_stats(self, nav_df: pd.DataFrame, port_returns: np.ndarray) -> Dict:
        bench_df = pd.DataFrame(self._bench_px, columns=["ts","px"]).set_index("ts")
        bench_df.index = pd.to_datetime(bench_df.index)
        bench_ret = bench_df["px"].pct_change().dropna().values

        min_len = min(len(port_returns), len(bench_ret))
        if min_len < 5:
            return {}

        p = port_returns[-min_len:]
        b = bench_ret[-min_len:]

        beta   = float(np.cov(p, b)[0, 1] / np.var(b)) if np.var(b) > 0 else 0.0
        alpha  = float(np.mean(p) - beta * np.mean(b)) * self.ann_factor
        corr   = float(np.corrcoef(p, b)[0, 1])
        te     = float(np.std(p - b, ddof=1)) * np.sqrt(self.ann_factor)
        ir     = (float(np.mean(p - b)) * self.ann_factor / te) if te > 0 else 0.0

        bench_total = float(bench_df["px"].iloc[-1] / bench_df["px"].iloc[0] - 1)

        return {
            "benchmark_symbol":   self.benchmark_symbol,
            "benchmark_return":   round(bench_total, 6),
            "beta":               round(beta, 4),
            "alpha_annualised":   round(alpha, 6),
            "correlation":        round(corr, 4),
            "tracking_error":     round(te, 6),
            "information_ratio":  round(ir, 4),
        }

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _nav_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._nav, columns=["ts", "nav"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
        # De-duplicate intraday → keep last per day
        df = df.resample("D").last().dropna()
        return df

    def _daily_returns(self, nav_df: pd.DataFrame) -> np.ndarray:
        return nav_df["nav"].pct_change().dropna().values

    # ──────────────────────────────────────────
    # Pretty print
    # ──────────────────────────────────────────

    def print_report(self, report: Optional[Dict] = None):
        if report is None:
            report = self.generate_report()

        sep = "─" * 52
        def fmt_pct(v): return f"{v*100:>+8.2f}%"
        def fmt_num(v): return f"{v:>+10.4f}"

        print(f"\n{'PERFORMANCE REPORT':^52}")
        print(sep)

        s = report.get("summary", {})
        print(f"  Period     : {s.get('start_date')} → {s.get('end_date')}")
        print(f"  Start NAV  : ${s.get('start_nav', 0):>14,.2f}")
        print(f"  End NAV    : ${s.get('end_nav', 0):>14,.2f}")
        print(f"  Total Ret  : {fmt_pct(s.get('total_return',0))}")
        print(f"  CAGR       : {fmt_pct(s.get('cagr', 0))}")

        print(sep)
        r = report.get("risk", {})
        print(f"  Ann. Vol   : {fmt_pct(r.get('annualised_vol',0))}")
        print(f"  Sharpe     : {fmt_num(r.get('sharpe',0))}")
        print(f"  Sortino    : {fmt_num(r.get('sortino',0))}")
        print(f"  Skewness   : {fmt_num(r.get('skewness',0))}")
        print(f"  Best Day   : {fmt_pct(r.get('best_day',0))}")
        print(f"  Worst Day  : {fmt_pct(r.get('worst_day',0))}")
        print(f"  Win Days % : {r.get('positive_days_pct',0)*100:>7.1f}%")

        print(sep)
        d = report.get("drawdown", {})
        print(f"  Max DD     : {fmt_pct(d.get('max_drawdown',0))}")
        print(f"  Avg DD     : {fmt_pct(d.get('avg_drawdown',0))}")
        print(f"  Max DD Dur : {d.get('max_dd_duration',0):>8} days")
        print(f"  Calmar     : {fmt_num(d.get('calmar',0))}")

        print(sep)
        t = report.get("trading", {})
        print(f"  # Trades   : {t.get('n_trades',0):>8,}")
        print(f"  Win Rate   : {t.get('win_rate',0)*100:>7.1f}%")
        print(f"  Commission : ${t.get('total_commission',0):>13,.2f}")
        print(f"  Slippage   : ${t.get('total_slippage',0):>13,.2f}")

        if "benchmark" in report:
            print(sep)
            b = report["benchmark"]
            print(f"  Beta       : {fmt_num(b.get('beta',0))}")
            print(f"  Alpha (ann): {fmt_pct(b.get('alpha_annualised',0))}")
            print(f"  Info Ratio : {fmt_num(b.get('information_ratio',0))}")
            print(f"  Correlation: {fmt_num(b.get('correlation',0))}")

        print(sep)
        top_sym = list(report.get("pnl_by_symbol", {}).items())[:5]
        if top_sym:
            print("  Top P&L contributors:")
            for sym, pnl in top_sym:
                print(f"    {sym:<8} ${pnl:>+12,.2f}")
        print(sep)
