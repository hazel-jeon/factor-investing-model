"""
portfolio/backtester.py
-----------------------
Walk-forward backtesting engine.

At each rebalancing date:
  1. Compute factor scores using data available *up to that date* (no look-ahead).
  2. Select top-N stocks by composite score.
  3. Hold equal-weight portfolio until the next rebalancing date.
  4. Record daily returns.

Optional: Volatility Targeting
  - Each day, measure the realised volatility of the portfolio over the past
    `vol_lookback` trading days.
  - Scale the day's return by (target_vol / realised_vol), capped at
    `vol_max_leverage` to prevent runaway leverage.
  - When realised_vol < target_vol the position is scaled UP (implicit leverage).
  - When realised_vol > target_vol the position is scaled DOWN (de-risking).
  - The scalar is recorded in the results DataFrame as ``vol_scalar``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..factors.scorer import FactorScorer

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


class Backtester:
    """
    Walk-forward factor-portfolio backtester with optional volatility targeting.

    Parameters
    ----------
    scorer : FactorScorer
        Configured composite factor scorer.
    prices : pd.DataFrame
        Daily adjusted-close panel (rows = dates, columns = tickers).
    fundamentals : pd.DataFrame
        Static fundamental data indexed by ticker.
        (In a production system, you would use point-in-time fundamentals.)
    rebalance_freq : str
        Pandas offset alias for rebalancing frequency.
        ``'QS'`` = quarter-start, ``'MS'`` = month-start, ``'AS'`` = year-start.
    n_stocks : int
        Number of stocks to hold in the portfolio.
    transaction_cost : float
        One-way transaction cost as a decimal (e.g. 0.001 = 0.1 %).
    benchmark_prices : pd.Series, optional
        Benchmark price series (e.g. SPY).  Used only for reporting.
    vol_target : float or None
        Annualised volatility target (e.g. 0.15 = 15 %).
        ``None`` disables volatility targeting entirely (original behaviour).
    vol_lookback : int
        Rolling window (in trading days) used to estimate realised volatility.
        Default 60 (~3 months).  Shorter = more reactive, more turnover.
    vol_max_leverage : float
        Hard cap on the vol scalar.  Default 2.0 (never more than 2× leveraged).
    """

    def __init__(
        self,
        scorer: FactorScorer,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        rebalance_freq: str = "QS",
        n_stocks: int = 20,
        transaction_cost: float = 0.001,
        benchmark_prices: Optional[pd.Series] = None,
        vol_target: Optional[float] = None,
        vol_lookback: int = 60,
        vol_max_leverage: float = 2.0,
    ) -> None:
        self.scorer = scorer
        self.prices = prices
        self.fundamentals = fundamentals
        self.rebalance_freq = rebalance_freq
        self.n_stocks = n_stocks
        self.transaction_cost = transaction_cost
        self.benchmark_prices = benchmark_prices
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.vol_max_leverage = vol_max_leverage

        self._results: Optional[pd.DataFrame] = None
        self._rebalance_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest and return daily portfolio returns.

        Returns
        -------
        pd.DataFrame
            Columns:
              ``portfolio``    — factor portfolio daily return (after vol scaling)
              ``benchmark``    — benchmark daily return (NaN if not provided)
              ``holdings``     — number of stocks held that day
              ``vol_scalar``   — scaling factor applied that day (1.0 = no scaling)
              ``realised_vol`` — annualised realised vol used for scaling (NaN = off)
        """
        rebal_dates = self._build_rebalance_schedule()
        daily_returns = self._simulate(rebal_dates)
        self._results = daily_returns
        return daily_returns

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Run .run() first.")
        return self._results

    @property
    def rebalance_log(self) -> pd.DataFrame:
        """DataFrame with each rebalancing event and the selected tickers."""
        return pd.DataFrame(self._rebalance_log)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_rebalance_schedule(self) -> pd.DatetimeIndex:
        start = self.prices.index[0]
        end = self.prices.index[-1]
        return pd.date_range(start=start, end=end, freq=self.rebalance_freq)

    def _simulate(self, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        daily_ret: list[dict] = []
        current_holdings: pd.Index = pd.Index([])
        next_rebal_idx = 0

        all_dates = self.prices.index
        price_returns = self.prices.pct_change()

        # Pre-compute benchmark pct_change once (avoids O(n²) .pct_change() calls)
        bench_pct: Optional[pd.Series] = None
        if self.benchmark_prices is not None:
            bench_pct = self.benchmark_prices.pct_change()

        # Rolling buffer of *unscaled* daily portfolio returns for vol estimation
        raw_ret_buffer: list[float] = []

        for i, date in enumerate(all_dates):
            # ── Rebalance ──────────────────────────────────────────────────
            if next_rebal_idx < len(rebal_dates) and date >= rebal_dates[next_rebal_idx]:
                prices_so_far = self.prices.loc[:date]
                scores = self.scorer.score(
                    prices=prices_so_far,
                    fundamentals=self.fundamentals,
                    as_of=date,
                )
                if not scores.empty:
                    new_holdings = self.scorer.select_portfolio(scores, self.n_stocks)
                    turnover = self._compute_turnover(current_holdings, new_holdings)
                    current_holdings = new_holdings
                    self._rebalance_log.append(
                        {
                            "date": date,
                            "n_holdings": len(current_holdings),
                            "turnover": turnover,
                            "tickers": list(current_holdings),
                        }
                    )
                next_rebal_idx += 1

            # ── Raw (unscaled) portfolio return ────────────────────────────
            raw_ret = np.nan
            if len(current_holdings) > 0:
                valid = [t for t in current_holdings if t in price_returns.columns]
                if valid:
                    raw_ret = float(price_returns.loc[date, valid].mean())
                    was_rebal = bool(
                        self._rebalance_log and self._rebalance_log[-1]["date"] == date
                    )
                    if was_rebal:
                        raw_ret -= self._rebalance_log[-1]["turnover"] * self.transaction_cost

            # ── Volatility targeting ───────────────────────────────────────
            vol_scalar = 1.0
            realised_vol = np.nan

            if self.vol_target is not None and not np.isnan(raw_ret):
                raw_ret_buffer.append(raw_ret)
                # Only start scaling once we have enough history
                if len(raw_ret_buffer) >= self.vol_lookback:
                    window = raw_ret_buffer[-self.vol_lookback:]
                    # Annualised daily std
                    realised_vol = float(np.std(window, ddof=1) * np.sqrt(_TRADING_DAYS))
                    if realised_vol > 0:
                        vol_scalar = min(
                            self.vol_target / realised_vol,
                            self.vol_max_leverage,
                        )
            elif not np.isnan(raw_ret):
                raw_ret_buffer.append(raw_ret)

            day_ret = raw_ret * vol_scalar if not np.isnan(raw_ret) else np.nan

            # ── Benchmark return ───────────────────────────────────────────
            bench_ret = np.nan
            if bench_pct is not None and date in bench_pct.index:
                bench_ret = float(bench_pct.loc[date])

            daily_ret.append(
                {
                    "date": date,
                    "portfolio": day_ret,
                    "benchmark": bench_ret,
                    "holdings": len(current_holdings),
                    "vol_scalar": vol_scalar,
                    "realised_vol": realised_vol,
                }
            )

        df = pd.DataFrame(daily_ret).set_index("date")
        return df

    @staticmethod
    def _compute_turnover(old: pd.Index, new: pd.Index) -> float:
        """Fraction of portfolio that changed hands."""
        if len(old) == 0:
            return 1.0
        sold = len(old.difference(new))
        return sold / len(old)