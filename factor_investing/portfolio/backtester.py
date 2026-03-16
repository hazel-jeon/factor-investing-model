"""
portfolio/backtester.py
-----------------------
Walk-forward backtesting engine.

At each rebalancing date:
  1. Compute factor scores using data available *up to that date* (no look-ahead).
  2. Select top-N stocks by composite score.
  3. Compute portfolio weights via the chosen optimizer.
  4. Hold the weighted portfolio until the next rebalancing date.
  5. Record daily returns.

Weighting schemes
-----------------
equal (default)
    Each selected stock receives weight 1/N.  Fast and robust.

min_var
    Minimum-variance weights computed from the trailing ``cov_lookback``-day
    covariance matrix.  Tends to reduce portfolio volatility and improve the
    Sharpe ratio when the covariance structure is stable.

Pass ``optimizer="min_var"`` (or any key in ``OPTIMIZERS``) to switch.

Optional: Volatility Targeting
    Each day, scale the portfolio return by (target_vol / realised_vol),
    capped at ``vol_max_leverage``.  Set ``vol_target=None`` to disable.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..factors.scorer import FactorScorer
from .optimizer import equal_weight, minimum_variance, OPTIMIZERS

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


class Backtester:
    """
    Walk-forward factor-portfolio backtester.

    Parameters
    ----------
    scorer : FactorScorer
        Configured composite factor scorer.
    prices : pd.DataFrame
        Daily adjusted-close panel (rows = dates, columns = tickers).
    fundamentals : pd.DataFrame
        Static fundamental data indexed by ticker.
    rebalance_freq : str
        Pandas offset alias: ``'QS'`` quarter-start, ``'MS'`` month-start,
        ``'AS'`` year-start.
    n_stocks : int
        Number of stocks selected per rebalance (default 20).
    transaction_cost : float
        One-way transaction cost as a decimal (e.g. 0.001 = 0.1 %).
    benchmark_prices : pd.Series, optional
        Benchmark price series (e.g. SPY) for comparison.
    optimizer : str or callable
        Weighting scheme.  Either a key from ``OPTIMIZERS``
        (``'equal'`` or ``'min_var'``) or a custom callable with signature
        ``f(prices, tickers) -> pd.Series``.  Default ``'equal'``.
    cov_lookback : int
        Trading-day window for covariance estimation (used by ``min_var``).
        Default 60.
    weight_min : float
        Per-stock weight lower bound for ``min_var`` (default 0.01 = 1 %).
    weight_max : float
        Per-stock weight upper bound for ``min_var`` (default 0.15 = 15 %).
    vol_target : float or None
        Annualised volatility target (e.g. 0.15).  ``None`` = disabled.
    vol_lookback : int
        Rolling window for realised-vol estimation (default 60 days).
    vol_max_leverage : float
        Hard cap on the vol scalar (default 2.0).
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
        optimizer: str | Callable = "equal",
        cov_lookback: int = 60,
        weight_min: float = 0.01,
        weight_max: float = 0.15,
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
        self.cov_lookback = cov_lookback
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.vol_max_leverage = vol_max_leverage

        # Resolve optimizer
        if callable(optimizer):
            self._optimizer_fn: Callable = optimizer
            self._optimizer_name = getattr(optimizer, "__name__", "custom")
        elif isinstance(optimizer, str):
            if optimizer not in OPTIMIZERS:
                raise ValueError(
                    f"Unknown optimizer '{optimizer}'. "
                    f"Choose from: {list(OPTIMIZERS.keys())}"
                )
            self._optimizer_fn = OPTIMIZERS[optimizer]  # type: ignore[assignment]
            self._optimizer_name = optimizer
        else:
            raise TypeError(f"optimizer must be str or callable, got {type(optimizer)}")

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
              ``portfolio``    — weighted portfolio daily return
              ``benchmark``    — benchmark daily return (NaN if not provided)
              ``holdings``     — number of stocks held that day
              ``vol_scalar``   — vol-targeting scalar (1.0 when disabled)
              ``realised_vol`` — annualised realised vol used for scaling
        """
        logger.info(
            "Backtester: optimizer=%s  n_stocks=%d  rebalance=%s",
            self._optimizer_name, self.n_stocks, self.rebalance_freq,
        )
        rebal_dates = self._build_rebalance_schedule()
        self._results = self._simulate(rebal_dates)
        return self._results

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call .run() first.")
        return self._results

    @property
    def rebalance_log(self) -> pd.DataFrame:
        """DataFrame with one row per rebalancing event."""
        return pd.DataFrame(self._rebalance_log)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_rebalance_schedule(self) -> pd.DatetimeIndex:
        return pd.date_range(
            start=self.prices.index[0],
            end=self.prices.index[-1],
            freq=self.rebalance_freq,
        )

    def _compute_weights(
        self,
        prices_so_far: pd.DataFrame,
        holdings: pd.Index,
    ) -> pd.Series:
        """
        Delegate weight computation to the configured optimizer.

        For ``equal`` this is trivially 1/N.
        For ``min_var`` we pass the full price history available so far
        so the optimizer can slice its own lookback window.
        """
        tickers = list(holdings)

        if self._optimizer_name == "min_var":
            return minimum_variance(
                prices=prices_so_far,
                tickers=tickers,
                cov_lookback=self.cov_lookback,
                weight_min=self.weight_min,
                weight_max=self.weight_max,
            )
        # equal or custom callable
        return self._optimizer_fn(prices_so_far, tickers)

    def _simulate(self, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        daily_ret: list[dict] = []
        current_holdings: pd.Index = pd.Index([])
        current_weights: pd.Series = pd.Series(dtype=float)
        next_rebal_idx = 0

        price_returns = self.prices.pct_change()

        bench_pct: Optional[pd.Series] = None
        if self.benchmark_prices is not None:
            bench_pct = self.benchmark_prices.pct_change()

        raw_ret_buffer: list[float] = []

        for i, date in enumerate(self.prices.index):

            # ── Rebalance ─────────────────────────────────────────────
            if next_rebal_idx < len(rebal_dates) and date >= rebal_dates[next_rebal_idx]:
                prices_so_far = self.prices.loc[:date]
                scores = self.scorer.score(
                    prices=prices_so_far,
                    fundamentals=self.fundamentals,
                    as_of=date,
                )
                if not scores.empty:
                    new_holdings = self.scorer.select_portfolio(scores, self.n_stocks)
                    if len(new_holdings) == 0:
                        next_rebal_idx += 1
                        continue
                    new_weights  = self._compute_weights(prices_so_far, new_holdings)

                    turnover = self._compute_turnover(current_holdings, new_holdings)
                    current_holdings = new_holdings
                    current_weights  = new_weights

                    self._rebalance_log.append({
                        "date":       date,
                        "n_holdings": len(current_holdings),
                        "turnover":   turnover,
                        "tickers":    list(current_holdings),
                        "weights":    current_weights.to_dict(),
                        "optimizer":  self._optimizer_name,
                    })
                next_rebal_idx += 1

            # ── Weighted portfolio return ──────────────────────────────
            raw_ret = np.nan
            if len(current_holdings) > 0 and not current_weights.empty:
                valid = [
                    t for t in current_weights.index
                    if t in price_returns.columns
                ]
                if valid:
                    w = current_weights.loc[valid]
                    w = w / w.sum()   # re-normalise after dropping missing
                    day_rets = price_returns.loc[date, valid]
                    raw_ret = float((w * day_rets).sum())

                    was_rebal = bool(
                        self._rebalance_log
                        and self._rebalance_log[-1]["date"] == date
                    )
                    if was_rebal:
                        raw_ret -= self._rebalance_log[-1]["turnover"] * self.transaction_cost

            # ── Volatility targeting ───────────────────────────────────
            vol_scalar   = 1.0
            realised_vol = np.nan

            if self.vol_target is not None and not np.isnan(raw_ret):
                raw_ret_buffer.append(raw_ret)
                if len(raw_ret_buffer) >= self.vol_lookback:
                    window = raw_ret_buffer[-self.vol_lookback:]
                    realised_vol = float(np.std(window, ddof=1) * np.sqrt(_TRADING_DAYS))
                    if realised_vol > 0:
                        vol_scalar = min(
                            self.vol_target / realised_vol,
                            self.vol_max_leverage,
                        )
            elif not np.isnan(raw_ret):
                raw_ret_buffer.append(raw_ret)

            day_ret = raw_ret * vol_scalar if not np.isnan(raw_ret) else np.nan

            # ── Benchmark ─────────────────────────────────────────────
            bench_ret = np.nan
            if bench_pct is not None and date in bench_pct.index:
                bench_ret = float(bench_pct.loc[date])

            daily_ret.append({
                "date":         date,
                "portfolio":    day_ret,
                "benchmark":    bench_ret,
                "holdings":     len(current_holdings),
                "vol_scalar":   vol_scalar,
                "realised_vol": realised_vol,
            })

        return pd.DataFrame(daily_ret).set_index("date")

    @staticmethod
    def _compute_turnover(old: pd.Index, new: pd.Index) -> float:
        """Fraction of portfolio positions that changed."""
        if len(old) == 0:
            return 1.0
        return len(old.difference(new)) / len(old)