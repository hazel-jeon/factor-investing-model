"""
factors/momentum.py
-------------------
Momentum factor: 12-month total return skipping the most recent month
(standard Jegadeesh & Titman 1993 specification).

Score = normalise( return[t-12M : t-1M] )
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)

# Approximate trading-day constants
_DAYS_PER_MONTH = 21
_LOOKBACK_DAYS = 12 * _DAYS_PER_MONTH   # ~252 trading days
_SKIP_DAYS = 1 * _DAYS_PER_MONTH         # ~21  trading days (skip last month)


class MomentumFactor(BaseFactor):
    """
    12-1 month price momentum.

    Parameters
    ----------
    lookback_months : int
        Total lookback window in months (default 12).
    skip_months : int
        Most-recent months to skip to avoid short-term reversal (default 1).
    """

    name = "momentum"

    def __init__(self, lookback_months: int = 12, skip_months: int = 1) -> None:
        self.lookback_days = lookback_months * _DAYS_PER_MONTH
        self.skip_days = skip_months * _DAYS_PER_MONTH

    def compute(self, prices: pd.DataFrame, as_of: pd.Timestamp | None = None, **kwargs) -> pd.Series:  # type: ignore[override]
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted-close price panel (rows = dates, columns = tickers).
        as_of : pd.Timestamp, optional
            Reference date.  Defaults to the last row in *prices*.

        Returns
        -------
        pd.Series
            Momentum scores indexed by ticker.  Higher = stronger momentum.
        """
        if as_of is None:
            as_of = prices.index[-1]

        # Locate t-1M and t-12M rows
        price_at_t = prices.loc[:as_of].iloc[-(self.skip_days) : -(self.skip_days) + 1]
        price_at_t_12 = prices.loc[:as_of].iloc[-(self.lookback_days) : -(self.lookback_days) + 1]

        if price_at_t.empty or price_at_t_12.empty:
            logger.warning("MomentumFactor: not enough price history as of %s", as_of)
            return pd.Series(dtype=float, name=self.name)

        ret = (price_at_t.iloc[0] / price_at_t_12.iloc[0] - 1).dropna()

        # Need at least 5 stocks to z-score
        if len(ret) < 5:
            return ret.rename(self.name)

        return self.normalise(ret).rename(self.name)
