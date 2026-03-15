"""
factors/base.py
---------------
Abstract base class that every factor must implement.
Enforces a consistent .compute() → pd.Series interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFactor(ABC):
    """
    All factors inherit from this class.

    Subclasses must implement :meth:`compute`, which returns a *raw*
    (un-normalised) factor series indexed by ticker.
    """

    name: str = "base"

    @abstractmethod
    def compute(self, **kwargs) -> pd.Series:
        """
        Compute the raw factor scores.

        Returns
        -------
        pd.Series
            Index  = ticker symbols.
            Values = raw factor score (higher = more exposure to the factor).
        """

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cross_sectional_zscore(series: pd.Series) -> pd.Series:
        """Standardise a cross-sectional factor series to zero mean / unit std."""
        mu = series.mean()
        sigma = series.std(ddof=1)
        if sigma == 0 or pd.isna(sigma):
            return series - mu
        return (series - mu) / sigma

    @staticmethod
    def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Clip extreme values at the given quantile boundaries."""
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        return series.clip(lo, hi)

    def normalise(self, series: pd.Series) -> pd.Series:
        """Winsorise then z-score a raw factor series."""
        return self.cross_sectional_zscore(self.winsorise(series))
