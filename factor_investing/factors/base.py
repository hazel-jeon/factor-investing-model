"""
factors/base.py
---------------
Abstract base class that every factor must implement.
Enforces a consistent .compute() -> pd.Series interface.
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

    @staticmethod
    def sector_neutralize(
        series: pd.Series,
        sector_map: pd.Series,
        min_sector_size: int = 3,
    ) -> pd.Series:
        """
        Re-express factor scores as within-sector z-scores.

        For each sector, the tickers in that sector are z-scored against
        each other rather than the full universe.  This removes any
        systematic sector tilt from the raw factor signal so that the
        composite score reflects stock-specific alpha, not sector momentum.

        Parameters
        ----------
        series : pd.Series
            Raw (or normalised) factor scores indexed by ticker.
        sector_map : pd.Series
            Maps ticker -> sector string.  Index = tickers.
        min_sector_size : int
            Sectors with fewer than this many valid tickers are skipped
            (their scores are set to NaN rather than inflated z-scores).
            Default 3.

        Returns
        -------
        pd.Series
            Within-sector z-scores, same index as *series*.

        Notes
        -----
        Tickers absent from *sector_map* are grouped into a synthetic
        "Unknown" sector so they are scored relative to each other
        rather than silently dropped.
        """
        common = series.index.intersection(sector_map.index)
        scores = series.loc[common].copy()
        sectors = sector_map.loc[common].fillna("Unknown")

        result = pd.Series(float("nan"), index=common)

        for sector, group_idx in sectors.groupby(sectors).groups.items():
            grp = scores.loc[group_idx].dropna()
            if len(grp) < min_sector_size:
                continue
            mu = grp.mean()
            sigma = grp.std(ddof=1)
            if sigma == 0 or pd.isna(sigma):
                result.loc[grp.index] = 0.0
            else:
                result.loc[grp.index] = (grp - mu) / sigma

        return result