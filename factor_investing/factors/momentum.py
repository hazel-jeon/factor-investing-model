"""
factors/momentum.py
-------------------
Momentum factor — multi-lookback ensemble.

Classic spec (Jegadeesh & Titman 1993):
    Single 12-1M lookback -> Score = normalise( return[t-12M : t-1M] )

Ensemble spec (default):
    Z-scores from three lookback windows are combined as a weighted average.

    ┌──────────┬────────┬─────────────────────────────────────────┐
    │ Window   │ Weight │ What it captures                        │
    ├──────────┼────────┼─────────────────────────────────────────┤
    │ 12-1M    │  0.50  │ Long-term trend (Fama-French classic)   │
    │  6-1M    │  0.30  │ Medium-term momentum (faster response)  │
    │  3-1M    │  0.20  │ Short-term momentum (recent strength)   │
    └──────────┴────────┴─────────────────────────────────────────┘

    Two advantages over a single 12-1M signal:
    1. Signal diversification — each window captures price information at a
       different frequency, so noise partially cancels out when combined.
    2. Faster response — the 6-1M / 3-1M windows adapt to trend reversals
       more quickly, reducing drawdown in sharp momentum unwind periods.

Usage:
    # Classic single 12-1M (backward compatible)
    MomentumFactor(ensemble=False)

    # Ensemble (default)
    MomentumFactor()

    # Custom weights
    MomentumFactor(
        lookback_configs=[(12, 1, 0.5), (6, 1, 0.3), (3, 1, 0.2)]
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)

_DAYS_PER_MONTH = 21

# Default ensemble config: (lookback_months, skip_months, weight)
_DEFAULT_ENSEMBLE: list[tuple[int, int, float]] = [
    (12, 1, 0.50),
    ( 6, 1, 0.30),
    ( 3, 1, 0.20),
]


class MomentumFactor(BaseFactor):
    """
    Multi-lookback momentum factor (ensemble by default).

    Parameters
    ----------
    ensemble : bool
        True (default) = ensemble mode, False = single 12-1M classic.
    lookback_configs : list of (lookback_months, skip_months, weight)
        Custom ensemble configuration.  Only used when ensemble=True.
    lookback_months : int
        Lookback period for single mode (default 12).
    skip_months : int
        Short-term reversal avoidance window for single mode (default 1).
    """

    name = "momentum"

    def __init__(
        self,
        ensemble: bool = True,
        lookback_configs: Optional[list[tuple[int, int, float]]] = None,
        # Single-mode parameters (backward compatible)
        lookback_months: int = 12,
        skip_months: int = 1,
    ) -> None:
        self.ensemble = ensemble

        if ensemble:
            configs = lookback_configs or _DEFAULT_ENSEMBLE
            # Normalise weights so they sum to 1
            total_w = sum(w for _, _, w in configs)
            self.configs = [(lb, sk, w / total_w) for lb, sk, w in configs]
            logger.debug(
                "MomentumFactor ensemble: %s",
                [(lb, sk, f"{w:.2f}") for lb, sk, w in self.configs],
            )
        else:
            # Single lookback — preserve original behaviour
            self.configs = [(lookback_months, skip_months, 1.0)]

    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        as_of: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted-close price panel (rows = dates, columns = tickers).
        as_of : pd.Timestamp, optional
            Reference date.  Defaults to the last row in prices.

        Returns
        -------
        pd.Series
            Ensemble momentum scores indexed by ticker.  Higher = stronger momentum.
        """
        if as_of is None:
            as_of = prices.index[-1]

        prices_so_far = prices.loc[:as_of]

        component_scores: list[pd.Series] = []
        weights: list[float] = []

        for lookback_months, skip_months, weight in self.configs:
            score = self._single_momentum(
                prices_so_far,
                lookback_months=lookback_months,
                skip_months=skip_months,
            )
            if score is not None and not score.empty:
                component_scores.append(score)
                weights.append(weight)

        if not component_scores:
            logger.warning("MomentumFactor: no valid component scores as of %s", as_of)
            return pd.Series(dtype=float, name=self.name)

        # Intersect to tickers present in all components
        common_idx = component_scores[0].index
        for s in component_scores[1:]:
            common_idx = common_idx.intersection(s.index)

        if common_idx.empty:
            return pd.Series(dtype=float, name=self.name)

        # Weighted sum (re-normalised)
        total_w = sum(weights)
        composite = sum(
            s.loc[common_idx] * (w / total_w)
            for s, w in zip(component_scores, weights)
        )

        # Final cross-sectional z-score
        result = self.cross_sectional_zscore(composite)
        return result.rename(self.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_momentum(
        self,
        prices: pd.DataFrame,
        lookback_months: int,
        skip_months: int,
    ) -> Optional[pd.Series]:
        """
        Return a normalised return series for a single lookback window.
        Returns None when there is insufficient price history.
        """
        lookback_days = lookback_months * _DAYS_PER_MONTH
        skip_days     = skip_months     * _DAYS_PER_MONTH

        n = len(prices)
        if n < lookback_days + skip_days:
            logger.debug(
                "MomentumFactor(%dM-%dM): only %d rows, need %d",
                lookback_months, skip_months, n, lookback_days + skip_days,
            )
            return None

        # Price at t-skip and t-lookback
        idx_t    = -(skip_days)
        idx_base = -(lookback_days + skip_days)

        price_t    = prices.iloc[idx_t]
        price_base = prices.iloc[idx_base]

        ret = (price_t / price_base - 1).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ret) < 5:
            return None

        return self.normalise(ret)