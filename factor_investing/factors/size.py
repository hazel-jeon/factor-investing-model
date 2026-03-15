"""
factors/size.py
---------------
Size factor: SMALL-cap stocks score HIGH (Fama-French SMB convention).

Score = normalise( -log(market_cap) )

Using log market cap reduces the influence of mega-cap outliers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)


class SizeFactor(BaseFactor):
    """
    Market-cap size factor (small = high score).

    Parameters
    ----------
    use_log : bool
        Apply log transform before normalising (recommended, default True).
    """

    name = "size"

    def __init__(self, use_log: bool = True) -> None:
        self.use_log = use_log

    def compute(self, fundamentals: pd.DataFrame, **kwargs) -> pd.Series:  # type: ignore[override]
        """
        Parameters
        ----------
        fundamentals : pd.DataFrame
            Must contain column ``market_cap``.

        Returns
        -------
        pd.Series
            Size scores indexed by ticker.  Higher = smaller company.
        """
        if "market_cap" not in fundamentals.columns:
            logger.warning("SizeFactor: 'market_cap' column not found.")
            return pd.Series(dtype=float, name=self.name)

        mcap = fundamentals["market_cap"].dropna()
        mcap = mcap[mcap > 0]

        if mcap.empty:
            return pd.Series(dtype=float, name=self.name)

        raw = -np.log(mcap) if self.use_log else -mcap
        return self.normalise(raw).rename(self.name)
