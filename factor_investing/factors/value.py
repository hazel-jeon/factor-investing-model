"""
factors/value.py
----------------
Value factor: stocks with LOW price-to-book and LOW trailing P/E receive
high factor scores (cheap = high value exposure).

Score = normalise(-P/B) * pb_weight + normalise(-P/E) * pe_weight
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)


class ValueFactor(BaseFactor):
    """
    Composite value factor using P/B and trailing P/E ratios.

    Parameters
    ----------
    pb_weight : float
        Weight assigned to the P/B component (default 0.5).
    pe_weight : float
        Weight assigned to the P/E component (default 0.5).
    """

    name = "value"

    def __init__(self, pb_weight: float = 0.5, pe_weight: float = 0.5) -> None:
        total = pb_weight + pe_weight
        self.pb_weight = pb_weight / total
        self.pe_weight = pe_weight / total

    def compute(self, fundamentals: pd.DataFrame, **kwargs) -> pd.Series:  # type: ignore[override]
        """
        Parameters
        ----------
        fundamentals : pd.DataFrame
            Must contain columns ``price_to_book`` and ``trailing_pe``.

        Returns
        -------
        pd.Series
            Value scores indexed by ticker.  Higher = cheaper.
        """
        scores = pd.Series(0.0, index=fundamentals.index, name=self.name)

        # --- P/B component ---
        if "price_to_book" in fundamentals.columns:
            pb = fundamentals["price_to_book"].copy()
            pb = pb[pb > 0]  # drop negative / zero book values
            if not pb.empty:
                scores = scores.add(
                    self.normalise(-pb) * self.pb_weight, fill_value=0
                )
        else:
            logger.warning("ValueFactor: 'price_to_book' column not found.")

        # --- P/E component ---
        if "trailing_pe" in fundamentals.columns:
            pe = fundamentals["trailing_pe"].copy()
            pe = pe[(pe > 0) & (pe < 200)]  # remove nonsensical P/E values
            if not pe.empty:
                scores = scores.add(
                    self.normalise(-pe) * self.pe_weight, fill_value=0
                )
        else:
            logger.warning("ValueFactor: 'trailing_pe' column not found.")

        scores = scores.replace(0, np.nan).dropna()
        return scores.rename(self.name)
