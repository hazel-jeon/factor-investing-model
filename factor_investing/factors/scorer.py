"""
factors/scorer.py
-----------------
Combines individual factor scores into a single composite score
and selects the top-N portfolio.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)


class FactorScorer:
    """
    Aggregates multiple :class:`BaseFactor` instances into a composite score.

    Parameters
    ----------
    factors : list of (BaseFactor, float) tuples
        Each tuple is (factor_instance, weight).  Weights are auto-normalised.

    Example
    -------
    >>> scorer = FactorScorer([
    ...     (ValueFactor(), 0.4),
    ...     (MomentumFactor(), 0.4),
    ...     (SizeFactor(), 0.2),
    ... ])
    >>> composite = scorer.score(prices=prices, fundamentals=fundamentals)
    """

    def __init__(self, factors: list[tuple[BaseFactor, float]]) -> None:
        if not factors:
            raise ValueError("Provide at least one factor.")
        total_w = sum(w for _, w in factors)
        self.factors = [(f, w / total_w) for f, w in factors]

    def score(
        self,
        prices: Optional[pd.DataFrame] = None,
        fundamentals: Optional[pd.DataFrame] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute per-factor and composite scores for the current rebalancing date.

        Returns
        -------
        pd.DataFrame
            Columns = individual factor names + ``composite``.
            Index   = ticker symbols.
        """
        kwargs = dict(
            prices=prices,
            fundamentals=fundamentals,
            as_of=as_of,
        )

        raw_scores: dict[str, pd.Series] = {}
        for factor, _ in self.factors:
            try:
                s = factor.compute(**kwargs)
                if not s.empty:
                    raw_scores[factor.name] = s
                else:
                    logger.warning("Factor '%s' returned empty series.", factor.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Factor '%s' failed: %s", factor.name, exc)

        if not raw_scores:
            return pd.DataFrame()

        score_df = pd.DataFrame(raw_scores)

        # Composite = weighted average, ignoring NaN for a given stock
        composite = pd.Series(0.0, index=score_df.index)
        weight_sum = pd.Series(0.0, index=score_df.index)

        for factor, w in self.factors:
            if factor.name in score_df.columns:
                col = score_df[factor.name]
                composite += col.fillna(0) * w
                weight_sum += col.notna().astype(float) * w

        # Re-scale so weights always sum to 1 per stock
        weight_sum = weight_sum.replace(0, float("nan"))
        composite = composite / weight_sum

        score_df["composite"] = composite
        return score_df.sort_values("composite", ascending=False)

    def select_portfolio(
        self,
        scores: pd.DataFrame,
        n_stocks: int = 20,
        long_only: bool = True,
    ) -> pd.Index:
        """
        Return the tickers of the top-N stocks by composite score.

        Parameters
        ----------
        scores : pd.DataFrame
            Output of :meth:`score`.
        n_stocks : int
            Number of stocks to hold (default 20).
        long_only : bool
            If True return top-N; if False also return bottom-N as short leg.
        """
        valid = scores["composite"].dropna()
        top = valid.nlargest(n_stocks).index

        if long_only:
            return top

        bottom = valid.nsmallest(n_stocks).index
        return top.append(bottom)
