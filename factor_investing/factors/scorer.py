"""
factors/scorer.py
-----------------
Combines individual factor scores into a single composite score
and selects the top-N portfolio.

Sector neutralization (optional)
---------------------------------
When a ``sector_map`` (pd.Series: ticker -> sector string) is supplied to
:meth:`score`, each raw factor series is re-expressed as within-sector
z-scores BEFORE the weighted composite is computed.  This removes sector
tilts so the model selects the best stocks *within* each sector rather than
simply overweighting whichever sector happens to screen well on a given date.

The neutralization is applied per-factor independently, then the composite
is formed from the neutralized scores.  An extra column ``sector`` is
appended to the returned DataFrame for diagnostics.
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
    >>> # Plain composite
    >>> scores = scorer.score(prices=prices, fundamentals=fundamentals)
    >>> # Sector-neutral composite
    >>> scores = scorer.score(prices=prices, fundamentals=fundamentals,
    ...                       sector_map=sector_map)
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
        sector_map: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute per-factor and composite scores for the current rebalancing date.

        Parameters
        ----------
        prices : pd.DataFrame, optional
        fundamentals : pd.DataFrame, optional
        as_of : pd.Timestamp, optional
        sector_map : pd.Series, optional
            ticker -> sector string mapping.  When provided, each factor score
            is sector-neutralized before the composite is formed.

        Returns
        -------
        pd.DataFrame
            Columns = individual factor names + ``composite`` [+ ``sector``].
            Index   = ticker symbols, sorted descending by composite score.
        """
        kwargs = dict(prices=prices, fundamentals=fundamentals, as_of=as_of)

        # 1. Compute raw factor scores
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

        # 2. Sector neutralization (applied per-factor before aggregation)
        if sector_map is not None:
            neutralized: dict[str, pd.Series] = {}
            for fname, fseries in raw_scores.items():
                try:
                    neutralized[fname] = BaseFactor.sector_neutralize(
                        fseries, sector_map
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Sector neutralization failed for '%s': %s — using raw score.",
                        fname, exc,
                    )
                    neutralized[fname] = fseries
            score_df = pd.DataFrame(neutralized)
        else:
            score_df = pd.DataFrame(raw_scores)

        # 3. Composite = weighted average (NaN-aware)
        composite = pd.Series(0.0, index=score_df.index)
        weight_sum = pd.Series(0.0, index=score_df.index)

        for factor, w in self.factors:
            if factor.name in score_df.columns:
                col = score_df[factor.name]
                composite += col.fillna(0) * w
                weight_sum += col.notna().astype(float) * w

        weight_sum = weight_sum.replace(0, float("nan"))
        composite = composite / weight_sum
        score_df["composite"] = composite

        # 4. Attach sector label for diagnostics
        if sector_map is not None:
            score_df["sector"] = sector_map.reindex(score_df.index).fillna("Unknown")

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