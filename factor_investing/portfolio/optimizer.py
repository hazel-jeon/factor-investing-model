"""
portfolio/optimizer.py
----------------------
Portfolio weight optimizers.

Provides a clean functional interface: each optimizer takes a price panel
and a list of selected tickers, and returns a weight Series that sums to 1.

Available optimizers
--------------------
equal_weight        : 1/N baseline (no optimization).
minimum_variance    : Minimum-variance portfolio via quadratic programming.
                      Uses scipy.optimize.minimize (SLSQP).

Design notes
------------
- All optimizers are look-ahead-safe: only prices UP TO the rebalancing date
  are passed in.
- When optimization fails (singular covariance, infeasible constraints, etc.)
  the function falls back gracefully to equal weights and logs a warning.
- Weight bounds default to [1%, 15%] per stock to prevent extreme concentration.
  The upper bound ensures at least ceil(1/0.15) = 7 stocks are always held.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def equal_weight(
    prices: pd.DataFrame,          # noqa: ARG001  (unused — kept for uniform signature)
    tickers: list[str],
) -> pd.Series:
    """Return a 1/N equal-weight Series for *tickers*."""
    if not tickers:
        return pd.Series(dtype=float, name="weight")
    n = len(tickers)
    return pd.Series(1.0 / n, index=tickers, name="weight")


def minimum_variance(
    prices: pd.DataFrame,
    tickers: list[str],
    cov_lookback: int = 60,
    weight_min: float = 0.01,
    weight_max: float = 0.15,
    annualise_cov: bool = True,
) -> pd.Series:
    """
    Compute minimum-variance portfolio weights.

    Solves:
        min  w' Σ w
        s.t. sum(w) = 1
             weight_min <= w_i <= weight_max  for all i

    where Σ is the sample covariance matrix estimated from the most recent
    ``cov_lookback`` daily returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted-close price panel up to the rebalancing date.
        Rows = trading days, columns = tickers.
    tickers : list[str]
        Subset of columns in *prices* to optimize over.
    cov_lookback : int
        Number of trading days used to estimate the covariance matrix.
        Default 60 (~3 months).  Shorter windows react faster but are noisier.
    weight_min : float
        Lower bound per stock (default 0.01 = 1 %).
    weight_max : float
        Upper bound per stock (default 0.15 = 15 %).
        Ensures diversification: ceil(1/weight_max) >= 7 stocks always held.
    annualise_cov : bool
        Multiply the daily covariance by 252 before optimization.
        Does not change the optimal weights mathematically, but improves
        numerical conditioning of the solver (default True).

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker, summing to 1.
        Falls back to equal weights on any solver failure.
    """
    # Filter to valid tickers present in the price panel
    valid = [t for t in tickers if t in prices.columns]
    if len(valid) < 2:
        logger.warning("minimum_variance: fewer than 2 valid tickers — using equal weight.")
        return equal_weight(prices, tickers)

    # Slice the most recent cov_lookback rows
    ret = prices[valid].pct_change().dropna().iloc[-cov_lookback:]

    if len(ret) < 10:
        logger.warning(
            "minimum_variance: only %d return rows (need >= 10) — using equal weight.",
            len(ret),
        )
        return equal_weight(prices, tickers)

    cov = ret.cov()
    if annualise_cov:
        cov = cov * _TRADING_DAYS

    # Handle missing / degenerate covariance entries
    cov = cov.fillna(0)
    np.fill_diagonal(cov.values, np.maximum(np.diag(cov.values), 1e-8))

    n = len(valid)
    w0 = np.full(n, 1.0 / n)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(weight_min, weight_max)] * n

    try:
        result: OptimizeResult = minimize(
            fun=lambda w: float(w @ cov.values @ w),
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if not result.success:
            logger.warning(
                "minimum_variance solver did not converge (%s) — using equal weight.",
                result.message,
            )
            return equal_weight(prices, tickers)

        weights = pd.Series(result.x, index=valid, name="weight")
        # Re-normalise to correct for floating-point drift
        weights = weights / weights.sum()
        return weights

    except Exception as exc:  # noqa: BLE001
        logger.warning("minimum_variance failed: %s — using equal weight.", exc)
        return equal_weight(prices, tickers)


# ---------------------------------------------------------------------------
# Registry — maps CLI string -> function
# ---------------------------------------------------------------------------

OPTIMIZERS: dict[str, object] = {
    "equal":    equal_weight,
    "min_var":  minimum_variance,
}