"""
portfolio/metrics.py
--------------------
Compute standard portfolio performance statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.04) -> dict:
    """
    Compute a standard set of performance metrics from a daily return series.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (as decimals, e.g. 0.01 = 1 %).
    risk_free_rate : float
        Annualised risk-free rate (default 4 %).

    Returns
    -------
    dict
        total_return, cagr, volatility, sharpe_ratio, sortino_ratio,
        max_drawdown, calmar_ratio, win_rate, best_day, worst_day.
    """
    r = returns.dropna()

    if r.empty:
        return {}

    # Cumulative wealth
    wealth = (1 + r).cumprod()
    total_return = float(wealth.iloc[-1] - 1)

    # CAGR
    n_years = len(r) / TRADING_DAYS
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # Volatility (annualised)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))

    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside = r[r < daily_rf] - daily_rf
    downside_std = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS))
    sortino = float(excess.mean() * TRADING_DAYS / downside_std) if downside_std > 0 else 0.0

    # Max drawdown
    rolling_max = wealth.cummax()
    drawdowns = wealth / rolling_max - 1
    max_dd = float(drawdowns.min())

    # Calmar ratio
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Other
    win_rate = float((r > 0).mean())
    best_day = float(r.max())
    worst_day = float(r.min())

    return {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "volatility": round(vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "win_rate": round(win_rate * 100, 2),
        "best_day": round(best_day * 100, 2),
        "worst_day": round(worst_day * 100, 2),
    }


def compute_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Compute rolling annualised return and Sharpe ratio.

    Returns
    -------
    pd.DataFrame  columns: rolling_return, rolling_sharpe
    """
    r = returns.dropna()
    rolling_ret = (1 + r).rolling(window).apply(lambda x: x.prod() - 1, raw=True) * 100
    rolling_sharpe = (
        r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    return pd.DataFrame({"rolling_return": rolling_ret, "rolling_sharpe": rolling_sharpe})
