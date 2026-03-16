"""
tests/test_factors.py
---------------------
Unit tests for the factor computation modules.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_investing.factors import ValueFactor, MomentumFactor, SizeFactor, FactorScorer
from factor_investing.factors.base import BaseFactor
from factor_investing.portfolio.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fundamentals() -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "BRK-B", "JNJ", "XOM", "JPM"]
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "market_cap":    rng.uniform(1e10, 3e12, len(tickers)),
            "book_value":    rng.uniform(5, 200, len(tickers)),
            "trailing_pe":   rng.uniform(8, 60, len(tickers)),
            "price_to_book": rng.uniform(0.5, 15, len(tickers)),
        },
        index=tickers,
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "BRK-B", "JNJ", "XOM", "JPM"]
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, (300, len(tickers))), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# BaseFactor utilities
# ---------------------------------------------------------------------------

class DummyFactor(BaseFactor):
    name = "dummy"
    def compute(self, **kwargs):
        return pd.Series(dtype=float)


def test_zscore():
    f = DummyFactor()
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = f.cross_sectional_zscore(s)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std(ddof=1) - 1.0) < 1e-10


def test_winsorise():
    f = DummyFactor()
    s = pd.Series(list(range(100)))
    w = f.winsorise(s, lower=0.05, upper=0.95)
    assert w.min() >= s.quantile(0.05) - 1e-9
    assert w.max() <= s.quantile(0.95) + 1e-9


# ---------------------------------------------------------------------------
# ValueFactor
# ---------------------------------------------------------------------------

def test_value_factor_returns_series(sample_fundamentals):
    factor = ValueFactor()
    scores = factor.compute(fundamentals=sample_fundamentals)
    assert isinstance(scores, pd.Series)
    assert scores.name == "value"
    assert len(scores) > 0


def test_value_factor_weights_normalised():
    f = ValueFactor(pb_weight=3, pe_weight=1)
    assert abs(f.pb_weight + f.pe_weight - 1.0) < 1e-9


def test_value_factor_higher_score_for_cheaper(sample_fundamentals):
    """A stock with lower P/B should score higher than one with higher P/B, all else equal."""
    data = pd.DataFrame(
        {"price_to_book": [1.0, 10.0], "trailing_pe": [15.0, 15.0]},
        index=["cheap", "expensive"],
    )
    f = ValueFactor()
    s = f.compute(fundamentals=data)
    assert s["cheap"] > s["expensive"]


# ---------------------------------------------------------------------------
# MomentumFactor
# ---------------------------------------------------------------------------

def test_momentum_factor_returns_series(sample_prices):
    factor = MomentumFactor()
    scores = factor.compute(prices=sample_prices)
    assert isinstance(scores, pd.Series)
    assert scores.name == "momentum"


def test_momentum_factor_insufficient_history():
    tiny = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="B"),
    )
    f = MomentumFactor()
    scores = f.compute(prices=tiny)
    # Should return empty or near-empty — no crash
    assert isinstance(scores, pd.Series)


# ---------------------------------------------------------------------------
# SizeFactor
# ---------------------------------------------------------------------------

def test_size_factor_returns_series(sample_fundamentals):
    factor = SizeFactor()
    scores = factor.compute(fundamentals=sample_fundamentals)
    assert isinstance(scores, pd.Series)
    assert scores.name == "size"


def test_size_factor_smaller_scores_higher():
    data = pd.DataFrame(
        {"market_cap": [1e9, 1e12]},  # small vs mega
        index=["small", "large"],
    )
    f = SizeFactor()
    s = f.compute(fundamentals=data)
    assert s["small"] > s["large"]


def test_size_factor_drops_negative_mcap():
    data = pd.DataFrame({"market_cap": [-1e9, 5e10, 1e12]}, index=["neg", "mid", "big"])
    f = SizeFactor()
    s = f.compute(fundamentals=data)
    assert "neg" not in s.index


# ---------------------------------------------------------------------------
# FactorScorer
# ---------------------------------------------------------------------------

def test_scorer_composite_column_present(sample_prices, sample_fundamentals):
    scorer = FactorScorer([
        (ValueFactor(), 0.4),
        (MomentumFactor(), 0.4),
        (SizeFactor(), 0.2),
    ])
    scores = scorer.score(prices=sample_prices, fundamentals=sample_fundamentals)
    assert "composite" in scores.columns


def test_scorer_select_portfolio(sample_prices, sample_fundamentals):
    scorer = FactorScorer([
        (ValueFactor(), 0.4),
        (MomentumFactor(), 0.4),
        (SizeFactor(), 0.2),
    ])
    scores = scorer.score(prices=sample_prices, fundamentals=sample_fundamentals)
    top = scorer.select_portfolio(scores, n_stocks=5)
    assert len(top) <= 5


def test_scorer_raises_with_no_factors():
    with pytest.raises(ValueError):
        FactorScorer([])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_keys():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.012, 252),
                        index=pd.date_range("2023-01-01", periods=252, freq="B"))
    m = compute_metrics(returns)
    for key in ["total_return", "cagr", "volatility", "sharpe_ratio", "max_drawdown"]:
        assert key in m


def test_compute_metrics_empty():
    m = compute_metrics(pd.Series(dtype=float))
    assert m == {}


def test_max_drawdown_is_negative_or_zero():
    rng = np.random.default_rng(99)
    returns = pd.Series(rng.normal(0.0005, 0.015, 500),
                        index=pd.date_range("2020-01-01", periods=500, freq="B"))
    m = compute_metrics(returns)
    assert m["max_drawdown"] <= 0


# ---------------------------------------------------------------------------
# Volatility Targeting
# ---------------------------------------------------------------------------

def test_vol_targeting_reduces_volatility(sample_prices, sample_fundamentals):
    """Vol-targeted portfolio should have lower realised vol than the baseline."""
    from factor_investing.portfolio.backtester import Backtester

    scorer = FactorScorer([
        (ValueFactor(), 0.4),
        (MomentumFactor(), 0.4),
        (SizeFactor(), 0.2),
    ])

    common = dict(
        scorer=scorer,
        prices=sample_prices,
        fundamentals=sample_fundamentals,
        rebalance_freq="QS",
        n_stocks=5,
        transaction_cost=0.001,
    )

    bt_base = Backtester(**common, vol_target=None)
    bt_vt   = Backtester(**common, vol_target=0.10, vol_lookback=30, vol_max_leverage=2.0)

    ret_base = bt_base.run()["portfolio"].dropna()
    ret_vt   = bt_vt.run()["portfolio"].dropna()

    vol_base = ret_base.std(ddof=1) * np.sqrt(252)
    vol_vt   = ret_vt.std(ddof=1) * np.sqrt(252)

    assert vol_vt < vol_base, (
        f"Vol-targeted vol ({vol_vt:.3f}) should be < baseline ({vol_base:.3f})"
    )


def test_vol_targeting_columns_present(sample_prices, sample_fundamentals):
    """Results DataFrame must have vol_scalar and realised_vol columns."""
    from factor_investing.portfolio.backtester import Backtester

    scorer = FactorScorer([
        (ValueFactor(), 0.5),
        (MomentumFactor(), 0.3),
        (SizeFactor(), 0.2),
    ])
    bt = Backtester(
        scorer=scorer,
        prices=sample_prices,
        fundamentals=sample_fundamentals,
        rebalance_freq="QS",
        n_stocks=5,
        vol_target=0.15,
        vol_lookback=30,
    )
    results = bt.run()
    assert "vol_scalar" in results.columns
    assert "realised_vol" in results.columns


def test_vol_scalar_respects_max_leverage(sample_prices, sample_fundamentals):
    """vol_scalar must never exceed vol_max_leverage."""
    from factor_investing.portfolio.backtester import Backtester

    max_lev = 1.5
    scorer = FactorScorer([
        (ValueFactor(), 0.4),
        (MomentumFactor(), 0.4),
        (SizeFactor(), 0.2),
    ])
    bt = Backtester(
        scorer=scorer,
        prices=sample_prices,
        fundamentals=sample_fundamentals,
        rebalance_freq="QS",
        n_stocks=5,
        vol_target=0.30,     # high target → tends to push scalar up
        vol_lookback=20,
        vol_max_leverage=max_lev,
    )
    results = bt.run()
    scalars = results["vol_scalar"].dropna()
    assert scalars.max() <= max_lev + 1e-9, (
        f"vol_scalar exceeded max_leverage: {scalars.max():.4f} > {max_lev}"
    )


def test_vol_targeting_disabled_gives_scalar_one(sample_prices, sample_fundamentals):
    """When vol_target=None every vol_scalar should be exactly 1.0."""
    from factor_investing.portfolio.backtester import Backtester

    scorer = FactorScorer([
        (ValueFactor(), 0.5),
        (MomentumFactor(), 0.3),
        (SizeFactor(), 0.2),
    ])
    bt = Backtester(
        scorer=scorer,
        prices=sample_prices,
        fundamentals=sample_fundamentals,
        rebalance_freq="QS",
        n_stocks=5,
        vol_target=None,
    )
    results = bt.run()
    assert (results["vol_scalar"] == 1.0).all()


# ---------------------------------------------------------------------------
# Sector Neutralization
# ---------------------------------------------------------------------------

def test_sector_neutralize_zero_mean_per_sector():
    """Within each sector, the neutralized scores must have mean ~0."""
    scores = pd.Series({
        "AAPL": 1.5, "MSFT": 0.8, "NVDA": -0.3,   # Tech
        "JPM":  1.2, "BAC": -0.5, "GS":   0.1,     # Financials
        "JNJ": -1.0, "UNH":  0.6, "MRK":  0.2,     # Health Care
    })
    sector_map = pd.Series({
        "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
        "JNJ": "Health Care", "UNH": "Health Care", "MRK": "Health Care",
    })
    from factor_investing.factors.base import BaseFactor

    result = BaseFactor.sector_neutralize(scores, sector_map)

    for sector in ["Tech", "Financials", "Health Care"]:
        members = sector_map[sector_map == sector].index
        sector_scores = result.loc[members].dropna()
        assert abs(sector_scores.mean()) < 1e-9, (
            f"Sector '{sector}' mean = {sector_scores.mean():.6f}, expected ~0"
        )


def test_sector_neutralize_unit_std_per_sector():
    """Within each sector (with >=3 stocks), std should be ~1."""
    scores = pd.Series({t: float(i) for i, t in enumerate(
        ["A", "B", "C", "D", "E", "F"]
    )})
    sector_map = pd.Series({"A": "X", "B": "X", "C": "X",
                             "D": "Y", "E": "Y", "F": "Y"})
    from factor_investing.factors.base import BaseFactor

    result = BaseFactor.sector_neutralize(scores, sector_map)

    for sector in ["X", "Y"]:
        members = sector_map[sector_map == sector].index
        std = result.loc[members].dropna().std(ddof=1)
        assert abs(std - 1.0) < 1e-9, (
            f"Sector '{sector}' std = {std:.6f}, expected 1.0"
        )


def test_sector_neutralize_small_sector_returns_nan():
    """Sectors below min_sector_size should produce NaN, not inflated z-scores."""
    scores     = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    sector_map = pd.Series({"A": "Big", "B": "Big", "C": "Big", "D": "Tiny"})
    from factor_investing.factors.base import BaseFactor

    result = BaseFactor.sector_neutralize(scores, sector_map, min_sector_size=3)

    # "Tiny" sector has only 1 member -> should be NaN
    assert pd.isna(result["D"]), "Single-member sector should yield NaN"
    # "Big" sector has 3 members -> should be scored
    assert result[["A", "B", "C"]].notna().all()


def test_scorer_sector_neutral_columns(sample_prices, sample_fundamentals):
    """When sector_map is passed, DataFrame should contain a 'sector' column."""
    sector_map = pd.Series(
        {t: ("Tech" if i % 2 == 0 else "Finance")
         for i, t in enumerate(sample_prices.columns)},
        name="sector",
    )
    scorer = FactorScorer([
        (ValueFactor(), 0.4),
        (MomentumFactor(), 0.4),
        (SizeFactor(), 0.2),
    ])
    scores = scorer.score(
        prices=sample_prices,
        fundamentals=sample_fundamentals,
        sector_map=sector_map,
    )
    assert "sector" in scores.columns
    assert "composite" in scores.columns


def test_sector_neutral_scores_differ_from_plain(sample_prices, sample_fundamentals):
    """Sector-neutral composite scores should differ from plain composite scores."""
    sector_map = pd.Series(
        {t: ("Tech" if i % 2 == 0 else "Finance")
         for i, t in enumerate(sample_prices.columns)},
        name="sector",
    )
    scorer = FactorScorer([
        (ValueFactor(), 0.4),
        (MomentumFactor(), 0.4),
        (SizeFactor(), 0.2),
    ])
    plain  = scorer.score(prices=sample_prices, fundamentals=sample_fundamentals)
    neutral = scorer.score(
        prices=sample_prices,
        fundamentals=sample_fundamentals,
        sector_map=sector_map,
    )
    common = plain.index.intersection(neutral.index)
    # At least some composite scores should differ
    diff = (plain.loc[common, "composite"] - neutral.loc[common, "composite"]).abs()
    assert diff.max() > 1e-6, "Sector-neutral and plain scores are identical — neutralization had no effect"