"""
data/loader.py
--------------
Fetches price history and fundamental data from yfinance.
Provides a clean DataFrame interface for the rest of the pipeline.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S&P 500 universe
# ---------------------------------------------------------------------------

SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "TSLA", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "MRK", "COST",
    "ABBV", "CVX", "CRM", "BAC", "AMD", "NFLX", "PEP", "KO", "ADBE", "TMO",
    "WMT", "ACN", "CSCO", "MCD", "LIN", "ABT", "DHR", "TXN", "NKE", "PM",
    "NEE", "ORCL", "INTC", "HON", "QCOM", "IBM", "AMGN", "INTU", "MS", "GS",
    "CAT", "UPS", "BA", "RTX", "SPGI", "BLK", "GILD", "AXP", "DE", "SYK",
    "T", "MDLZ", "ADI", "REGN", "ISRG", "PLD", "VRTX", "MMM", "GE", "ZTS",
    "SCHW", "C", "WFC", "USB", "ADP", "ELV", "CI", "CVS", "BMY", "LRCX",
    "PANW", "MU", "KLAC", "AMAT", "SNPS", "CDNS", "MCHP", "TGT", "LOW", "F",
]


def get_sp500_tickers() -> list[str]:
    """Return the built-in S&P 500 sample universe."""
    return list(SP500_TICKERS)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: list[str],
    start: str,
    end: str,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download adjusted-close prices for *tickers* between *start* and *end*.

    Returns
    -------
    pd.DataFrame
        index = DatetimeIndex (trading days), columns = ticker symbols.
        Tickers with insufficient data are silently dropped.
    """
    logger.info("Downloading price data for %d tickers …", len(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers are passed
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    prices = prices.dropna(axis=1, how="all")
    logger.info("Price data ready: %d tickers, %d rows", prices.shape[1], prices.shape[0])
    return prices


# ---------------------------------------------------------------------------
# Fundamental data  (market cap, book value, earnings)
# ---------------------------------------------------------------------------

def fetch_fundamental_data(
    tickers: list[str],
    sleep: float = 0.1,
) -> pd.DataFrame:
    """
    Pull key fundamentals from yfinance for each ticker.

    Columns returned
    ----------------
    market_cap      : float  — latest market capitalisation (USD)
    book_value      : float  — book value per share
    trailing_pe     : float  — trailing P/E ratio
    forward_pe      : float  — forward P/E ratio
    price_to_book   : float  — P/B ratio (used to derive value factor)

    Returns
    -------
    pd.DataFrame  with *tickers* as index.
    """
    records: list[dict] = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append(
                {
                    "ticker": ticker,
                    "market_cap": info.get("marketCap"),
                    "book_value": info.get("bookValue"),
                    "trailing_pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "price_to_book": info.get("priceToBook"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "total_revenue": info.get("totalRevenue"),
                }
            )
            time.sleep(sleep)  # be polite to the API
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch fundamentals for %s: %s", ticker, exc)
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    logger.info("Fundamental data ready: %d tickers", len(df))
    return df


# ---------------------------------------------------------------------------
# Sector data
# ---------------------------------------------------------------------------

# GICS sector mapping built-in (no API call needed for the sample universe).
# yfinance is used as a fallback for tickers not in this table.
_SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "CRM":  "Information Technology", "ADBE": "Information Technology",
    "AMD":  "Information Technology", "TXN":  "Information Technology",
    "QCOM": "Information Technology", "INTC": "Information Technology",
    "IBM":  "Information Technology", "INTU": "Information Technology",
    "ADI":  "Information Technology", "LRCX": "Information Technology",
    "PANW": "Information Technology", "MU":   "Information Technology",
    "KLAC": "Information Technology", "AMAT": "Information Technology",
    "SNPS": "Information Technology", "CDNS": "Information Technology",
    "MCHP": "Information Technology",
    # Communication Services
    "GOOGL":"Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "T":    "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD":   "Consumer Discretionary", "MCD":  "Consumer Discretionary",
    "NKE":  "Consumer Discretionary", "TGT":  "Consumer Discretionary",
    "LOW":  "Consumer Discretionary", "F":    "Consumer Discretionary",
    # Consumer Staples
    "PG":   "Consumer Staples", "KO":   "Consumer Staples",
    "PEP":  "Consumer Staples", "COST": "Consumer Staples",
    "WMT":  "Consumer Staples", "MDLZ": "Consumer Staples",
    "PM":   "Consumer Staples",
    # Health Care
    "LLY":  "Health Care", "UNH":  "Health Care", "JNJ":  "Health Care",
    "MRK":  "Health Care", "ABBV": "Health Care", "TMO":  "Health Care",
    "ABT":  "Health Care", "DHR":  "Health Care", "AMGN": "Health Care",
    "GILD": "Health Care", "SYK":  "Health Care", "REGN": "Health Care",
    "ISRG": "Health Care", "VRTX": "Health Care", "ZTS":  "Health Care",
    "ELV":  "Health Care", "CI":   "Health Care", "CVS":  "Health Care",
    "BMY":  "Health Care",
    # Financials
    "BRK-B":"Financials", "JPM":  "Financials", "V":    "Financials",
    "MA":   "Financials", "BAC":  "Financials", "MS":   "Financials",
    "GS":   "Financials", "SPGI": "Financials", "BLK":  "Financials",
    "AXP":  "Financials", "SCHW": "Financials", "C":    "Financials",
    "WFC":  "Financials", "USB":  "Financials", "ADP":  "Financials",
    # Industrials
    "HON":  "Industrials", "CAT":  "Industrials", "UPS":  "Industrials",
    "BA":   "Industrials", "RTX":  "Industrials", "DE":   "Industrials",
    "MMM":  "Industrials", "GE":   "Industrials",
    # Energy
    "XOM":  "Energy", "CVX":  "Energy",
    # Materials
    "LIN":  "Materials",
    # Real Estate
    "PLD":  "Real Estate",
    # Utilities
    "NEE":  "Utilities",
}


def fetch_sector_map(
    tickers: list[str],
    sleep: float = 0.05,
    use_cache: bool = True,
) -> pd.Series:
    """
    Return a Series mapping ticker -> GICS sector string.

    Strategy
    --------
    1. Use the built-in ``_SECTOR_MAP`` table for known tickers (instant, no API).
    2. Fall back to ``yf.Ticker(t).info['sector']`` for any unknown ticker.
    3. Tickers where neither source has data are mapped to ``"Unknown"``.

    Parameters
    ----------
    tickers : list[str]
    sleep : float
        Seconds to wait between yfinance calls for unknown tickers.
    use_cache : bool
        If True (default) skip yfinance for any ticker already in the
        built-in table.  Set False to always refresh from yfinance.

    Returns
    -------
    pd.Series
        index = tickers, values = sector strings.
    """
    result: dict[str, str] = {}
    unknown: list[str] = []

    for t in tickers:
        if use_cache and t in _SECTOR_MAP:
            result[t] = _SECTOR_MAP[t]
        else:
            unknown.append(t)

    if unknown:
        logger.info(
            "Fetching sector data from yfinance for %d unknown tickers …", len(unknown)
        )
        for t in unknown:
            try:
                info = yf.Ticker(t).info
                result[t] = info.get("sector") or "Unknown"
                time.sleep(sleep)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch sector for %s: %s", t, exc)
                result[t] = "Unknown"

    sector_series = pd.Series(result, name="sector").reindex(tickers).fillna("Unknown")
    counts = sector_series.value_counts()
    logger.info("Sector map ready — %d sectors:\n%s", len(counts), counts.to_string())
    return sector_series