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
