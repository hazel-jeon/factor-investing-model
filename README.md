# Factor Investing Model

A modular, research-grade factor investing framework for US equities built on **yfinance** and **pandas**.

Implements three classic Fama-French factors with a walk-forward backtesting engine and a full performance visualisation suite.

---

## Factors

| Factor | Signal | Higher score = |
|--------|--------|----------------|
| **Value** | P/B ratio, trailing P/E | Cheaper stock |
| **Momentum** | 12-1 month price return (Jegadeesh & Titman 1993) | Stronger recent trend |
| **Size** | Log market capitalisation | Smaller company |

Composite score = weighted average of normalised (winsorised z-score) factor scores.

---

## Project Structure

```
factor-investing/
├── factor_investing/          # Main package
│   ├── data/
│   │   └── loader.py          # yfinance price + fundamental fetcher
│   ├── factors/
│   │   ├── base.py            # Abstract BaseFactor (z-score, winsorise)
│   │   ├── value.py           # ValueFactor    (P/B + P/E)
│   │   ├── momentum.py        # MomentumFactor (12-1M return)
│   │   ├── size.py            # SizeFactor     (log market cap)
│   │   └── scorer.py          # FactorScorer   (composite + portfolio selection)
│   ├── portfolio/
│   │   ├── backtester.py      # Walk-forward backtest engine
│   │   └── metrics.py         # CAGR, Sharpe, MDD, Sortino …
│   └── visualization/
│       └── plots.py           # matplotlib charts & dashboard
├── notebooks/
│   └── factor_investing_walkthrough.ipynb
├── tests/
│   └── test_factors.py        # pytest unit tests
├── results/                   # Auto-created on first run
├── run_backtest.py            # CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/factor-investing.git
cd factor-investing
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the backtest

```bash
python run_backtest.py
```

Default settings: S&P 500 sample universe, 2015-01-01 → 2024-12-31, quarterly rebalancing, 20 stocks.

### 3. Customise via CLI flags

```bash
python run_backtest.py \
  --start 2015-01-01 \
  --end   2024-12-31 \
  --n-stocks 20 \
  --rebalance QS \
  --value-weight 0.4 \
  --mom-weight   0.4 \
  --size-weight  0.2 \
  --txn-cost 0.001
```

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | `2015-01-01` | Backtest start date |
| `--end` | `2024-12-31` | Backtest end date |
| `--n-stocks` | `20` | Portfolio size |
| `--rebalance` | `QS` | `QS`=quarterly, `MS`=monthly, `AS`=annual |
| `--value-weight` | `0.4` | Weight for Value factor |
| `--mom-weight` | `0.4` | Weight for Momentum factor |
| `--size-weight` | `0.2` | Weight for Size factor |
| `--txn-cost` | `0.001` | One-way transaction cost (0.1%) |
| `--no-plot` | — | Skip chart generation |

### 4. Interactive notebook

```bash
jupyter notebook notebooks/factor_investing_walkthrough.ipynb
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Output

After running, the `results/` directory contains:

| File | Description |
|------|-------------|
| `dashboard.png` | 4-panel performance overview |
| `factor_scores.png` | Latest factor scores for top holdings |
| `daily_returns.csv` | Daily portfolio & benchmark returns |
| `metrics.csv` | Summary performance statistics |
| `rebalance_log.csv` | Holdings and turnover at each rebalance date |

---

## Key Design Decisions

**No look-ahead bias** — at each rebalancing date the backtester only uses price data available up to that date. Fundamental data is static (yfinance snapshot); for a production system, use point-in-time fundamentals (e.g. Compustat).

**Winsorise → z-score pipeline** — all raw factor values are winsorised at the 1st/99th percentile before cross-sectional standardisation, reducing the impact of outliers.

**Equal weighting** — the portfolio holds each selected stock at equal weight. Value-weighted or factor-tilted weighting can be added in `FactorScorer.select_portfolio`.

**Transaction costs** — applied as a flat one-way cost proportional to portfolio turnover at each rebalance.

---

## Extending the Model

### Add a new factor

```python
# factor_investing/factors/quality.py
from .base import BaseFactor
import pandas as pd

class QualityFactor(BaseFactor):
    name = "quality"

    def compute(self, fundamentals: pd.DataFrame, **kwargs) -> pd.Series:
        roe = fundamentals["return_on_equity"].dropna()
        return self.normalise(roe).rename(self.name)
```

Then plug it into the scorer:

```python
from factor_investing.factors.quality import QualityFactor

scorer = FactorScorer([
    (ValueFactor(),    0.35),
    (MomentumFactor(), 0.35),
    (SizeFactor(),     0.15),
    (QualityFactor(),  0.15),
])
```

---

## Dependencies

- [yfinance](https://github.com/ranaroussi/yfinance) — market data
- [pandas](https://pandas.pydata.org/) — data manipulation
- [numpy](https://numpy.org/) — numerical computing
- [matplotlib](https://matplotlib.org/) — visualisation
- [pytest](https://pytest.org/) — testing

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past backtest performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

## License

MIT
