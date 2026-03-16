# Factor Investing Model

A modular, research-grade factor investing framework for US equities built on
**yfinance**, **pandas**, and **scipy**.

Implements classic Fama-French factors with a walk-forward backtesting engine,
minimum-variance portfolio optimization, and a full performance visualization suite.

---

## Results

Backtest period: **2015-01-01 → 2024-12-31** · Universe: S&P 500 sample (90 tickers)  
Rebalancing: quarterly · Transaction cost: 0.1% one-way · Portfolio size: 20 stocks

### Sharpe ratio progression

Each row is a cumulative improvement over the previous best configuration.

| # | Configuration | Sharpe | Δ | Note |
|---|---------------|-------:|---|------|
| 1 | Baseline — Value + Momentum + Size, equal weight | 0.717 | — | Starting point |
| 2 | + Volatility targeting (15 % target) | 0.580 | −0.137 | Scaled down Momentum alpha in high-vol regimes |
| 3 | + Sector neutralization | 0.570 | −0.147 | Diluted factor signal with noisy fundamentals |
| 4 | Momentum only, equal weight | 0.728 | +0.011 | Removed Value/Size noise from static fundamentals |
| 5 | **Momentum ensemble** (12-1M × 0.5 + 6-1M × 0.3 + 3-1M × 0.2) | 0.766 | +0.049 | Multi-lookback diversification |
| 6 | **Momentum ensemble + minimum-variance weights** | **0.773** | **+0.056** | Best configuration |

Benchmark (SPY buy-and-hold): **0.558**  
Best model outperforms benchmark by **+0.215 Sharpe points**.

### Key findings

**Momentum is the dominant alpha source.**
Value and Size factors added noise rather than signal. The root cause is data
quality: `yfinance` provides a single point-in-time fundamental snapshot, so
P/B and P/E ratios used in the 2015 backtest are actually 2024 values — a
look-ahead bias that corrupts those two factors. Momentum uses only price
history and is unaffected.

**Multi-lookback ensemble improves signal purity.**
Combining 12-1M, 6-1M, and 3-1M return windows reduces idiosyncratic noise
through diversification across time horizons. Each window captures a distinct
frequency of price trend.

**Minimum-variance weights outperform equal weights marginally.**
The covariance-based optimizer reduces realized volatility and improves
risk-adjusted returns, but the gain is modest (+0.007 Sharpe) because the
selected stocks are already relatively uncorrelated at the 20-stock level.

**Volatility targeting and sector neutralization hurt this model.**
Both techniques are designed to remove market-regime and sector-tilt risks.
However, Momentum alpha in this universe is concentrated in exactly those
high-volatility, sector-trending regimes — so scaling them down or
neutralizing them removes the signal along with the noise.

---

## Project Structure

```
factor-investing/
├── factor_investing/               # Main package
│   ├── data/
│   │   └── loader.py               # yfinance price, fundamental, sector fetcher
│   ├── factors/
│   │   ├── base.py                 # BaseFactor: z-score, winsorise, sector_neutralize
│   │   ├── value.py                # ValueFactor    (P/B + P/E)
│   │   ├── momentum.py             # MomentumFactor (multi-lookback ensemble)
│   │   ├── size.py                 # SizeFactor     (log market cap)
│   │   └── scorer.py               # FactorScorer   (composite + sector neutralization)
│   ├── portfolio/
│   │   ├── backtester.py           # Walk-forward backtest engine
│   │   ├── optimizer.py            # equal_weight, minimum_variance
│   │   └── metrics.py              # CAGR, Sharpe, MDD, Sortino, Calmar
│   └── visualization/
│       └── plots.py                # matplotlib dashboard (5 chart types)
├── notebooks/
│   └── factor_investing_walkthrough.ipynb
├── tests/
│   └── test_factors.py             # 35+ pytest unit tests
├── results/                        # Auto-created on first run
├── run_backtest.py                 # CLI entry point
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

### 2. Run the best configuration

```bash
python run_backtest.py \
  --value-weight 0 --mom-weight 1 --size-weight 0 \
  --optimizer min_var
```

### 3. Full CLI reference

```bash
python run_backtest.py \
  --start        2015-01-01 \   # backtest start date
  --end          2024-12-31 \   # backtest end date
  --n-stocks     20         \   # portfolio size
  --rebalance    QS         \   # QS=quarterly, MS=monthly, AS=annual
  --value-weight 0.0        \   # Value factor weight
  --mom-weight   1.0        \   # Momentum factor weight
  --size-weight  0.0        \   # Size factor weight
  --txn-cost     0.001      \   # one-way transaction cost
  --optimizer    min_var    \   # equal | min_var  (prints comparison table)
  --cov-lookback 60         \   # covariance window for min_var (trading days)
  --weight-min   0.01       \   # per-stock lower bound for min_var
  --weight-max   0.15       \   # per-stock upper bound for min_var
  --sector-neutral          \   # enable sector neutralization
  --vol-target   0.15       \   # annualised vol target (omit to disable)
  --vol-lookback 60         \   # realised-vol estimation window
  --no-plot                     # skip chart generation
```

When `--optimizer min_var` is set, the script runs **both** equal-weight and
minimum-variance backtests and prints a side-by-side comparison table.

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
| `dashboard.png` | 4-panel performance overview (cumulative return, drawdown, annual returns, rolling Sharpe) |
| `factor_scores.png` | Factor scores for top holdings at the latest rebalancing |
| `daily_returns.csv` | Daily portfolio and benchmark returns |
| `daily_returns_min_var.csv` | Daily returns for the min-var run (when `--optimizer min_var`) |
| `metrics.csv` | Summary statistics (CAGR, Sharpe, MDD, …) |

---

## Design Notes

**No look-ahead bias** — the backtester only uses price data available up to
each rebalancing date. Fundamental data is a static yfinance snapshot; for a
production system, replace with point-in-time fundamentals (e.g. Compustat,
Sharadar).

**Winsorise → z-score pipeline** — raw factor values are winsorised at the
1st / 99th percentile before cross-sectional standardisation, reducing the
impact of outliers.

**Momentum ensemble** — three independent lookback windows (12-1M, 6-1M,
3-1M) are each normalised and combined with fixed weights. The final composite
is z-scored once more across the universe. This reduces noise from any single
window without requiring parameter optimization.

**Minimum-variance optimizer** — uses `scipy.optimize.minimize` (SLSQP) on
the trailing 60-day sample covariance matrix. Weight bounds of [1%, 15%]
ensure diversification (at least 7 stocks always held at meaningful size).
Falls back to equal weights gracefully when the solver fails.

**Sector neutralization** — `BaseFactor.sector_neutralize()` re-expresses
factor scores as within-sector z-scores, removing sector-tilt risk. GICS
sector labels are sourced from a built-in table for the S&P 500 sample
universe (instant, no API call) with yfinance fallback for unknown tickers.

**Transaction costs** — applied as a flat one-way cost proportional to
portfolio turnover at each rebalance.

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
        roe = fundamentals["returnOnEquity"].dropna()
        roe = roe[(roe > -1) & (roe < 5)]  # remove outliers
        return self.normalise(roe).rename(self.name)
```

Plug it into the scorer:

```python
from factor_investing.factors.quality import QualityFactor

scorer = FactorScorer([
    (MomentumFactor(), 0.7),
    (QualityFactor(),  0.3),
])
```

### Add a custom optimizer

```python
from factor_investing.portfolio.optimizer import OPTIMIZERS
import pandas as pd

def risk_parity(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Inverse-volatility weighting."""
    ret = prices[tickers].pct_change().dropna().iloc[-60:]
    inv_vol = 1.0 / ret.std().replace(0, float("nan"))
    weights = inv_vol / inv_vol.sum()
    return weights.rename("weight")

OPTIMIZERS["risk_parity"] = risk_parity
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [yfinance](https://github.com/ranaroussi/yfinance) | Price and fundamental data |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
| [numpy](https://numpy.org/) | Numerical computing |
| [scipy](https://scipy.org/) | Portfolio optimization (SLSQP) |
| [matplotlib](https://matplotlib.org/) | Visualization |
| [pytest](https://pytest.org/) | Unit testing |

---

## Disclaimer

This project is for **educational and research purposes only**. It is not
financial advice. Past backtest performance does not guarantee future results.
Always conduct your own due diligence before making investment decisions.

---

## License

MIT