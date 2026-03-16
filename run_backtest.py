"""
run_backtest.py
---------------
End-to-end pipeline:
  1. Fetch price + fundamental + sector data
  2. Build factor scorer
  3. Run walk-forward backtest  (baseline, and optionally sector-neutral / vol-targeted)
  4. Print side-by-side Sharpe comparison
  5. Generate charts -> results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from factor_investing.data import (
    fetch_price_data,
    fetch_fundamental_data,
    fetch_sector_map,
    get_sp500_tickers,
)
from factor_investing.factors import ValueFactor, MomentumFactor, SizeFactor, FactorScorer
from factor_investing.portfolio import Backtester, compute_metrics, compute_rolling_metrics
from factor_investing.visualization import plot_performance_dashboard, plot_factor_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Factor Investing Backtest")
    p.add_argument("--start",        default="2015-01-01")
    p.add_argument("--end",          default="2024-12-31")
    p.add_argument("--n-stocks",     type=int,   default=20)
    p.add_argument("--rebalance",    default="QS")
    p.add_argument("--value-weight", type=float, default=0.4)
    p.add_argument("--mom-weight",   type=float, default=0.4)
    p.add_argument("--size-weight",  type=float, default=0.2)
    p.add_argument("--txn-cost",     type=float, default=0.001)
    # Vol targeting
    p.add_argument("--vol-target",       type=float, default=None,  metavar="FLOAT")
    p.add_argument("--vol-lookback",     type=int,   default=60,    metavar="DAYS")
    p.add_argument("--vol-max-leverage", type=float, default=2.0,   metavar="FLOAT")
    # Sector neutralization
    p.add_argument(
        "--sector-neutral",
        action="store_true",
        help=(
            "Run a second backtest with sector-neutral factor scores "
            "and print a side-by-side Sharpe comparison."
        ),
    )
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def _print_comparison(label_a: str, metrics_a: dict,
                       label_b: str, metrics_b: dict) -> None:
    metrics_order = [
        "total_return", "cagr", "volatility",
        "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio",
        "win_rate", "best_day", "worst_day",
    ]
    labels = {
        "total_return":  "Total return (%)",
        "cagr":          "CAGR (%)",
        "volatility":    "Volatility (%)",
        "sharpe_ratio":  "Sharpe ratio",
        "sortino_ratio": "Sortino ratio",
        "max_drawdown":  "Max drawdown (%)",
        "calmar_ratio":  "Calmar ratio",
        "win_rate":      "Win rate (%)",
        "best_day":      "Best day (%)",
        "worst_day":     "Worst day (%)",
    }
    higher_better = {"sharpe_ratio", "sortino_ratio", "cagr", "total_return",
                     "calmar_ratio", "win_rate", "best_day"}
    lower_better  = {"volatility", "max_drawdown", "worst_day"}

    col_w = max(len(label_a), len(label_b), 14)
    header = f"\n  {'Metric':<22} {label_a:>{col_w}} {label_b:>{col_w}}  {'Delta':>8}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for key in metrics_order:
        a = metrics_a.get(key, float("nan"))
        b = metrics_b.get(key, float("nan"))
        delta = b - a
        if key in higher_better:
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        elif key in lower_better:
            arrow = "↑" if delta < -0.005 else ("↓" if delta > 0.005 else "→")
        else:
            arrow = "→"
        sign = "+" if delta >= 0 else ""
        print(f"  {labels[key]:<22} {a:{col_w}.2f} {b:{col_w}.2f}  {sign}{delta:>6.3f} {arrow}")

    sharpe_delta = metrics_b.get("sharpe_ratio", 0) - metrics_a.get("sharpe_ratio", 0)
    sign = "+" if sharpe_delta >= 0 else ""
    print(sep)
    print(f"  {'Sharpe improvement':<22} {sign}{sharpe_delta:.3f}")
    print(sep)


# ---------------------------------------------------------------------------
# Single backtest runner
# ---------------------------------------------------------------------------

def _run_single(
    scorer: FactorScorer,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark: pd.Series,
    args: argparse.Namespace,
    label: str,
    sector_map: pd.Series | None = None,
    vol_target: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run one full backtest pass.  sector_map=None -> no neutralization."""
    logger.info("Running: %s …", label)

    # Wrap scorer to inject sector_map at each rebalance via monkey-patch
    # (cleanest approach without modifying Backtester)
    original_score = scorer.score

    def patched_score(**kwargs):
        return original_score(sector_map=sector_map, **kwargs)

    scorer.score = patched_score  # type: ignore[method-assign]

    bt = Backtester(
        scorer=scorer,
        prices=prices,
        fundamentals=fundamentals,
        rebalance_freq=args.rebalance,
        n_stocks=args.n_stocks,
        transaction_cost=args.txn_cost,
        benchmark_prices=benchmark,
        vol_target=vol_target,
        vol_lookback=args.vol_lookback,
        vol_max_leverage=args.vol_max_leverage,
    )
    results = bt.run()

    scorer.score = original_score  # restore

    metrics = compute_metrics(results["portfolio"].dropna())
    return results, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("=== Factor Investing Backtest ===")
    logger.info("Period         : %s -> %s", args.start, args.end)
    logger.info("Factors        : Value=%.0f%% | Momentum=%.0f%% | Size=%.0f%%",
                args.value_weight * 100, args.mom_weight * 100, args.size_weight * 100)
    logger.info("Sector neutral : %s", args.sector_neutral)
    if args.vol_target:
        logger.info("Vol target     : %.0f%%", args.vol_target * 100)

    # 1. Data
    tickers = get_sp500_tickers()
    prices  = fetch_price_data(tickers, start=args.start, end=args.end)

    logger.info("Fetching fundamental data …")
    fundamentals = fetch_fundamental_data(list(prices.columns))

    # Sector map (fast — uses built-in table for known tickers)
    sector_map = fetch_sector_map(list(prices.columns))

    # Benchmark
    spy_raw   = yf.download("SPY", start=args.start, end=args.end,
                             auto_adjust=True, progress=False)
    benchmark = spy_raw["Close"].squeeze()

    # 2. Scorer
    scorer = FactorScorer([
        (ValueFactor(pb_weight=0.5, pe_weight=0.5),         args.value_weight),
        (MomentumFactor(lookback_months=12, skip_months=1), args.mom_weight),
        (SizeFactor(use_log=True),                           args.size_weight),
    ])

    # 3a. Baseline (no sector neutral)
    results_base, metrics_base = _run_single(
        scorer, prices, fundamentals, benchmark, args,
        label="Baseline",
        sector_map=None,
        vol_target=args.vol_target,
    )

    # 3b. Sector-neutral run (only when --sector-neutral flag is set)
    if args.sector_neutral:
        results_sn, metrics_sn = _run_single(
            scorer, prices, fundamentals, benchmark, args,
            label="Sector-Neutral",
            sector_map=sector_map,
            vol_target=args.vol_target,
        )

        print("\n" + "=" * 60)
        print("  SECTOR NEUTRALIZATION — IMPACT ON SHARPE")
        print("=" * 60)
        _print_comparison("Baseline", metrics_base, "Sector-Neutral", metrics_sn)

        results_sn.to_csv(RESULTS_DIR / "daily_returns_sector_neutral.csv")
        logger.info("Sector-neutral returns saved -> results/daily_returns_sector_neutral.csv")

        primary_results = results_sn
        primary_metrics = metrics_sn
    else:
        primary_results = results_base
        primary_metrics = metrics_base

    # 4. Benchmark metrics
    bench_metrics = compute_metrics(results_base["benchmark"].dropna())

    logger.info("\n--- Baseline ---")
    for k, v in metrics_base.items():
        logger.info("  %-22s %s", k, v)
    logger.info("\n--- Benchmark (SPY) ---")
    for k, v in bench_metrics.items():
        logger.info("  %-22s %s", k, v)

    # 5. Save
    results_base.to_csv(RESULTS_DIR / "daily_returns.csv")
    pd.DataFrame({"portfolio": metrics_base, "benchmark": bench_metrics}).to_csv(
        RESULTS_DIR / "metrics.csv"
    )

    # 6. Charts
    if not args.no_plot:
        port_ret  = primary_results["portfolio"].dropna()
        bench_ret = primary_results["benchmark"].dropna()

        plot_performance_dashboard(
            portfolio_returns=port_ret,
            benchmark_returns=bench_ret,
            metrics=primary_metrics,
            save_path=RESULTS_DIR / "dashboard.png",
        )

        latest_scores = scorer.score(
            prices=prices,
            fundamentals=fundamentals,
            as_of=prices.index[-1],
            sector_map=sector_map if args.sector_neutral else None,
        )
        if not latest_scores.empty:
            plot_factor_scores(
                latest_scores, top_n=20,
                title="Factor Scores — Latest Rebalancing",
                save_path=RESULTS_DIR / "factor_scores.png",
            )

        logger.info("Charts saved -> results/")

    logger.info("\n=== Done. Results in ./%s/ ===", RESULTS_DIR)


if __name__ == "__main__":
    main()