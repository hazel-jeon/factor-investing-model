"""
run_backtest.py
---------------
End-to-end pipeline:
  1. Fetch price + fundamental + sector data
  2. Build factor scorer
  3. Run baseline backtest, then optionally a second run with a different optimizer
  4. Print side-by-side Sharpe comparison
  5. Save results + generate charts -> results/
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
from factor_investing.portfolio import (
    Backtester,
    compute_metrics,
    compute_rolling_metrics,
    OPTIMIZERS,
)
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
    # Optimizer
    p.add_argument(
        "--optimizer",
        default="equal",
        choices=list(OPTIMIZERS.keys()),
        help=(
            "Portfolio weighting scheme.  "
            "'equal' = 1/N (default).  "
            "'min_var' = minimum-variance.  "
            "When set to 'min_var', both equal and min_var runs are executed "
            "and a side-by-side comparison is printed."
        ),
    )
    p.add_argument("--cov-lookback",  type=int,   default=60,   metavar="DAYS",
                   help="Covariance lookback window for min_var (default 60).")
    p.add_argument("--weight-min",    type=float, default=0.01,
                   help="Per-stock weight lower bound for min_var (default 0.01).")
    p.add_argument("--weight-max",    type=float, default=0.15,
                   help="Per-stock weight upper bound for min_var (default 0.15).")
    # Vol targeting
    p.add_argument("--vol-target",       type=float, default=None,  metavar="FLOAT")
    p.add_argument("--vol-lookback",     type=int,   default=60,    metavar="DAYS")
    p.add_argument("--vol-max-leverage", type=float, default=2.0,   metavar="FLOAT")
    # Sector neutralization
    p.add_argument("--sector-neutral",  action="store_true")
    p.add_argument("--no-plot",         action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def _print_comparison(label_a: str, metrics_a: dict,
                       label_b: str, metrics_b: dict) -> None:
    order = [
        "total_return", "cagr", "volatility",
        "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio",
        "win_rate", "best_day", "worst_day",
    ]
    display = {
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

    col = max(len(label_a), len(label_b), 14)
    header = f"\n  {'Metric':<22} {label_a:>{col}} {label_b:>{col}}  {'Delta':>8}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for key in order:
        a = metrics_a.get(key, float("nan"))
        b = metrics_b.get(key, float("nan"))
        d = b - a
        if key in higher_better:
            arrow = "↑" if d > 0.005 else ("↓" if d < -0.005 else "→")
        elif key in lower_better:
            arrow = "↑" if d < -0.005 else ("↓" if d > 0.005 else "→")
        else:
            arrow = "→"
        sign = "+" if d >= 0 else ""
        print(f"  {display[key]:<22} {a:{col}.2f} {b:{col}.2f}  {sign}{d:>6.3f} {arrow}")

    sd = metrics_b.get("sharpe_ratio", 0) - metrics_a.get("sharpe_ratio", 0)
    sign = "+" if sd >= 0 else ""
    print(sep)
    print(f"  {'Sharpe improvement':<22} {sign}{sd:.3f}")
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
    optimizer: str = "equal",
    sector_map: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run one full backtest pass and return (results_df, metrics_dict)."""
    logger.info("Running: %s ...", label)

    original_score = scorer.score

    def patched_score(**kwargs):
        return original_score(sector_map=sector_map, **kwargs)

    if sector_map is not None:
        scorer.score = patched_score  # type: ignore[method-assign]

    bt = Backtester(
        scorer=scorer,
        prices=prices,
        fundamentals=fundamentals,
        rebalance_freq=args.rebalance,
        n_stocks=args.n_stocks,
        transaction_cost=args.txn_cost,
        benchmark_prices=benchmark,
        optimizer=optimizer,
        cov_lookback=args.cov_lookback,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        vol_target=args.vol_target,
        vol_lookback=args.vol_lookback,
        vol_max_leverage=args.vol_max_leverage,
    )
    results = bt.run()

    if sector_map is not None:
        scorer.score = original_score

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
    logger.info("Optimizer      : %s", args.optimizer)
    logger.info("Sector neutral : %s", args.sector_neutral)

    # 1. Data
    tickers = get_sp500_tickers()
    prices  = fetch_price_data(tickers, start=args.start, end=args.end)

    logger.info("Fetching fundamental data ...")
    fundamentals = fetch_fundamental_data(list(prices.columns))
    sector_map   = fetch_sector_map(list(prices.columns))

    spy_raw   = yf.download("SPY", start=args.start, end=args.end,
                             auto_adjust=True, progress=False)
    benchmark = spy_raw["Close"].squeeze()

    # 2. Scorer
    scorer = FactorScorer([
        (ValueFactor(pb_weight=0.5, pe_weight=0.5),         args.value_weight),
        (MomentumFactor(),                                    args.mom_weight),
        (SizeFactor(use_log=True),                           args.size_weight),
    ])

    smap = sector_map if args.sector_neutral else None

    # 3a. Baseline — always equal weight
    results_base, metrics_base = _run_single(
        scorer, prices, fundamentals, benchmark, args,
        label="Baseline (equal weight)",
        optimizer="equal",
        sector_map=smap,
    )

    # 3b. Optimized run — only when --optimizer min_var is requested
    primary_results = results_base
    primary_metrics = metrics_base

    if args.optimizer != "equal":
        results_opt, metrics_opt = _run_single(
            scorer, prices, fundamentals, benchmark, args,
            label=f"Optimized ({args.optimizer})",
            optimizer=args.optimizer,
            sector_map=smap,
        )

        print("\n" + "=" * 64)
        print(f"  PORTFOLIO OPTIMIZATION: equal  vs  {args.optimizer.upper()}")
        print("=" * 64)
        _print_comparison(
            "Equal weight", metrics_base,
            args.optimizer.upper(), metrics_opt,
        )

        results_opt.to_csv(RESULTS_DIR / f"daily_returns_{args.optimizer}.csv")
        logger.info("Optimized returns saved -> results/daily_returns_%s.csv", args.optimizer)

        primary_results = results_opt
        primary_metrics = metrics_opt

    # 4. Benchmark metrics + logging
    bench_metrics = compute_metrics(results_base["benchmark"].dropna())

    logger.info("\n--- Baseline (equal weight) ---")
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
            sector_map=smap,
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