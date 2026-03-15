"""
run_backtest.py
---------------
End-to-end pipeline:
  1. Fetch price + fundamental data
  2. Build factor scorer
  3. Run walk-forward backtest  (baseline + optional vol-targeted run)
  4. Print side-by-side comparison when --vol-target is set
  5. Generate charts → results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from factor_investing.data import fetch_price_data, fetch_fundamental_data, get_sp500_tickers
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
    p.add_argument("--start",        default="2015-01-01", help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end",          default="2024-12-31", help="Backtest end date   (YYYY-MM-DD)")
    p.add_argument("--n-stocks",     type=int,   default=20,    help="Portfolio size (default 20)")
    p.add_argument("--rebalance",    default="QS",              help="Rebalancing freq: QS/MS/AS")
    p.add_argument("--value-weight", type=float, default=0.4,   help="Value factor weight")
    p.add_argument("--mom-weight",   type=float, default=0.4,   help="Momentum factor weight")
    p.add_argument("--size-weight",  type=float, default=0.2,   help="Size factor weight")
    p.add_argument("--txn-cost",     type=float, default=0.001, help="One-way transaction cost")
    # ── Volatility targeting ──────────────────────────────────────────────
    p.add_argument(
        "--vol-target",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Annualised volatility target, e.g. 0.15 for 15%%. "
            "When set, the script runs BOTH a baseline and a vol-targeted backtest "
            "and prints a side-by-side comparison. "
            "Omit (default) to run the baseline only."
        ),
    )
    p.add_argument(
        "--vol-lookback",
        type=int,
        default=60,
        metavar="DAYS",
        help="Rolling window (trading days) for realised-vol estimation. Default 60.",
    )
    p.add_argument(
        "--vol-max-leverage",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="Maximum leverage allowed by vol scaling. Default 2.0.",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_comparison(baseline: dict, vol_targeted: dict) -> None:
    """Pretty-print a side-by-side metrics table."""
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
    improve = {"sharpe_ratio", "sortino_ratio", "cagr", "total_return",
               "calmar_ratio", "win_rate", "best_day"}  # higher = better
    worsen  = {"volatility", "max_drawdown", "worst_day"}               # lower = better

    header = f"\n{'Metric':<22} {'Baseline':>12} {'Vol Target':>12} {'Δ':>10}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for key in metrics_order:
        label = labels.get(key, key)
        b = baseline.get(key, float("nan"))
        v = vol_targeted.get(key, float("nan"))
        delta = v - b
        # Arrow: ↑ good, ↓ bad, → neutral
        if key in improve:
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        elif key in worsen:
            arrow = "↑" if delta < -0.01 else ("↓" if delta > 0.01 else "→")
        else:
            arrow = "→"
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {b:>12.2f} {v:>12.2f}   {sign}{delta:.2f} {arrow}")

    sharpe_delta = vol_targeted.get("sharpe_ratio", 0) - baseline.get("sharpe_ratio", 0)
    sign = "+" if sharpe_delta >= 0 else ""
    print(sep)
    print(f"  {'Sharpe improvement':<20} {sign}{sharpe_delta:.3f}")
    print(sep)


def _run_single(
    scorer: FactorScorer,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark: pd.Series,
    args: argparse.Namespace,
    vol_target: float | None,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """Run one backtest pass and return (results_df, metrics_dict)."""
    logger.info("Running %s …", label)
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
    metrics = compute_metrics(results["portfolio"].dropna())
    return results, metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("=== Factor Investing Backtest ===")
    logger.info("Period     : %s → %s", args.start, args.end)
    logger.info("Factors    : Value=%.0f%% | Momentum=%.0f%% | Size=%.0f%%",
                args.value_weight * 100, args.mom_weight * 100, args.size_weight * 100)
    if args.vol_target:
        logger.info("Vol target : %.0f%%  (lookback=%d d, max leverage=%.1fx)",
                    args.vol_target * 100, args.vol_lookback, args.vol_max_leverage)

    # 1. Universe & data
    tickers = get_sp500_tickers()
    logger.info("Universe   : %d tickers", len(tickers))
    prices = fetch_price_data(tickers, start=args.start, end=args.end)

    logger.info("Fetching fundamental data (this may take a minute) …")
    fundamentals = fetch_fundamental_data(list(prices.columns))

    # 2. Benchmark (SPY)
    logger.info("Fetching SPY benchmark …")
    spy_raw = yf.download("SPY", start=args.start, end=args.end,
                          auto_adjust=True, progress=False)
    benchmark = spy_raw["Close"].squeeze()

    # 3. Factor scorer (shared between both runs)
    scorer = FactorScorer([
        (ValueFactor(pb_weight=0.5, pe_weight=0.5),          args.value_weight),
        (MomentumFactor(lookback_months=12, skip_months=1),  args.mom_weight),
        (SizeFactor(use_log=True),                            args.size_weight),
    ])

    # 4. Baseline backtest (always)
    results_base, metrics_base = _run_single(
        scorer, prices, fundamentals, benchmark, args,
        vol_target=None, label="baseline (no vol targeting)",
    )

    # 5. Vol-targeted backtest (only when --vol-target is given)
    if args.vol_target is not None:
        results_vt, metrics_vt = _run_single(
            scorer, prices, fundamentals, benchmark, args,
            vol_target=args.vol_target, label=f"vol-targeted ({args.vol_target*100:.0f}%)",
        )
        _print_comparison(metrics_base, metrics_vt)

        # Save vol-targeted returns separately
        results_vt.to_csv(RESULTS_DIR / "daily_returns_vol_targeted.csv")
        logger.info("Vol-targeted returns saved → results/daily_returns_vol_targeted.csv")

        # Use the vol-targeted results for charts
        primary_results  = results_vt
        primary_metrics  = metrics_vt
        chart_label      = f"Vol-Targeted ({args.vol_target*100:.0f}%)"
    else:
        primary_results = results_base
        primary_metrics = metrics_base
        chart_label     = "Factor Portfolio"

    # ── Baseline metrics summary ──────────────────────────────────────────
    bench_metrics = compute_metrics(results_base["benchmark"].dropna())
    logger.info("\n─── Baseline ───")
    for k, v in metrics_base.items():
        logger.info("  %-20s %s", k, v)
    logger.info("\n─── Benchmark (SPY) ───")
    for k, v in bench_metrics.items():
        logger.info("  %-20s %s", k, v)

    # Save baseline
    results_base.to_csv(RESULTS_DIR / "daily_returns.csv")
    pd.DataFrame({"portfolio": metrics_base, "benchmark": bench_metrics}).to_csv(
        RESULTS_DIR / "metrics.csv"
    )
    logger.info("Baseline results saved → results/")

    # 6. Charts
    if not args.no_plot:
        logger.info("Generating charts …")
        port_ret  = primary_results["portfolio"].dropna()
        bench_ret = primary_results["benchmark"].dropna()

        plot_performance_dashboard(
            portfolio_returns=port_ret,
            benchmark_returns=bench_ret,
            metrics=primary_metrics,
            save_path=RESULTS_DIR / "dashboard.png",
        )
        logger.info("Dashboard saved → results/dashboard.png")

        latest_scores = scorer.score(
            prices=prices,
            fundamentals=fundamentals,
            as_of=prices.index[-1],
        )
        if not latest_scores.empty:
            plot_factor_scores(
                latest_scores, top_n=20,
                title="Factor Scores — Latest Rebalancing",
                save_path=RESULTS_DIR / "factor_scores.png",
            )
            logger.info("Factor scores saved → results/factor_scores.png")

    logger.info("\n=== Done. Results in ./%s/ ===", RESULTS_DIR)


if __name__ == "__main__":
    main()