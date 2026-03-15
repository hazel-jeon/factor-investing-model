"""
visualization/plots.py
----------------------
All matplotlib-based charts for the factor investing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Style defaults
# --------------------------------------------------------------------------
PORTFOLIO_COLOR = "#2563EB"   # blue
BENCHMARK_COLOR = "#6B7280"   # gray
DRAWDOWN_COLOR  = "#EF4444"   # red
FACTOR_COLORS   = {
    "value":    "#7C3AED",
    "momentum": "#059669",
    "size":     "#D97706",
    "composite":"#2563EB",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        120,
})


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Cumulative Returns",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot cumulative wealth curves for portfolio (and optional benchmark).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    port_wealth = (1 + portfolio_returns.dropna()).cumprod() * 100
    ax.plot(port_wealth, color=PORTFOLIO_COLOR, linewidth=2, label="Factor Portfolio")

    if benchmark_returns is not None:
        bench_wealth = (1 + benchmark_returns.dropna()).cumprod() * 100
        ax.plot(bench_wealth, color=BENCHMARK_COLOR, linewidth=1.5,
                linestyle="--", label="Benchmark", alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Growth of $100", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(framealpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_drawdown(
    portfolio_returns: pd.Series,
    title: str = "Drawdown",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the underwater equity curve (drawdown from rolling peak).
    """
    r = portfolio_returns.dropna()
    wealth = (1 + r).cumprod()
    drawdown = (wealth / wealth.cummax() - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(drawdown.index, drawdown, 0, color=DRAWDOWN_COLOR, alpha=0.4)
    ax.plot(drawdown, color=DRAWDOWN_COLOR, linewidth=1)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_factor_scores(
    scores: pd.DataFrame,
    top_n: int = 15,
    title: str = "Factor Scores — Top Holdings",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing per-factor and composite scores for top-N stocks.
    """
    factor_cols = [c for c in scores.columns if c != "composite"]
    top = scores.nlargest(top_n, "composite")

    fig, axes = plt.subplots(1, len(factor_cols) + 1,
                              figsize=(4 * (len(factor_cols) + 1), max(4, top_n * 0.4 + 1)),
                              sharey=True)

    all_cols = factor_cols + ["composite"]
    for ax, col in zip(axes, all_cols):
        color = FACTOR_COLORS.get(col, PORTFOLIO_COLOR)
        vals = top[col] if col in top.columns else pd.Series(dtype=float)
        ax.barh(top.index, vals, color=color, alpha=0.8)
        ax.set_title(col.title(), fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_annual_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Annual Returns",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart of calendar-year returns.
    """
    port_annual = portfolio_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1) * 100
    years = port_annual.index.year

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(years))
    width = 0.35

    if benchmark_returns is not None:
        bench_annual = benchmark_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1) * 100
        ax.bar(x - width / 2, port_annual.values, width, label="Factor Portfolio",
               color=PORTFOLIO_COLOR, alpha=0.85)
        ax.bar(x + width / 2, bench_annual.reindex(port_annual.index).values, width,
               label="Benchmark", color=BENCHMARK_COLOR, alpha=0.65)
    else:
        colors = [PORTFOLIO_COLOR if v >= 0 else DRAWDOWN_COLOR for v in port_annual.values]
        ax.bar(x, port_annual.values, width * 2, color=colors, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.legend(framealpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_rolling_sharpe(
    rolling_metrics: pd.DataFrame,
    title: str = "Rolling 1-Year Sharpe Ratio",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Line chart of rolling Sharpe ratio.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    rs = rolling_metrics["rolling_sharpe"].dropna()
    ax.plot(rs, color=PORTFOLIO_COLOR, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(1, color=BENCHMARK_COLOR, linewidth=0.7, linestyle="--", alpha=0.6, label="Sharpe = 1")
    ax.fill_between(rs.index, rs, 0, where=rs > 0, alpha=0.15, color=PORTFOLIO_COLOR)
    ax.fill_between(rs.index, rs, 0, where=rs < 0, alpha=0.15, color=DRAWDOWN_COLOR)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.legend(framealpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_performance_dashboard(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    metrics: Optional[dict] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    4-panel dashboard: cumulative returns, drawdown, annual returns, rolling Sharpe.
    """
    from ..portfolio.metrics import compute_rolling_metrics

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])    # cumulative — full width
    ax2 = fig.add_subplot(gs[1, :])    # drawdown   — full width
    ax3 = fig.add_subplot(gs[2, 0])    # annual returns
    ax4 = fig.add_subplot(gs[2, 1])    # rolling sharpe

    # 1. Cumulative
    port_wealth = (1 + portfolio_returns.dropna()).cumprod() * 100
    ax1.plot(port_wealth, color=PORTFOLIO_COLOR, linewidth=2, label="Factor Portfolio")
    if benchmark_returns is not None:
        bench_wealth = (1 + benchmark_returns.dropna()).cumprod() * 100
        ax1.plot(bench_wealth, color=BENCHMARK_COLOR, linewidth=1.5,
                 linestyle="--", label="Benchmark", alpha=0.8)
    ax1.set_title("Cumulative Returns (Growth of $100)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("$", fontsize=10)
    ax1.legend(framealpha=0.5, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 2. Drawdown
    r = portfolio_returns.dropna()
    wealth = (1 + r).cumprod()
    dd = (wealth / wealth.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd, 0, color=DRAWDOWN_COLOR, alpha=0.4)
    ax2.plot(dd, color=DRAWDOWN_COLOR, linewidth=1)
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("(%)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3. Annual returns
    port_ann = portfolio_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    yrs = np.arange(len(port_ann))
    colors_bar = [PORTFOLIO_COLOR if v >= 0 else DRAWDOWN_COLOR for v in port_ann.values]
    ax3.bar(yrs, port_ann.values, color=colors_bar, alpha=0.85)
    ax3.set_xticks(yrs)
    ax3.set_xticklabels(port_ann.index.year, rotation=45, ha="right", fontsize=8)
    ax3.axhline(0, color="black", linewidth=0.7)
    ax3.set_title("Annual Returns (%)", fontsize=12, fontweight="bold")

    # 4. Rolling Sharpe
    rm = compute_rolling_metrics(portfolio_returns)
    rs = rm["rolling_sharpe"].dropna()
    ax4.plot(rs, color=PORTFOLIO_COLOR, linewidth=1.5)
    ax4.axhline(0, color="black", linewidth=0.7)
    ax4.axhline(1, color=BENCHMARK_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    ax4.fill_between(rs.index, rs, 0, where=rs >= 0, alpha=0.15, color=PORTFOLIO_COLOR)
    ax4.fill_between(rs.index, rs, 0, where=rs < 0,  alpha=0.15, color=DRAWDOWN_COLOR)
    ax4.set_title("Rolling 1Y Sharpe Ratio", fontsize=12, fontweight="bold")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Metrics text box
    if metrics:
        txt = (
            f"CAGR: {metrics.get('cagr', 0):.1f}%   "
            f"Vol: {metrics.get('volatility', 0):.1f}%   "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}   "
            f"MDD: {metrics.get('max_drawdown', 0):.1f}%"
        )
        fig.text(0.5, 0.97, txt, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6", alpha=0.8))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
