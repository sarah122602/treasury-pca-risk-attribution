"""Create polished GitHub showcase figures from verified Step 7 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PORTFOLIOS = [
    "Bullet_10Y",
    "Barbell_2Y_30Y",
    "Steepener_2Y10Y",
    "Butterfly_2Y5Y10Y",
]
PORTFOLIO_LABELS = {
    "Bullet_10Y": "10Y bullet",
    "Barbell_2Y_30Y": "2Y/30Y barbell",
    "Steepener_2Y10Y": "2Y/10Y steepener",
    "Butterfly_2Y5Y10Y": "2Y/5Y/10Y butterfly",
}
STRATEGIES = ["fixed_3", "fixed_5", "yield_95", "yield_99", "portfolio_95", "portfolio_99", "all_7"]
STRATEGY_LABELS = {
    "fixed_3": "Fixed PC1–PC3",
    "fixed_5": "Fixed PC1–PC5",
    "yield_95": "Yield variance ≥95%",
    "yield_99": "Yield variance ≥99%",
    "portfolio_95": "Portfolio P&L ≥95%",
    "portfolio_99": "Portfolio P&L ≥99%",
    "all_7": "All seven factors",
}
COLORS = {
    "fixed_3": "#17365D",
    "fixed_5": "#2F75B5",
    "yield_95": "#8FA6BF",
    "yield_99": "#4F81BD",
    "portfolio_95": "#1B998B",
    "portfolio_99": "#D95F59",
    "all_7": "#7A7A7A",
}
NAVY = "#17365D"
TEAL = "#1B998B"
BUTTERFLY = "#D95F59"
TEXT = "#203040"
MUTED = "#607080"
GRID = "#D9E1E8"


def _load_results() -> pd.DataFrame:
    """Load only the verified Step 7 comparison table."""
    path = RESULTS / "step7_strategy_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing verified Step 7 results: {path}")
    frame = pd.read_csv(path)
    required = {
        "portfolio",
        "strategy",
        "explained_pnl_variance_pct",
        "average_selected_factor_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Step 7 table is missing required columns: {sorted(missing)}")
    if set(frame.portfolio) != set(PORTFOLIOS) or set(frame.strategy) != set(STRATEGIES):
        raise ValueError("Step 7 table does not contain the expected portfolios and strategies")
    return frame


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _figure_1(frame: pd.DataFrame) -> None:
    """Compare fixed PC1–PC3 portfolio P&L coverage."""
    values = (
        frame[frame.strategy == "fixed_3"]
        .set_index("portfolio")
        .loc[PORTFOLIOS, "explained_pnl_variance_pct"]
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(PORTFOLIOS))
    colors = [BUTTERFLY if portfolio == "Butterfly_2Y5Y10Y" else NAVY for portfolio in PORTFOLIOS]
    bars = ax.barh(y, values.to_numpy(), color=colors, height=0.58)
    ax.set_yticks(y, [PORTFOLIO_LABELS[portfolio] for portfolio in PORTFOLIOS])
    ax.set_xlim(0, 106)
    ax.set_xlabel("Out-of-sample portfolio P&L variance explained (%)", labelpad=10)
    ax.set_title("Portfolio-level PCA coverage gap", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    fig.text(0.125, 0.925, "Fixed PC1–PC3 baseline across identical 21-day test blocks", fontsize=10.5, color=MUTED)
    for bar, value, portfolio in zip(bars, values.to_numpy(), PORTFOLIOS):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", ha="left", fontsize=11, fontweight="bold", color=BUTTERFLY if portfolio == "Butterfly_2Y5Y10Y" else NAVY)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.55)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.94, top=0.82, bottom=0.17)
    _save_figure(fig, "showcase_portfolio_coverage_gap.png")


def _figure_2(frame: pd.DataFrame) -> None:
    """Show butterfly coverage gains against average retained factor count."""
    strategies = ["fixed_3", "fixed_5", "portfolio_95", "portfolio_99"]
    butterfly = frame[(frame.portfolio == "Butterfly_2Y5Y10Y") & frame.strategy.isin(strategies)].set_index("strategy").loc[strategies]
    x = np.arange(len(strategies))
    values = butterfly.explained_pnl_variance_pct.to_numpy()
    factors = butterfly.average_selected_factor_count.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    bars = ax.bar(x, values, color=[COLORS[strategy] for strategy in strategies], width=0.62)
    ax.set_ylim(0, 108)
    ax.set_xticks(x, [f"{STRATEGY_LABELS[strategy]}\n{factor:.2f} avg factors" for strategy, factor in zip(strategies, factors)])
    ax.set_ylabel("Out-of-sample butterfly P&L variance explained (%)", labelpad=10)
    ax.set_title("Butterfly risk coverage improves—but complexity rises", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    fig.text(0.125, 0.925, "Each bar uses the verified Step 7 walk-forward evaluation; labels show average retained factors", fontsize=10.5, color=MUTED)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.7, f"{value:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=TEXT)
    incremental = values[-1] - values[-2]
    factor_increment = factors[-1] - factors[-2]
    ax.annotate(
        f"+{incremental:.2f} pp for\n+{factor_increment:.2f} factors",
        xy=(3, values[-1]),
        xytext=(2.55, 86),
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1.0},
        fontsize=9.5,
        color=MUTED,
        ha="center",
    )
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.82, bottom=0.23)
    _save_figure(fig, "showcase_butterfly_coverage_improvement.png")


def _figure_3(frame: pd.DataFrame) -> None:
    """Show the complexity/coverage frontier in portfolio-specific panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, portfolio in zip(axes, PORTFOLIOS):
        subset = frame[frame.portfolio == portfolio].set_index("strategy").loc[STRATEGIES]
        for strategy in STRATEGIES:
            row = subset.loc[strategy]
            ax.scatter(
                row.average_selected_factor_count,
                row.explained_pnl_variance_pct,
                s=82 if strategy in {"fixed_3", "fixed_5", "portfolio_95", "portfolio_99"} else 48,
                color=COLORS[strategy],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        ax.axhline(95, color=GRID, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_title(PORTFOLIO_LABELS[portfolio], loc="left", fontsize=12.5, fontweight="bold", color=BUTTERFLY if portfolio == "Butterfly_2Y5Y10Y" else TEXT, pad=8)
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0, 103)
        ax.set_xticks([1, 3, 5, 7])
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        if portfolio in {"Steepener_2Y10Y", "Butterfly_2Y5Y10Y"}:
            ax.set_xlabel("Average retained factors")
        if portfolio in {"Bullet_10Y", "Butterfly_2Y5Y10Y"}:
            ax.set_ylabel("OOS P&L variance explained (%)")
        if portfolio == "Butterfly_2Y5Y10Y":
            for strategy, offset in [("fixed_3", (-24, -14)), ("fixed_5", (-22, 8)), ("portfolio_95", (-70, 15))]:
                row = subset.loc[strategy]
                ax.annotate(
                    STRATEGY_LABELS[strategy].replace("Portfolio P&L ", "P&L "),
                    (row.average_selected_factor_count, row.explained_pnl_variance_pct),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8.5,
                    color=COLORS[strategy],
                )
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[strategy], markeredgecolor="white", markersize=8, label=STRATEGY_LABELS[strategy]) for strategy in STRATEGIES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Model complexity versus historical portfolio risk coverage", fontsize=18, fontweight="bold", color=TEXT, x=0.08, ha="left", y=0.99)
    fig.text(0.08, 0.955, "Each panel uses the same verified out-of-sample evaluation; dashed line marks 95% coverage", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.16, wspace=0.18, hspace=0.26)
    _save_figure(fig, "showcase_complexity_vs_coverage.png")


def main() -> None:
    """Create the three GitHub-ready showcase charts."""
    frame = _load_results()
    _figure_1(frame)
    _figure_2(frame)
    _figure_3(frame)
    print("Created showcase_portfolio_coverage_gap.png")
    print("Created showcase_butterfly_coverage_improvement.png")
    print("Created showcase_complexity_vs_coverage.png")


if __name__ == "__main__":
    main()
