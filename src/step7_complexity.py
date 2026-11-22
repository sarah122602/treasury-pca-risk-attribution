"""Step 7: model-complexity, stability, and risk-coverage analysis.

This module consumes the already validated Step 6 walk-forward outputs.  It
does not refit a model on test observations or change the Step 6 selection
rules; it adds decision-oriented summaries and visual diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.factor_selection import STRATEGIES
from src.portfolio_attribution import PORTFOLIOS


PORTFOLIO_ORDER = list(PORTFOLIOS.columns)
STRATEGY_ORDER = list(STRATEGIES)
BASELINE = "fixed_3"


def _load_step6(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Step 6 outputs, which are the fixed evaluation sample for Step 7."""
    names = {
        "daily": "step6_daily_strategy_metrics.csv",
        "selections": "step6_training_factor_selection.csv",
        "summary": "step6_strategy_summary.csv",
        "regime": "step6_regime_strategy_summary.csv",
        "bootstrap": "step6_paired_bootstrap_ci.csv",
        "verification": "step6_verification.csv",
        "block_verification": "step6_block_verification.csv",
    }
    missing = [name for name in names.values() if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Run scripts/06_portfolio_factor_selection.py first; missing " + ", ".join(missing))
    return {key: pd.read_csv(results_dir / filename) for key, filename in names.items()}


def _complexity_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Add marginal accuracy and accuracy-per-added-factor fields."""
    rows = []
    for portfolio in PORTFOLIO_ORDER:
        base = summary[(summary.portfolio == portfolio) & (summary.strategy == BASELINE)].iloc[0]
        for _, row in summary[summary.portfolio == portfolio].set_index("strategy").loc[STRATEGY_ORDER].reset_index().iterrows():
            added = float(row.average_selected_factor_count - base.average_selected_factor_count)
            explained_delta = float(row.explained_pnl_variance_pct - base.explained_pnl_variance_pct)
            rmse_reduction = float(base.rmse_usd - row.rmse_usd)
            rows.append(
                {
                    **row.to_dict(),
                    "additional_factors_vs_fixed_3": added,
                    "explained_variance_improvement_pp_vs_fixed_3": explained_delta,
                    "rmse_reduction_usd_vs_fixed_3": rmse_reduction,
                    "normalized_rmse_change_vs_fixed_3": float(row.normalized_rmse - base.normalized_rmse),
                    "explained_pp_per_additional_factor": (
                        explained_delta / added if added > 1e-12 else np.nan
                    ),
                    "rmse_reduction_usd_per_additional_factor": (
                        rmse_reduction / added if added > 1e-12 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _stability_summary(selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize count distributions, changes, and ex-ante regimes."""
    rows = []
    distribution = (
        selections.groupby(["portfolio", "strategy", "selected_factor_count"], as_index=False)
        .size()
        .rename(columns={"size": "test_blocks"})
    )
    totals = selections.groupby(["portfolio", "strategy"])["test_block"].nunique().rename("total_test_blocks")
    distribution = distribution.merge(totals, on=["portfolio", "strategy"])
    distribution["share_of_test_blocks_pct"] = 100 * distribution.test_blocks / distribution.total_test_blocks
    distribution = distribution.drop(columns="total_test_blocks")

    for (portfolio, strategy), frame in selections.groupby(["portfolio", "strategy"], sort=False):
        ordered = frame.sort_values("test_block").reset_index(drop=True)
        counts = ordered.selected_factor_count.to_numpy(dtype=float)
        changes = np.diff(counts)
        rows.append(
            {
                "portfolio": portfolio,
                "strategy": strategy,
                "test_blocks": len(counts),
                "average_selected_factor_count": counts.mean(),
                "median_selected_factor_count": np.median(counts),
                "minimum_selected_factor_count": counts.min(),
                "maximum_selected_factor_count": counts.max(),
                "distinct_factor_counts": ",".join(str(int(x)) for x in sorted(set(counts))),
                "count_change_events": int(np.sum(changes != 0)),
                "count_change_frequency_pct": 100 * float(np.mean(changes != 0)),
                "average_absolute_count_change": float(np.mean(np.abs(changes))),
                "regime_high_average_factor_count": ordered.loc[ordered.regime == "high_volatility", "selected_factor_count"].mean(),
                "regime_low_average_factor_count": ordered.loc[ordered.regime == "low_volatility", "selected_factor_count"].mean(),
                "high_minus_low_average_factor_count": (
                    ordered.loc[ordered.regime == "high_volatility", "selected_factor_count"].mean()
                    - ordered.loc[ordered.regime == "low_volatility", "selected_factor_count"].mean()
                ),
            }
        )
    return pd.DataFrame(rows), distribution


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    """Mark strategies not dominated on average complexity and OOS normalized RMSE."""
    rows = []
    for portfolio, frame in summary.groupby("portfolio", sort=False):
        for _, row in frame.iterrows():
            competitors = frame[frame.strategy != row.strategy]
            dominates = competitors[
                (competitors.average_selected_factor_count <= row.average_selected_factor_count)
                & (competitors.normalized_rmse <= row.normalized_rmse)
                & (
                    (competitors.average_selected_factor_count < row.average_selected_factor_count)
                    | (competitors.normalized_rmse < row.normalized_rmse)
                )
            ]
            rows.append(
                {
                    "portfolio": portfolio,
                    "strategy": row.strategy,
                    "pareto_efficient": dominates.empty,
                    "dominated_by_count": int(len(dominates)),
                    "complexity_metric": "average selected factors",
                    "accuracy_metric": "out-of-sample normalized RMSE",
                    "ranking_scope": "retrospective research sample; not prospective model selection",
                }
            )
    return pd.DataFrame(rows)


def _save_figures(summary: pd.DataFrame, stability: pd.DataFrame, selections: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {"fixed_3": "#1f77b4", "fixed_5": "#17becf", "yield_95": "#7f7f7f", "yield_99": "#9467bd", "portfolio_95": "#ff7f0e", "portfolio_99": "#d62728", "all_7": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    for ax, portfolio in zip(axes.flat, PORTFOLIO_ORDER):
        frame = summary[summary.portfolio == portfolio]
        for _, row in frame.iterrows():
            ax.scatter(row.average_selected_factor_count, row.explained_pnl_variance_pct, s=75, color=colors[row.strategy])
            ax.annotate(row.strategy, (row.average_selected_factor_count, row.explained_pnl_variance_pct), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(portfolio)
        ax.set_xlabel("Average retained factors")
        ax.set_ylabel("OOS explained P&L variance (%)")
        ax.grid(alpha=0.25)
    fig.suptitle("Step 7: Accuracy versus Model Complexity", y=1.01)
    fig.tight_layout()
    fig.savefig(figures_dir / "step7_accuracy_vs_complexity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    marginal = summary[summary.strategy != BASELINE]
    for strategy in STRATEGY_ORDER:
        if strategy == BASELINE:
            continue
        frame = marginal[marginal.strategy == strategy]
        ax.plot(frame.portfolio, frame.explained_pp_per_additional_factor, marker="o", label=strategy)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("OOS explained-variance improvement per added factor (pp)")
    ax.set_title("Marginal OOS Accuracy per Additional Factor vs Fixed PC1–PC3")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "step7_marginal_improvement_per_factor.png", dpi=160)
    plt.close(fig)

    focus = selections[selections.strategy.isin(["portfolio_95", "portfolio_99"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, portfolio in zip(axes.flat, PORTFOLIO_ORDER):
        frame = focus[focus.portfolio == portfolio].sort_values("test_block")
        for strategy, group in frame.groupby("strategy"):
            ax.plot(group.test_block, group.selected_factor_count, marker=".", linewidth=0.8, label=strategy)
        ax.set_title(portfolio)
        ax.set_xlabel("Test block")
        ax.set_ylabel("Selected factors")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.5, 7.5)
    axes[0, 0].legend()
    fig.suptitle("Portfolio-Aware Factor Counts Through Time", y=1.01)
    fig.tight_layout()
    fig.savefig(figures_dir / "step7_factor_count_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    for strategy in STRATEGY_ORDER:
        frame = summary[summary.strategy == strategy]
        ax.scatter(frame.average_selected_factor_count, frame.explained_pnl_variance_pct, s=65, label=strategy)
    for _, row in summary.iterrows():
        ax.annotate(row.portfolio.replace("_", " "), (row.average_selected_factor_count, row.explained_pnl_variance_pct), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Average retained factors")
    ax.set_ylabel("OOS P&L variance explained (%)")
    ax.set_title("Portfolio-Level Complexity versus Historical Risk Coverage")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "step7_complexity_risk_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_step7(results_dir: Path, figures_dir: Path) -> dict[str, pd.DataFrame]:
    """Create all Step 7 tables and figures from the fixed Step 6 sample."""
    data = _load_step6(results_dir)
    summary = _complexity_summary(data["summary"])
    stability, distribution = _stability_summary(data["selections"])
    pareto = _pareto(summary)
    regime = data["regime"].copy()
    regime["risk_coverage_metric"] = "OOS explained P&L variance (%)"

    daily = data["daily"]
    reconciliation = pd.DataFrame(
        [
            {"check": "same_test_dates_across_strategies", "passed": daily.groupby("strategy").Date.nunique().nunique() == 1, "value": daily.Date.nunique(), "detail": "identical out-of-sample dates"},
            {"check": "pnl_prediction_plus_error_reconciles", "passed": bool(np.max(np.abs(daily.total_pnl_usd - daily.prediction_usd - daily.reconstruction_error_usd)) < 1e-6), "value": float(np.max(np.abs(daily.total_pnl_usd - daily.prediction_usd - daily.reconstruction_error_usd))), "detail": "actual P&L = reconstructed P&L + residual"},
            {"check": "all_step6_checks_passed", "passed": bool(data["verification"].passed.all()), "value": int(data["verification"].passed.sum()), "detail": "inherited Step 6 training-only and block checks"},
            {"check": "selection_counts_in_range", "passed": bool(data["selections"].selected_factor_count.between(1, 7).all()), "value": int(data["selections"].selected_factor_count.between(1, 7).sum()), "detail": "consecutive PC1-to-PCk selections"},
        ]
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "step7_strategy_comparison.csv", index=False)
    stability.to_csv(results_dir / "step7_factor_count_stability.csv", index=False)
    distribution.to_csv(results_dir / "step7_factor_count_distribution.csv", index=False)
    regime.to_csv(results_dir / "step7_regime_strategy_summary.csv", index=False)
    pareto.to_csv(results_dir / "step7_pareto_strategies.csv", index=False)
    data["bootstrap"].to_csv(results_dir / "step7_paired_bootstrap_ci.csv", index=False)
    reconciliation.to_csv(results_dir / "step7_verification.csv", index=False)
    _save_figures(summary, stability, data["selections"], figures_dir)
    assert reconciliation["passed"].all()
    return {"summary": summary, "stability": stability, "distribution": distribution, "regime": regime, "pareto": pareto, "bootstrap": data["bootstrap"], "verification": reconciliation}
