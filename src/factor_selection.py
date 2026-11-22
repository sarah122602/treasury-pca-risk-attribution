"""Step 6 portfolio-specific PCA factor selection and validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pca_model import TENORS, fit_pca, load_daily_changes
from src.portfolio_attribution import PORTFOLIOS
from src.regime_stability import _training_regime
from src.walk_forward_attribution import TEST_BLOCK, TRAINING_WINDOW


STRATEGIES = ("fixed_3", "fixed_5", "yield_95", "yield_99", "portfolio_95", "portfolio_99", "all_7")
THRESHOLDS = (0.95, 0.99)
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261122


def _minimum_factors(explained: np.ndarray, threshold: float) -> int:
    """Return the smallest consecutive PC1-to-PCk set meeting a threshold."""
    return int(np.searchsorted(np.cumsum(explained), threshold, side="left") + 1)


def _training_factor_counts(train_changes: pd.DataFrame, pca: object) -> tuple[dict[str, int], dict[str, float]]:
    """Select factor counts from training yield and portfolio P&L variance only."""
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    centered_scores = pca.transform(train_changes.to_numpy(dtype=float))
    yield_counts = {
        "yield_95": _minimum_factors(explained, 0.95),
        "yield_99": _minimum_factors(explained, 0.99),
    }
    counts: dict[str, int] = {"fixed_3": 3, **yield_counts, "all_7": len(TENORS)}
    training_shares: dict[str, float] = {
        "yield_95": float(np.cumsum(explained)[counts["yield_95"] - 1]),
        "yield_99": float(np.cumsum(explained)[counts["yield_99"] - 1]),
    }
    for portfolio, dv01_series in PORTFOLIOS.items():
        dv01 = dv01_series.to_numpy(dtype=float)
        total_pnl = -(train_changes.to_numpy(dtype=float) @ dv01)
        factor_pnl = np.column_stack(
            [-(centered_scores[:, factor, None] * pca.components_[factor] @ dv01) for factor in range(len(TENORS))]
        )
        component_variances = np.var(factor_pnl, axis=0, ddof=1)
        pnl_shares = np.cumsum(component_variances) / np.var(total_pnl, ddof=1)
        for threshold in THRESHOLDS:
            strategy = f"portfolio_{int(threshold * 100)}"
            count = _minimum_factors(component_variances / np.var(total_pnl, ddof=1), threshold)
            counts[f"{strategy}__{portfolio}"] = count
            training_shares[f"{strategy}__{portfolio}"] = float(pnl_shares[count - 1])
    return counts, training_shares


def _metric(actual: np.ndarray, error: np.ndarray, benchmark: np.ndarray) -> dict[str, float]:
    """Calculate the requested out-of-sample attribution metrics."""
    actual_variance = float(np.var(actual, ddof=1))
    sse = float(np.sum(error**2))
    benchmark_sse = float(np.sum((actual - benchmark) ** 2))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "explained_pnl_variance_pct": 100 * (1 - np.var(error, ddof=1) / actual_variance),
        "rmse_usd": rmse,
        "normalized_rmse": rmse / float(np.std(actual, ddof=1)),
        "oos_r2_vs_training_mean": 1 - sse / benchmark_sse if benchmark_sse > 0 else np.nan,
        "mean_residual_usd": float(np.mean(error)),
        "residual_variance_usd_squared": float(np.var(error, ddof=1)),
        "actual_pnl_std_usd": float(np.std(actual, ddof=1)),
    }


def _evaluate_blocks(changes: pd.DataFrame, dates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit each training window, select factors, and evaluate identical test blocks."""
    x = changes.to_numpy(dtype=float)
    test_starts = list(range(TRAINING_WINDOW, len(changes) - TEST_BLOCK + 1, TEST_BLOCK))
    daily_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, float | int | str]] = []
    verification_rows: list[dict[str, float | int | str]] = []

    for block_number, test_start in enumerate(test_starts, start=1):
        train_start = test_start - TRAINING_WINDOW
        test_end = test_start + TEST_BLOCK
        train_changes = changes.iloc[train_start:test_start].reset_index(drop=True)
        train_dates = dates.iloc[train_start:test_start].reset_index(drop=True)
        test_dates = dates.iloc[test_start:test_end].reset_index(drop=True)
        test_x = x[test_start:test_end]
        fitted = fit_pca(train_changes, train_dates, 0, TRAINING_WINDOW)
        pca = fitted.pca
        counts, training_shares = _training_factor_counts(train_changes, pca)
        regime, recent_vol, regime_threshold, _ = _training_regime(train_changes)
        scores = (test_x - pca.mean_) @ pca.components_.T
        component_changes = [scores[:, factor, None] * pca.components_[factor] for factor in range(len(TENORS))]
        block_id = f"block_{block_number:03d}"

        for portfolio, dv01_series in PORTFOLIOS.items():
            dv01 = dv01_series.to_numpy(dtype=float)
            actual = -(test_x @ dv01)
            training_mean_pnl = float(-(pca.mean_ @ dv01))
            strategy_counts = {
                "fixed_3": counts["fixed_3"],
                "fixed_5": 5,
                "yield_95": counts["yield_95"],
                "yield_99": counts["yield_99"],
                "portfolio_95": counts[f"portfolio_95__{portfolio}"],
                "portfolio_99": counts[f"portfolio_99__{portfolio}"],
                "all_7": counts["all_7"],
            }
            for strategy, factor_count in strategy_counts.items():
                prediction = -((pca.mean_ + sum(component_changes[:factor_count])) @ dv01)
                error = actual - prediction
                frame = pd.DataFrame(
                    {
                        "Date": test_dates.dt.strftime("%Y-%m-%d"),
                        "test_block": block_id,
                        "portfolio": portfolio,
                        "strategy": strategy,
                        "regime": regime,
                        "total_pnl_usd": actual,
                        "training_mean_pnl_usd": training_mean_pnl,
                        "selected_factor_count": factor_count,
                        "prediction_usd": prediction,
                        "reconstruction_error_usd": error,
                    }
                )
                daily_rows.append(frame)
                selection_rows.append(
                    {
                        "test_block": block_id,
                        "portfolio": portfolio,
                        "strategy": strategy,
                        "selected_factor_count": factor_count,
                        "training_explained_share": training_shares.get(
                            f"{strategy}__{portfolio}",
                            training_shares.get(strategy, np.nan),
                        ),
                        "regime": regime,
                        "recent_21d_curve_volatility_bp": recent_vol,
                        "training_volatility_threshold_bp": regime_threshold,
                    }
                )

        verification_rows.append(
            {
                "test_block": block_id,
                "train_start_index": train_start,
                "train_end_index_exclusive": test_start,
                "test_start_index": test_start,
                "test_end_index_exclusive": test_end,
                "train_end_equals_test_start": bool(test_start == train_start + TRAINING_WINDOW),
                "test_block_length": test_end - test_start,
                "max_pca7_reconstruction_error_bp": float(
                    np.max(np.abs(test_x - (pca.mean_ + sum(component_changes))))
                ),
                "max_pca_orthogonality_error": float(
                    np.abs(pca.components_ @ pca.components_.T - np.eye(len(TENORS))).max()
                ),
            }
        )
    return pd.concat(daily_rows, ignore_index=True), pd.DataFrame(selection_rows), pd.DataFrame(verification_rows)


def _summarize(daily: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize metrics and selected-factor distributions."""
    metric_rows: list[dict[str, float | int | str]] = []
    for (portfolio, strategy), frame in daily.groupby(["portfolio", "strategy"], sort=False):
        metrics = _metric(
            frame["total_pnl_usd"].to_numpy(),
            frame["reconstruction_error_usd"].to_numpy(),
            frame["training_mean_pnl_usd"].to_numpy(),
        )
        selected = frame["selected_factor_count"].drop_duplicates().to_numpy()
        blocks = selections[(selections["portfolio"] == portfolio) & (selections["strategy"] == strategy)]
        metrics.update(
            {
                "portfolio": portfolio,
                "strategy": strategy,
                "test_observations": len(frame),
                "test_blocks": len(blocks),
                "average_selected_factor_count": float(blocks["selected_factor_count"].mean()),
                "median_selected_factor_count": float(blocks["selected_factor_count"].median()),
                "minimum_selected_factor_count": int(blocks["selected_factor_count"].min()),
                "maximum_selected_factor_count": int(blocks["selected_factor_count"].max()),
                "selection_stability_pct": 100 * float(
                    (blocks["selected_factor_count"].to_numpy()[1:] == blocks["selected_factor_count"].to_numpy()[:-1]).mean()
                ),
                "distinct_selected_factor_counts": ",".join(str(value) for value in sorted(blocks["selected_factor_count"].unique())),
            }
        )
        metric_rows.append(metrics)

    distribution_rows = []
    for (portfolio, strategy, factor_count), frame in selections.groupby(["portfolio", "strategy", "selected_factor_count"], sort=False):
        distribution_rows.append(
            {
                "portfolio": portfolio,
                "strategy": strategy,
                "selected_factor_count": int(factor_count),
                "test_blocks": len(frame),
                "share_of_test_blocks_pct": 100 * len(frame) / selections[(selections["portfolio"] == portfolio) & (selections["strategy"] == strategy)]["test_block"].nunique(),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(distribution_rows)


def _regime_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate strategy performance separately for ex-ante volatility regimes."""
    rows = []
    for (portfolio, strategy, regime), frame in daily.groupby(["portfolio", "strategy", "regime"], sort=False):
        rows.append(
            {
                "portfolio": portfolio,
                "strategy": strategy,
                "regime": regime,
                "test_observations": len(frame),
                **_metric(frame["total_pnl_usd"].to_numpy(), frame["reconstruction_error_usd"].to_numpy(), frame["training_mean_pnl_usd"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_paired(daily: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap paired strategies and portfolios by complete 21-day blocks."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    block_ids = daily["test_block"].drop_duplicates().to_numpy()
    portfolios = list(PORTFOLIOS.columns)
    strategies = list(STRATEGIES)
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    observed_differences: dict[tuple[str, str, str], float] = {}
    for portfolio in portfolios:
        for strategy in strategies:
            frame = daily[(daily["portfolio"] == portfolio) & (daily["strategy"] == strategy)].sort_values(["test_block", "Date"])
            arrays[(portfolio, strategy)] = tuple(
                frame[column].to_numpy().reshape(len(block_ids), TEST_BLOCK)
                for column in ("total_pnl_usd", "reconstruction_error_usd", "training_mean_pnl_usd")
            )
        fixed_actual, fixed_error, fixed_benchmark = arrays[(portfolio, "fixed_3")]
        fixed_metric = _metric(fixed_actual.reshape(-1), fixed_error.reshape(-1), fixed_benchmark.reshape(-1))
        for strategy in (strategy for strategy in strategies if strategy != "fixed_3"):
            actual_blocks, error_blocks, benchmark_blocks = arrays[(portfolio, strategy)]
            strategy_metric = _metric(actual_blocks.reshape(-1), error_blocks.reshape(-1), benchmark_blocks.reshape(-1))
            observed_differences[(portfolio, strategy, "rmse_difference_vs_fixed_3")] = strategy_metric["rmse_usd"] - fixed_metric["rmse_usd"]
            observed_differences[(portfolio, strategy, "explained_variance_difference_vs_fixed_3")] = strategy_metric["explained_pnl_variance_pct"] - fixed_metric["explained_pnl_variance_pct"]
    draws: dict[tuple[str, str, str], list[float]] = {}
    comparisons = [strategy for strategy in strategies if strategy != "fixed_3"]
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, len(block_ids), size=len(block_ids))
        for portfolio in portfolios:
            metrics_by_strategy = {}
            for strategy in strategies:
                actual_blocks, error_blocks, benchmark_blocks = arrays[(portfolio, strategy)]
                actual = actual_blocks[sampled].reshape(-1)
                error = error_blocks[sampled].reshape(-1)
                benchmark = benchmark_blocks[sampled].reshape(-1)
                metrics_by_strategy[strategy] = _metric(actual, error, benchmark)
            for strategy in comparisons:
                draws.setdefault((portfolio, strategy, "rmse_difference_vs_fixed_3"), []).append(
                    metrics_by_strategy[strategy]["rmse_usd"] - metrics_by_strategy["fixed_3"]["rmse_usd"]
                )
                draws.setdefault((portfolio, strategy, "explained_variance_difference_vs_fixed_3"), []).append(
                    metrics_by_strategy[strategy]["explained_pnl_variance_pct"] - metrics_by_strategy["fixed_3"]["explained_pnl_variance_pct"]
                )

    rows = []
    for (portfolio, strategy, metric), values in draws.items():
        values_array = np.asarray(values)
        rows.append(
            {
                "portfolio": portfolio,
                "strategy": strategy,
                "metric": metric,
                "observed_difference": observed_differences[(portfolio, strategy, metric)],
                "bootstrap_lower_95": float(np.quantile(values_array, 0.025)),
                "bootstrap_upper_95": float(np.quantile(values_array, 0.975)),
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "random_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "paired non-overlapping 21-day test block",
            }
        )
    return pd.DataFrame(rows)


def _save_figures(summary: pd.DataFrame, distribution: pd.DataFrame, regime: pd.DataFrame, figures_dir: Path) -> None:
    """Save Step 6 factor-selection charts."""
    pivot = summary.pivot(index="portfolio", columns="strategy", values="explained_pnl_variance_pct")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("OOS P&L Variance Explained by Factor-Selection Strategy")
    axes[0].set_ylabel("Explained P&L variance (%)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    count_pivot = summary.pivot(index="portfolio", columns="strategy", values="average_selected_factor_count")
    count_pivot.plot(kind="bar", ax=axes[1], color="#70AD47")
    axes[1].set_title("Average Selected Factor Count")
    axes[1].set_ylabel("Factors")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step6_strategy_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in ("yield_95", "yield_99", "portfolio_95", "portfolio_99"):
        frame = distribution[distribution["strategy"] == strategy]
        ax.plot(frame["selected_factor_count"], frame["share_of_test_blocks_pct"], marker="o", label=strategy)
    ax.set_xlabel("Selected factor count")
    ax.set_ylabel("Share of test blocks (%)")
    ax.set_title("Distribution of Selected Factor Counts")
    ax.set_xticks(range(1, 8))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step6_factor_count_distribution.png", dpi=160)
    plt.close(fig)

    butterfly = regime[regime["portfolio"] == "Butterfly_2Y5Y10Y"]
    pivot = butterfly.pivot(index="strategy", columns="regime", values="explained_pnl_variance_pct")
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Explained P&L variance (%)")
    ax.set_title("Butterfly Factor Selection by Volatility Regime")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step6_regime_comparison.png", dpi=160)
    plt.close(fig)


def run_factor_selection(data_path: Path, results_dir: Path, figures_dir: Path) -> dict[str, pd.DataFrame]:
    """Run Step 6 and save all numerical outputs."""
    dates, changes, excluded, total = load_daily_changes(data_path)
    daily, selections, verification_blocks = _evaluate_blocks(changes, dates)
    summary, distribution = _summarize(daily, selections)
    regime = _regime_summary(daily)
    bootstrap = _bootstrap_paired(daily)

    checks = pd.DataFrame(
        [
            {"check": "input_rows", "passed": total == 2973, "value": total, "detail": "raw daily-change rows"},
            {"check": "excluded_rows", "passed": excluded == 1, "value": excluded, "detail": "initial missing-difference row only"},
            {"check": "same_test_observations", "passed": daily.groupby("strategy")["Date"].nunique().nunique() == 1, "value": daily.groupby("strategy")["Date"].nunique().iloc[0], "detail": "all strategies use identical test dates"},
            {"check": "selected_counts_in_range", "passed": bool(selections["selected_factor_count"].between(1, 7).all()), "value": int(selections["selected_factor_count"].between(1, 7).sum()), "detail": "all selections are consecutive PC1-to-PCk models"},
            {"check": "test_blocks_non_overlapping", "passed": bool((verification_blocks["test_start_index"].diff().dropna() == TEST_BLOCK).all()), "value": len(verification_blocks), "detail": "test starts advance exactly one block"},
            {"check": "training_ends_at_test_start", "passed": bool(verification_blocks["train_end_equals_test_start"].all()), "value": int(verification_blocks["train_end_equals_test_start"].sum()), "detail": "PCA fits end before test observations"},
            {"check": "all_7_reconstruction", "passed": bool(verification_blocks["max_pca7_reconstruction_error_bp"].max() < 1e-8), "value": float(verification_blocks["max_pca7_reconstruction_error_bp"].max()), "detail": "all-seven yield reconstruction is numerical identity"},
            {"check": "pca_orthogonality", "passed": bool(verification_blocks["max_pca_orthogonality_error"].max() < 1e-10), "value": float(verification_blocks["max_pca_orthogonality_error"].max()), "detail": "training PCA components are orthogonal"},
        ]
    )
    assert checks["passed"].all()

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(results_dir / "step6_daily_strategy_metrics.csv", index=False)
    selections.to_csv(results_dir / "step6_training_factor_selection.csv", index=False)
    summary.to_csv(results_dir / "step6_strategy_summary.csv", index=False)
    distribution.to_csv(results_dir / "step6_factor_count_distribution.csv", index=False)
    regime.to_csv(results_dir / "step6_regime_strategy_summary.csv", index=False)
    bootstrap.to_csv(results_dir / "step6_paired_bootstrap_ci.csv", index=False)
    verification_blocks.to_csv(results_dir / "step6_block_verification.csv", index=False)
    checks.to_csv(results_dir / "step6_verification.csv", index=False)
    _save_figures(summary, distribution, regime, figures_dir)
    return {
        "daily": daily,
        "selections": selections,
        "summary": summary,
        "distribution": distribution,
        "regime": regime,
        "bootstrap": bootstrap,
        "verification": checks,
        "block_verification": verification_blocks,
    }
