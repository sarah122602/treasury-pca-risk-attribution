"""Independent Step 5 research-quality and robustness audit."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pca_model import TENORS, fit_pca, load_daily_changes
from src.portfolio_attribution import PORTFOLIOS
from src.walk_forward_attribution import TEST_BLOCK, TRAINING_WINDOW


COMPONENT_COUNTS = (1, 2, 3, 4, 5, 7)
WINDOW_LENGTHS = (125, 250, 500)


def _component_match(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Align current PCA vectors to prior vectors, allowing signs and switches."""
    similarity = np.abs(previous @ current.T)
    order = max(permutations(range(len(TENORS))), key=lambda candidate: sum(similarity[i, candidate[i]] for i in range(len(TENORS))))
    aligned = current[list(order)].copy()
    similarities = np.empty(len(TENORS))
    for index in range(len(TENORS)):
        dot_product = float(previous[index] @ aligned[index])
        if dot_product < 0:
            aligned[index] *= -1
        similarities[index] = abs(dot_product)
    return aligned, similarities, [value + 1 for value in order]


def _subspace_similarity(previous: np.ndarray, current: np.ndarray, n_factors: int = 3) -> tuple[float, np.ndarray]:
    """Return the mean and individual principal-angle cosines."""
    cosines = np.linalg.svd(previous[:n_factors] @ current[:n_factors].T, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    return float(cosines.mean()), cosines


def _training_regime(train_changes: pd.DataFrame, quantile: float) -> tuple[str, float, float]:
    """Classify a test block using only trailing volatility in its training data."""
    daily_curve_volatility = np.sqrt(np.mean(train_changes.to_numpy(dtype=float) ** 2, axis=1))
    rolling_rms = np.array(
        [
            np.sqrt(np.mean(daily_curve_volatility[index - TEST_BLOCK + 1 : index + 1] ** 2))
            for index in range(TEST_BLOCK - 1, len(daily_curve_volatility))
        ]
    )
    threshold = float(np.quantile(rolling_rms, quantile))
    recent = float(rolling_rms[-1])
    regime = "high_volatility" if recent >= threshold else "low_volatility"
    return regime, recent, threshold


def audit_data(data_path: Path, yields_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Independently verify dates, units, missingness, and daily-change alignment."""
    yields = pd.read_csv(yields_path)
    changes = pd.read_csv(data_path)
    yield_dates = pd.to_datetime(yields["Date"], errors="coerce")
    change_dates = pd.to_datetime(changes["Date"], errors="coerce")
    expected_changes = yields[TENORS].apply(pd.to_numeric, errors="coerce").diff() * 100.0
    observed_changes = changes[TENORS].apply(pd.to_numeric, errors="coerce")
    comparable = expected_changes.notna() & observed_changes.notna()
    max_change_difference = float((expected_changes[comparable] - observed_changes[comparable]).abs().max().max())

    checks = [
        {"check": "yield_date_parse", "passed": bool(yield_dates.notna().all()), "value": int(yield_dates.notna().sum()), "detail": "all cleaned yield dates parse"},
        {"check": "change_date_parse", "passed": bool(change_dates.notna().all()), "value": int(change_dates.notna().sum()), "detail": "all change dates parse"},
        {"check": "yield_dates_unique_sorted", "passed": bool(yield_dates.is_unique and yield_dates.is_monotonic_increasing), "value": len(yields), "detail": "cleaned yield dates are unique and sorted"},
        {"check": "change_dates_equal_yield_dates", "passed": bool(yield_dates.equals(change_dates)), "value": int(yield_dates.equals(change_dates)), "detail": "daily-change rows align one-for-one with yield rows"},
        {"check": "yield_missing_values", "passed": bool(yields[TENORS].isna().sum().sum() == 0), "value": int(yields[TENORS].isna().sum().sum()), "detail": "cleaned yield panel contains no missing maturity observations"},
        {"check": "initial_change_missing_values", "passed": bool(observed_changes.iloc[0].isna().all() and observed_changes.iloc[1:].notna().all().all()), "value": int(observed_changes.isna().sum().sum()), "detail": "only the first difference row is missing by construction"},
        {"check": "daily_changes_reproduce_first_differences", "passed": bool(max_change_difference < 1e-10), "value": max_change_difference, "detail": "observed changes equal 100 times yield first differences"},
        {"check": "yield_units_plausible_percent", "passed": bool(yields[TENORS].to_numpy(dtype=float).min() >= 0 and yields[TENORS].to_numpy(dtype=float).max() < 20), "value": float(yields[TENORS].to_numpy(dtype=float).max()), "detail": "values are plausible percent yields"},
        {"check": "change_units_plausible_bp", "passed": bool(observed_changes.iloc[1:].abs().max().max() < 500), "value": float(observed_changes.iloc[1:].abs().max().max()), "detail": "values are treated as basis-point changes"},
    ]
    return pd.DataFrame(checks), yields, changes


def evaluate_walk_forward(
    changes: pd.DataFrame,
    dates: pd.Series,
    window: int,
    test_starts: list[int],
    regime_quantile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit training-only PCA models and return daily, block, and fit diagnostics."""
    x = changes.to_numpy(dtype=float)
    daily_frames: list[pd.DataFrame] = []
    block_rows: list[dict[str, float | int | str]] = []
    fit_rows: list[dict[str, float | int | str]] = []
    for block_number, test_start in enumerate(test_starts, start=1):
        train_start = test_start - window
        test_end = test_start + TEST_BLOCK
        train_changes = changes.iloc[train_start:test_start].reset_index(drop=True)
        train_dates = dates.iloc[train_start:test_start].reset_index(drop=True)
        test_dates = dates.iloc[test_start:test_end].reset_index(drop=True)
        test_x = x[test_start:test_end]
        fitted = fit_pca(train_changes, train_dates, 0, window)
        pca = fitted.pca
        regime = "unclassified"
        recent_vol = np.nan
        threshold = np.nan
        if regime_quantile is not None:
            regime, recent_vol, threshold = _training_regime(train_changes, regime_quantile)
        scores = (test_x - pca.mean_) @ pca.components_.T
        factor_changes = [scores[:, factor, None] * pca.components_[factor] for factor in range(len(TENORS))]
        block_id = f"block_{block_number:03d}"
        fit_rows.append(
            {
                "test_block": block_id,
                "test_start_index": test_start,
                "train_start_index": train_start,
                "train_end_index_exclusive": test_start,
                "test_end_index_exclusive": test_end,
                "train_start": train_dates.iloc[0].strftime("%Y-%m-%d"),
                "train_end": train_dates.iloc[-1].strftime("%Y-%m-%d"),
                "test_start": test_dates.iloc[0].strftime("%Y-%m-%d"),
                "test_end": test_dates.iloc[-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "recent_21d_curve_volatility_bp": recent_vol,
                "training_volatility_threshold_bp": threshold,
                "max_pca_orthogonality_error": float(np.abs(pca.components_ @ pca.components_.T - np.eye(len(TENORS))).max()),
            }
        )
        for portfolio, dv01_series in PORTFOLIOS.items():
            dv01 = dv01_series.to_numpy(dtype=float)
            actual = -(test_x @ dv01)
            factor_pnl = np.column_stack([-(component @ dv01) for component in factor_changes])
            training_mean_pnl = float(-(pca.mean_ @ dv01))
            row = pd.DataFrame(
                {
                    "Date": test_dates.dt.strftime("%Y-%m-%d"),
                    "test_block": block_id,
                    "portfolio": portfolio,
                    "regime": regime,
                    "total_pnl_usd": actual,
                    "training_mean_pnl_usd": training_mean_pnl,
                }
            )
            for component_count in COMPONENT_COUNTS:
                prediction = -((pca.mean_ + sum(factor_changes[:component_count])) @ dv01)
                row[f"pca{component_count}_prediction_usd"] = prediction
                row[f"pca{component_count}_error_usd"] = actual - prediction
            for factor in range(len(TENORS)):
                row[f"pc{factor + 1}_pnl_usd"] = factor_pnl[:, factor]
            row["pca3_additive_reconciliation_error_usd"] = actual - (
                row["pc1_pnl_usd"] + row["pc2_pnl_usd"] + row["pc3_pnl_usd"]
                + training_mean_pnl + row["pca3_error_usd"]
            )
            daily_frames.append(row)

            block_row: dict[str, float | int | str] = {
                "test_block": block_id,
                "portfolio": portfolio,
                "regime": regime,
                "test_start": test_dates.iloc[0].strftime("%Y-%m-%d"),
                "test_end": test_dates.iloc[-1].strftime("%Y-%m-%d"),
                "max_all_factor_reconstruction_error_usd": float(np.max(np.abs(actual - (-((pca.mean_ + sum(factor_changes)) @ dv01))))),
            }
            for component_count in COMPONENT_COUNTS:
                error = row[f"pca{component_count}_error_usd"].to_numpy()
                block_row[f"pca{component_count}_rmse_usd"] = float(np.sqrt(np.mean(error**2)))
                block_row[f"pca{component_count}_error_variance_usd_squared"] = float(np.var(error, ddof=1))
            total_variance = float(np.var(actual, ddof=1))
            block_row["actual_pnl_std_usd"] = float(np.std(actual, ddof=1))
            block_row["pc4_pc7_variance_share_pct"] = 100 * float(np.var(factor_pnl[:, 3:].sum(axis=1), ddof=1)) / total_variance
            block_rows.append(block_row)

    return pd.concat(daily_frames, ignore_index=True), pd.DataFrame(block_rows), pd.DataFrame(fit_rows)


def _aggregate_metrics(daily: pd.DataFrame, component_count: int, group_columns: list[str]) -> pd.DataFrame:
    """Calculate explained variance, RMSE, OOS R2, and residual diagnostics."""
    rows: list[dict[str, float | int | str]] = []
    groups = [((), daily)] if not group_columns else daily.groupby(group_columns, sort=False)
    for group_values, frame in groups:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        actual = frame["total_pnl_usd"].to_numpy()
        error = frame[f"pca{component_count}_error_usd"].to_numpy()
        training_mean = frame["training_mean_pnl_usd"].to_numpy()
        total_variance = float(np.var(actual, ddof=1))
        benchmark_sse = float(np.sum((actual - training_mean) ** 2))
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "component_count": component_count,
                "test_observations": len(frame),
                "explained_variance_pct": 100 * (1 - np.var(error, ddof=1) / total_variance),
                "rmse_usd": float(np.sqrt(np.mean(error**2))),
                "normalized_rmse": float(np.sqrt(np.mean(error**2)) / np.std(actual, ddof=1)),
                "oos_r2_vs_training_mean": 1 - float(np.sum(error**2)) / benchmark_sse if benchmark_sse > 0 else np.nan,
                "mean_residual_usd": float(np.mean(error)),
                "residual_variance_usd_squared": float(np.var(error, ddof=1)),
                "actual_pnl_std_usd": float(np.std(actual, ddof=1)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _stability_comparisons(changes: pd.DataFrame, dates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare adjacent, disjoint, and widely separated PCA training windows."""
    x = changes
    window = TRAINING_WINDOW
    all_starts = list(range(window, len(x) - TEST_BLOCK + 1, TEST_BLOCK))
    comparisons: list[dict[str, float | int | str]] = []

    def fit_at(test_start: int) -> np.ndarray:
        return fit_pca(x.iloc[test_start - window:test_start].reset_index(drop=True), dates.iloc[test_start - window:test_start].reset_index(drop=True), 0, window).pca.components_

    def add_pair(kind: str, first: int, second: int) -> None:
        first_components = fit_at(first)
        second_components = fit_at(second)
        _, similarities, matches = _component_match(first_components, second_components)
        subspace, cosines = _subspace_similarity(first_components, second_components)
        row: dict[str, float | int | str] = {
            "comparison_type": kind,
            "window_a_test_start": dates.iloc[first].strftime("%Y-%m-%d"),
            "window_b_test_start": dates.iloc[second].strftime("%Y-%m-%d"),
            "separation_observations": second - first,
            "pc123_subspace_similarity": subspace,
        }
        for factor in range(1, 4):
            row[f"pc{factor}_loading_similarity"] = similarities[factor - 1]
            row[f"pc{factor}_principal_angle_cosine"] = cosines[factor - 1]
            row[f"pc{factor}_matched_component"] = matches[factor - 1]
        comparisons.append(row)

    for first, second in zip(all_starts, all_starts[1:]):
        add_pair("adjacent_overlapping", first, second)
    disjoint_starts = list(range(window, len(x) - TEST_BLOCK + 1, window))
    for first, second in zip(disjoint_starts, disjoint_starts[1:]):
        add_pair("non_overlapping", first, second)
    separated = [all_starts[0], all_starts[len(all_starts) // 2], all_starts[-1]]
    for first, second in zip(separated, separated[1:]):
        add_pair("widely_separated", first, second)
    comparisons_frame = pd.DataFrame(comparisons)
    summary = (
        comparisons_frame.groupby("comparison_type", sort=False)
        .agg(
            comparisons=("comparison_type", "size"),
            mean_pc1_loading_similarity=("pc1_loading_similarity", "mean"),
            mean_pc2_loading_similarity=("pc2_loading_similarity", "mean"),
            mean_pc3_loading_similarity=("pc3_loading_similarity", "mean"),
            mean_pc123_subspace_similarity=("pc123_subspace_similarity", "mean"),
            component_switches_pc1=("pc1_matched_component", lambda values: int((values != 1).sum())),
            component_switches_pc2=("pc2_matched_component", lambda values: int((values != 2).sum())),
            component_switches_pc3=("pc3_matched_component", lambda values: int((values != 3).sum())),
        )
        .reset_index()
    )
    return comparisons_frame, summary


def _bootstrap_blocks(daily: pd.DataFrame, seed: int = 20261122, repetitions: int = 2000) -> pd.DataFrame:
    """Use a block bootstrap over 21-day test blocks for uncertainty intervals."""
    rng = np.random.default_rng(seed)
    block_ids = daily["test_block"].drop_duplicates().to_numpy()
    portfolios = list(PORTFOLIOS.columns)
    block_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for portfolio in portfolios:
        frame = daily[daily["portfolio"] == portfolio].sort_values(["test_block", "Date"])
        actual = frame["total_pnl_usd"].to_numpy().reshape(len(block_ids), TEST_BLOCK)
        error = frame["pca3_error_usd"].to_numpy().reshape(len(block_ids), TEST_BLOCK)
        block_arrays[portfolio] = actual, error
    draws: dict[tuple[str, str], list[float]] = {}
    for _ in range(repetitions):
        sampled = rng.choice(block_ids, size=len(block_ids), replace=True)
        sampled_indices = np.array([np.flatnonzero(block_ids == block)[0] for block in sampled])
        for portfolio in portfolios:
            actual_blocks, error_blocks = block_arrays[portfolio]
            actual = actual_blocks[sampled_indices].reshape(-1)
            error = error_blocks[sampled_indices].reshape(-1)
            explained = 100 * (1 - np.var(error, ddof=1) / np.var(actual, ddof=1))
            rmse = float(np.sqrt(np.mean(error**2)))
            draws.setdefault((portfolio, "explained_variance_pct"), []).append(float(explained))
            draws.setdefault((portfolio, "rmse_usd"), []).append(rmse)
        explained_values = {
            portfolio: draws[(portfolio, "explained_variance_pct")][-1] for portfolio in portfolios
        }
        rmse_values = {portfolio: draws[(portfolio, "rmse_usd")][-1] for portfolio in portfolios}
        non_butterfly = [portfolio for portfolio in portfolios if portfolio != "Butterfly_2Y5Y10Y"]
        draws.setdefault(("Butterfly_gap", "explained_variance_pct"), []).append(
            float(np.mean([explained_values[p] for p in non_butterfly]) - explained_values["Butterfly_2Y5Y10Y"])
        )
        draws.setdefault(("Butterfly_gap", "rmse_usd"), []).append(
            float(rmse_values["Butterfly_2Y5Y10Y"] - np.mean([rmse_values[p] for p in non_butterfly]))
        )

    rows = []
    for (entity, metric), values in draws.items():
        values_array = np.asarray(values)
        if entity == "Butterfly_gap":
            observed = float(values_array.mean())
        else:
            actual, error = block_arrays[entity]
            actual = actual.reshape(-1)
            error = error.reshape(-1)
            observed = float(100 * (1 - np.var(error, ddof=1) / np.var(actual, ddof=1))) if metric == "explained_variance_pct" else float(np.sqrt(np.mean(error**2)))
        rows.append(
            {
                "entity": entity,
                "metric": metric,
                "observed": observed,
                "bootstrap_lower_95_pct": float(np.quantile(values_array, 0.025)),
                "bootstrap_upper_95_pct": float(np.quantile(values_array, 0.975)),
                "bootstrap_repetitions": repetitions,
                "bootstrap_unit": "non-overlapping 21-day test block",
            }
        )
    return pd.DataFrame(rows)


def _save_figures(stability: pd.DataFrame, window_summary: pd.DataFrame, regime_summary: pd.DataFrame, figures_dir: Path) -> None:
    """Save quality-review charts."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for window in WINDOW_LENGTHS:
        frame = window_summary[(window_summary["window"] == window) & (window_summary["sample"] == "common") & (window_summary["portfolio"] == "Butterfly_2Y5Y10Y") & (window_summary["component_count"] == 3)]
        ax.bar(str(window), frame["explained_variance_pct"].iloc[0], color="#4472C4")
    ax.set_title("Butterfly PC1–PC3 Explained P&L Variance: Common Test Dates")
    ax.set_ylabel("Explained variance (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_review_window_robustness.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for kind, frame in stability.groupby("comparison_type", sort=False):
        ax.scatter(frame["separation_observations"], frame["pc123_subspace_similarity"], label=kind, alpha=0.75)
    ax.set_title("PC1–PC3 Subspace Similarity versus Window Separation")
    ax.set_xlabel("Training-window separation (observations)")
    ax.set_ylabel("Subspace similarity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_review_factor_stability.png", dpi=160)
    plt.close(fig)

    pivot = regime_summary[regime_summary["regime_quantile"] == 0.5].pivot(index="portfolio", columns="regime", values="normalized_rmse")
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, color=["#C00000", "#70AD47"])
    ax.set_title("Normalized PC1–PC3 Reconstruction Error by Regime")
    ax.set_ylabel("RMSE / actual P&L standard deviation")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_review_regime_errors.png", dpi=160)
    plt.close(fig)


def run_quality_review(root: Path) -> dict[str, pd.DataFrame]:
    """Run the complete independent audit and save all results."""
    data_path = root / "data" / "treasury_daily_changes_bp.csv"
    yields_path = root / "data" / "treasury_par_yields.csv"
    results_dir = root / "results"
    figures_dir = root / "figures"
    data_checks, yields, changes_raw = audit_data(data_path, yields_path)
    dates, changes, excluded, total = load_daily_changes(data_path)
    assert dates.is_monotonic_increasing and dates.is_unique
    assert excluded == 1
    assert len(changes) == len(yields) - 1

    stability, stability_summary = _stability_comparisons(changes, dates)
    baseline_starts = list(range(TRAINING_WINDOW, len(changes) - TEST_BLOCK + 1, TEST_BLOCK))
    all_window_rows: list[pd.DataFrame] = []
    component_rows: list[pd.DataFrame] = []
    for window in WINDOW_LENGTHS:
        full_starts = list(range(window, len(changes) - TEST_BLOCK + 1, TEST_BLOCK))
        common_starts = list(range(max(WINDOW_LENGTHS), len(changes) - TEST_BLOCK + 1, TEST_BLOCK))
        for sample, starts in (("full", full_starts), ("common", common_starts)):
            daily, blocks, fits = evaluate_walk_forward(changes, dates, window, starts)
            for component_count in COMPONENT_COUNTS:
                summary = _aggregate_metrics(daily, component_count, ["portfolio"])
                summary.insert(0, "sample", sample)
                summary.insert(0, "window", window)
                all_window_rows.append(summary)
            if window == TRAINING_WINDOW and sample == "full":
                baseline_daily = daily
                baseline_blocks = blocks
                baseline_fits = fits
        daily_common, _, _ = evaluate_walk_forward(changes, dates, window, common_starts)
        component_summary = _aggregate_metrics(daily_common, 3, ["portfolio"])
        component_summary.insert(0, "window", window)
        component_summary.insert(0, "sample", "common")
        component_rows.append(component_summary)

    window_summary = pd.concat(all_window_rows, ignore_index=True)
    component_summary = pd.concat(component_rows, ignore_index=True)

    regime_daily, _, regime_fits = evaluate_walk_forward(changes, dates, TRAINING_WINDOW, baseline_starts, regime_quantile=0.5)
    regime_rows = []
    for quantile in (0.5, 0.75):
        daily, _, fits = evaluate_walk_forward(changes, dates, TRAINING_WINDOW, baseline_starts, regime_quantile=quantile)
        summary = _aggregate_metrics(daily, 3, ["portfolio", "regime"])
        summary.insert(0, "regime_quantile", quantile)
        regime_rows.append(summary)
    regime_summary = pd.concat(regime_rows, ignore_index=True)

    extreme_rows = []
    butterfly = baseline_daily[baseline_daily["portfolio"] == "Butterfly_2Y5Y10Y"].copy()
    cutoff = butterfly["total_pnl_usd"].abs().quantile(0.99)
    for label, frame in (("all_days", butterfly), ("excluding_top_1pct_abs_pnl", butterfly[butterfly["total_pnl_usd"].abs() < cutoff])):
        metrics = _aggregate_metrics(frame, 3, [])
        metrics.insert(0, "sample", label)
        metrics.insert(0, "cutoff_abs_pnl_usd", cutoff)
        extreme_rows.append(metrics)
    extreme_summary = pd.concat(extreme_rows, ignore_index=True)

    bootstrap = _bootstrap_blocks(baseline_daily)
    verification_rows = data_checks.to_dict("records")
    verification_rows.extend(
        [
            {"check": "walk_forward_test_blocks_non_overlapping", "passed": bool(all(np.diff(baseline_fits["test_start_index"]) == TEST_BLOCK)), "value": len(baseline_fits), "detail": "test starts advance exactly one block"},
            {"check": "training_ends_before_test_starts", "passed": bool((baseline_fits["train_end_index_exclusive"] == baseline_fits["test_start_index"]).all()), "value": int((baseline_fits["train_end_index_exclusive"] == baseline_fits["test_start_index"]).sum()), "detail": "each PCA fit ends immediately before its test block"},
            {"check": "all_seven_factor_reconstruction", "passed": bool(baseline_blocks["max_all_factor_reconstruction_error_usd"].max() < 1e-6), "value": float(baseline_blocks["max_all_factor_reconstruction_error_usd"].max()), "detail": "seven-factor reconstruction equals actual P&L"},
            {"check": "pca_component_orthogonality", "passed": bool(baseline_fits["max_pca_orthogonality_error"].max() < 1e-10), "value": float(baseline_fits["max_pca_orthogonality_error"].max()), "detail": "training PCA components are orthogonal"},
            {"check": "baseline_regime_labels_training_only", "passed": bool(regime_fits["regime"].notna().all() and (regime_fits["test_start_index"] >= TRAINING_WINDOW).all()), "value": int(regime_fits["regime"].notna().sum()), "detail": "baseline labels were created from preceding training windows"},
            {"check": "baseline_additive_pnl_reconciliation", "passed": bool(baseline_daily["pca3_additive_reconciliation_error_usd"].abs().max() < 1e-6), "value": float(baseline_daily["pca3_additive_reconciliation_error_usd"].abs().max()), "detail": "PC1-PC3 plus training mean and residual equal actual P&L"},
        ]
    )
    verification = pd.DataFrame(verification_rows)
    assert verification["passed"].all()

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    data_checks.to_csv(results_dir / "quality_review_data_checks.csv", index=False)
    verification.to_csv(results_dir / "quality_review_verification.csv", index=False)
    stability.to_csv(results_dir / "quality_review_factor_stability.csv", index=False)
    stability_summary.to_csv(results_dir / "quality_review_factor_stability_summary.csv", index=False)
    window_summary.to_csv(results_dir / "quality_review_window_robustness.csv", index=False)
    component_summary.to_csv(results_dir / "quality_review_component_reconstruction.csv", index=False)
    regime_summary.to_csv(results_dir / "quality_review_regime_metrics.csv", index=False)
    extreme_summary.to_csv(results_dir / "quality_review_extreme_days.csv", index=False)
    bootstrap.to_csv(results_dir / "quality_review_bootstrap_ci.csv", index=False)
    baseline_fits.to_csv(results_dir / "quality_review_baseline_fit_audit.csv", index=False)
    _save_figures(stability, window_summary, regime_summary, figures_dir)
    return {
        "data_checks": data_checks,
        "verification": verification,
        "stability": stability,
        "stability_summary": stability_summary,
        "window_summary": window_summary,
        "component_summary": component_summary,
        "regime_summary": regime_summary,
        "extreme_summary": extreme_summary,
        "bootstrap": bootstrap,
        "baseline_daily": baseline_daily,
        "baseline_blocks": baseline_blocks,
    }
