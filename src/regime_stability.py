"""Step 5: rolling PCA stability and training-only volatility regimes."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pca_model import TENORS, fit_pca, load_daily_changes
from src.portfolio_attribution import PORTFOLIOS
from src.walk_forward_attribution import N_FACTORS, TEST_BLOCK, TRAINING_WINDOW


def _match_components(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Match current components to previous ones, allowing signs and switches."""
    similarity = np.abs(previous @ current.T)
    best_order = max(permutations(range(len(TENORS))), key=lambda order: sum(similarity[i, order[i]] for i in range(len(TENORS))))
    matched = current[list(best_order)].copy()
    signed_similarity = np.empty(len(TENORS))
    for index in range(len(TENORS)):
        alignment = float(previous[index] @ matched[index])
        if alignment < 0:
            matched[index] *= -1
        signed_similarity[index] = abs(alignment)
    return matched, signed_similarity, [index + 1 for index in best_order]


def _subspace_similarity(previous: np.ndarray, current: np.ndarray, n_factors: int = N_FACTORS) -> tuple[float, np.ndarray]:
    """Return mean and individual cosines of principal angles between subspaces."""
    singular_values = np.linalg.svd(previous[:n_factors] @ current[:n_factors].T, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(singular_values.mean()), singular_values


def _training_regime(train_changes: pd.DataFrame) -> tuple[str, float, float, float]:
    """Classify the next block using only the preceding training window.

    Daily curve volatility is the root mean square of the seven yield changes.
    The regime statistic is its trailing test-block-length RMS. A block is high
    volatility when that statistic is at least the median of all trailing
    test-block-length RMS values available inside the training window; otherwise
    it is low. This compares like with like and uses training data only.
    """
    values = train_changes.to_numpy(dtype=float)
    daily_curve_vol = np.sqrt(np.mean(values**2, axis=1))
    recent_vol = float(np.sqrt(np.mean(values[-TEST_BLOCK:] ** 2)))
    rolling_rms = np.array(
        [np.sqrt(np.mean(daily_curve_vol[index - TEST_BLOCK + 1 : index + 1] ** 2)) for index in range(TEST_BLOCK - 1, len(daily_curve_vol))]
    )
    threshold = float(np.median(rolling_rms))
    regime = "high_volatility" if recent_vol >= threshold else "low_volatility"
    return regime, recent_vol, threshold, float(daily_curve_vol[-1])


def run_stability_analysis(
    data_path: Path,
    results_dir: Path,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling PCA diagnostics, regime comparisons, and portfolio summaries."""
    dates, changes, excluded, total = load_daily_changes(data_path)
    x = changes.to_numpy(dtype=float)
    n_blocks = (len(changes) - TRAINING_WINDOW) // TEST_BLOCK
    used_test_rows = n_blocks * TEST_BLOCK
    trailing_rows = len(changes) - TRAINING_WINDOW - used_test_rows

    stability_rows: list[dict[str, float | int | str]] = []
    block_rows: list[dict[str, float | int | str]] = []
    daily_rows: list[pd.DataFrame] = []
    previous_components: np.ndarray | None = None

    for block_number in range(n_blocks):
        test_start = TRAINING_WINDOW + block_number * TEST_BLOCK
        test_end = test_start + TEST_BLOCK
        train_start = test_start - TRAINING_WINDOW
        train_end = test_start
        train_dates = dates.iloc[train_start:train_end].reset_index(drop=True)
        train_changes = changes.iloc[train_start:train_end].reset_index(drop=True)
        test_dates = dates.iloc[test_start:test_end].reset_index(drop=True)
        test_x = x[test_start:test_end]

        fitted = fit_pca(train_changes, train_dates, 0, TRAINING_WINDOW)
        pca = fitted.pca
        regime, recent_vol, threshold, last_daily_vol = _training_regime(train_changes)

        current_components = pca.components_.copy()
        if previous_components is None:
            aligned_components = current_components.copy()
            similarities = np.full(len(TENORS), np.nan)
            subspace_mean = np.nan
            subspace_cosines = np.full(N_FACTORS, np.nan)
            matched_components = list(range(1, len(TENORS) + 1))
        else:
            aligned_components, similarities, matched_components = _match_components(previous_components, current_components)
            subspace_mean, subspace_cosines = _subspace_similarity(previous_components, current_components)

        stability_row: dict[str, float | int | str] = {
            "test_block": f"block_{block_number + 1:03d}",
            "test_start": test_dates.iloc[0].strftime("%Y-%m-%d"),
            "test_end": test_dates.iloc[-1].strftime("%Y-%m-%d"),
            "train_start": train_dates.iloc[0].strftime("%Y-%m-%d"),
            "train_end": train_dates.iloc[-1].strftime("%Y-%m-%d"),
            "regime": regime,
            "recent_21d_curve_volatility_bp": recent_vol,
            "training_median_daily_curve_volatility_bp": threshold,
            "last_training_day_curve_volatility_bp": last_daily_vol,
            "subspace_similarity_pc123": subspace_mean,
        }
        for factor in range(1, len(TENORS) + 1):
            stability_row[f"pc{factor}_explained_variance_pct"] = 100 * pca.explained_variance_ratio_[factor - 1]
            stability_row[f"pc{factor}_matched_component"] = matched_components[factor - 1]
            stability_row[f"pc{factor}_loading_similarity"] = similarities[factor - 1]
            if factor <= N_FACTORS:
                stability_row[f"pc{factor}_subspace_cosine"] = subspace_cosines[factor - 1]
            for tenor_index, tenor in enumerate(TENORS):
                stability_row[f"pc{factor}_loading_{tenor}"] = aligned_components[factor - 1, tenor_index]
        stability_rows.append(stability_row)

        centered_test = test_x - pca.mean_
        test_scores = centered_test @ pca.components_.T
        component_changes = [test_scores[:, factor, None] * pca.components_[factor] for factor in range(len(TENORS))]
        previous_components = current_components

        for portfolio, dv01_series in PORTFOLIOS.items():
            dv01 = dv01_series.to_numpy(dtype=float)
            total_pnl = -(test_x @ dv01)
            factor_pnl = np.column_stack([-(component @ dv01) for component in component_changes])
            pc123_pnl = factor_pnl[:, :N_FACTORS].sum(axis=1)
            mean_pnl = float(-(pca.mean_ @ dv01))
            reconstructed_pnl = -(pca.mean_ + sum(component_changes[:N_FACTORS])) @ dv01
            reconstruction_error = total_pnl - reconstructed_pnl
            residual = mean_pnl + reconstruction_error
            factor_variances = np.var(factor_pnl, axis=0, ddof=1)
            total_variance = float(np.var(total_pnl, ddof=1))

            daily_rows.append(
                pd.DataFrame(
                    {
                        "Date": test_dates.dt.strftime("%Y-%m-%d"),
                        "test_block": f"block_{block_number + 1:03d}",
                        "regime": regime,
                        "portfolio": portfolio,
                        "total_pnl_usd": total_pnl,
                        "pc1_pnl_usd": factor_pnl[:, 0],
                        "pc2_pnl_usd": factor_pnl[:, 1],
                        "pc3_pnl_usd": factor_pnl[:, 2],
                        "residual_pnl_usd": residual,
                        "reconstruction_error_usd": reconstruction_error,
                    }
                )
            )
            block_row: dict[str, float | int | str] = {
                "test_block": f"block_{block_number + 1:03d}",
                "test_start": test_dates.iloc[0].strftime("%Y-%m-%d"),
                "test_end": test_dates.iloc[-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "portfolio": portfolio,
                "total_pnl_variance_usd_squared": total_variance,
                "oos_explained_variance_pct": 100 * (1 - np.var(reconstruction_error, ddof=1) / total_variance),
                "reconstruction_rmse_usd": float(np.sqrt(np.mean(reconstruction_error**2))),
                "residual_variance_pct": 100 * np.var(residual, ddof=1) / total_variance,
                "max_add_up_error_usd": float(np.max(np.abs(total_pnl - (pc123_pnl + residual)))),
            }
            for factor, variance in enumerate(factor_variances, start=1):
                block_row[f"pc{factor}_variance_share_pct"] = 100 * float(variance) / total_variance
            block_row["pc4_pc7_factor_variance_share_pct"] = sum(block_row[f"pc{factor}_variance_share_pct"] for factor in range(4, 8))
            block_rows.append(block_row)

    stability = pd.DataFrame(stability_rows)
    block_summary = pd.DataFrame(block_rows)
    daily = pd.concat(daily_rows, ignore_index=True)

    regime_rows: list[dict[str, float | int | str]] = []
    for regime, frame in stability.groupby("regime", sort=False):
        regime_rows.append(
            {
                "regime": regime,
                "test_blocks": len(frame),
                "mean_pc1_explained_variance_pct": frame["pc1_explained_variance_pct"].mean(),
                "mean_pc2_explained_variance_pct": frame["pc2_explained_variance_pct"].mean(),
                "mean_pc3_explained_variance_pct": frame["pc3_explained_variance_pct"].mean(),
                "mean_pc123_explained_variance_pct": frame[[f"pc{i}_explained_variance_pct" for i in range(1, 4)]].sum(axis=1).mean(),
                "mean_pc1_loading_similarity": frame["pc1_loading_similarity"].mean(),
                "mean_pc2_loading_similarity": frame["pc2_loading_similarity"].mean(),
                "mean_pc3_loading_similarity": frame["pc3_loading_similarity"].mean(),
                "mean_pc123_subspace_similarity": frame["subspace_similarity_pc123"].mean(),
                "mean_recent_21d_curve_volatility_bp": frame["recent_21d_curve_volatility_bp"].mean(),
            }
        )
    regime_summary = pd.DataFrame(regime_rows)

    portfolio_rows: list[dict[str, float | int | str]] = []
    for (portfolio, regime), frame in daily.groupby(["portfolio", "regime"], sort=False):
        total_pnl = frame["total_pnl_usd"].to_numpy()
        error = frame["reconstruction_error_usd"].to_numpy()
        residual = frame["residual_pnl_usd"].to_numpy()
        blocks = block_summary[(block_summary["portfolio"] == portfolio) & (block_summary["regime"] == regime)]
        total_variance = float(np.var(total_pnl, ddof=1))
        portfolio_rows.append(
            {
                "portfolio": portfolio,
                "regime": regime,
                "test_observations": len(frame),
                "test_blocks": len(blocks),
                "oos_explained_variance_pct": 100 * (1 - np.var(error, ddof=1) / total_variance),
                "residual_variance_pct": 100 * np.var(residual, ddof=1) / total_variance,
                "reconstruction_rmse_usd": float(np.sqrt(np.mean(error**2))),
                "mean_block_pc4_pc7_factor_variance_share_pct": blocks["pc4_pc7_factor_variance_share_pct"].mean(),
                "max_add_up_error_usd": float(np.max(np.abs(total_pnl - (frame[["pc1_pnl_usd", "pc2_pnl_usd", "pc3_pnl_usd"]].sum(axis=1) + residual))))
            }
        )
    portfolio_regime = pd.DataFrame(portfolio_rows)

    verification = pd.DataFrame([{
        "input_observations": total,
        "excluded_input_observations": excluded,
        "complete_observations": len(changes),
        "training_window": TRAINING_WINDOW,
        "test_block_length": TEST_BLOCK,
        "full_test_blocks": n_blocks,
        "test_observations_used": used_test_rows,
        "trailing_observations_not_tested": trailing_rows,
            "regime_rule": "high if trailing 21-day training-window curve RMS >= median of all available trailing 21-day RMS values in training window; else low",
        "max_daily_add_up_error_usd": float(portfolio_regime["max_add_up_error_usd"].max()),
    }])

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stability.to_csv(results_dir / "step5_rolling_pca_stability.csv", index=False)
    regime_summary.to_csv(results_dir / "step5_regime_summary.csv", index=False)
    portfolio_regime.to_csv(results_dir / "step5_portfolio_regime_attribution.csv", index=False)
    block_summary.to_csv(results_dir / "step5_block_diagnostics.csv", index=False)
    daily.to_csv(results_dir / "step5_daily_attribution.csv", index=False)
    verification.to_csv(results_dir / "step5_verification.csv", index=False)
    _save_figures(stability, regime_summary, portfolio_regime, figures_dir)
    return stability, regime_summary, portfolio_regime, verification


def _save_figures(stability: pd.DataFrame, regime_summary: pd.DataFrame, portfolio_regime: pd.DataFrame, figures_dir: Path) -> None:
    """Save Step 5 stability, regime, and portfolio comparison charts."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for factor in range(1, 4):
        ax.plot(stability["test_end"], stability[f"pc{factor}_explained_variance_pct"], label=f"PC{factor}")
    ax.set_title("Rolling PCA Explained Variance")
    ax.set_ylabel("Explained variance (%)")
    ax.set_xlabel("Test block end")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step5_rolling_explained_variance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for factor in range(1, 4):
        ax.plot(stability["test_end"], stability[f"pc{factor}_loading_similarity"], label=f"PC{factor}")
    ax.plot(stability["test_end"], stability["subspace_similarity_pc123"], color="black", linestyle="--", label="PC1–PC3 subspace")
    ax.set_title("Rolling PCA Loading and Subspace Stability")
    ax.set_ylabel("Similarity to prior block")
    ax.set_xlabel("Test block end")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step5_loading_stability.png", dpi=160)
    plt.close(fig)

    regime_plot = regime_summary.set_index("regime")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(regime_plot.index))
    width = 0.22
    for offset, factor in enumerate(range(1, 4)):
        ax.bar(x_positions + (offset - 1) * width, regime_plot[f"mean_pc{factor}_loading_similarity"], width, label=f"PC{factor} loading")
    ax.bar(x_positions + width, regime_plot["mean_pc123_subspace_similarity"], width, label="PC1–PC3 subspace", alpha=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regime_plot.index)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean similarity")
    ax.set_title("PCA Stability by Training-Only Volatility Regime")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step5_regime_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pivot = portfolio_regime.pivot(index="portfolio", columns="regime", values="reconstruction_rmse_usd")
    pivot.plot(kind="bar", ax=axes[0], color=["#4472C4", "#C00000"])
    axes[0].set_title("P&L Reconstruction RMSE by Regime")
    axes[0].set_ylabel("RMSE ($)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    factor_pivot = portfolio_regime.pivot(index="portfolio", columns="regime", values="mean_block_pc4_pc7_factor_variance_share_pct")
    factor_pivot.plot(kind="bar", ax=axes[1], color=["#70AD47", "#ED7D31"])
    axes[1].set_title("Omitted PC4–PC7 Share by Regime")
    axes[1].set_ylabel("Average variance share (%)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "step5_portfolio_regime_error.png", dpi=160)
    plt.close(fig)
