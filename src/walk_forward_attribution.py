"""Walk-forward out-of-sample PCA P&L attribution for Step 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pca_model import TENORS, fit_pca, load_daily_changes
from src.portfolio_attribution import PORTFOLIOS


TRAINING_WINDOW = 250
TEST_BLOCK = 21
N_FACTORS = 3


def run_walk_forward(
    data_path: Path,
    results_dir: Path,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Fit PCA on each trailing training window and score the next test block."""
    dates, changes, excluded, total = load_daily_changes(data_path)
    x = changes.to_numpy(dtype=float)
    daily_rows: list[pd.DataFrame] = []
    block_rows: list[dict[str, float | int | str]] = []
    factor_rows: list[dict[str, float | str]] = []

    available_test_rows = len(changes) - TRAINING_WINDOW
    n_blocks = available_test_rows // TEST_BLOCK
    used_test_rows = n_blocks * TEST_BLOCK
    trailing_rows = available_test_rows - used_test_rows

    for block_number in range(n_blocks):
        test_start = TRAINING_WINDOW + block_number * TEST_BLOCK
        test_end = test_start + TEST_BLOCK
        train_start = test_start - TRAINING_WINDOW
        train_end = test_start
        train_dates = dates.iloc[train_start:train_end].reset_index(drop=True)
        train_changes = changes.iloc[train_start:train_end].reset_index(drop=True)
        test_dates = dates.iloc[test_start:test_end].reset_index(drop=True)
        test_x = x[test_start:test_end]

        # fit_pca uses sklearn PCA, centers only the training data, and does
        # not standardize maturities. Test observations are transformed using
        # this fitted training mean and these fixed training loadings.
        fitted = fit_pca(train_changes, train_dates, 0, TRAINING_WINDOW)
        pca = fitted.pca
        centered_test = test_x - pca.mean_
        test_scores = centered_test @ pca.components_.T
        component_changes = [
            test_scores[:, factor, None] * pca.components_[factor]
            for factor in range(len(TENORS))
        ]
        pca3_reconstructed_changes = pca.mean_ + sum(component_changes[:N_FACTORS])

        block_id = f"block_{block_number + 1:03d}"
        for portfolio, dv01_series in PORTFOLIOS.items():
            dv01 = dv01_series.to_numpy(dtype=float)
            total_pnl = -(test_x @ dv01)
            factor_pnl = np.column_stack([-(component @ dv01) for component in component_changes])
            pc123_pnl = factor_pnl[:, :N_FACTORS].sum(axis=1)
            mean_pnl = float(-(pca.mean_ @ dv01))
            reconstructed_pnl = -(pca3_reconstructed_changes @ dv01)
            pca3_error = total_pnl - reconstructed_pnl
            # Exact decomposition: total = PC1 + PC2 + PC3 + residual.
            residual = mean_pnl + pca3_error

            daily_rows.append(
                pd.DataFrame(
                    {
                        "Date": test_dates.dt.strftime("%Y-%m-%d"),
                        "test_block": block_id,
                        "train_start": train_dates.iloc[0].strftime("%Y-%m-%d"),
                        "train_end": train_dates.iloc[-1].strftime("%Y-%m-%d"),
                        "portfolio": portfolio,
                        "total_pnl_usd": total_pnl,
                        "pc1_pnl_usd": factor_pnl[:, 0],
                        "pc2_pnl_usd": factor_pnl[:, 1],
                        "pc3_pnl_usd": factor_pnl[:, 2],
                        "residual_pnl_usd": residual,
                        "pc123_reconstructed_pnl_usd": reconstructed_pnl,
                        "pca3_reconstruction_error_usd": pca3_error,
                    }
                )
            )

            total_variance = float(np.var(total_pnl, ddof=1))
            reconstruction_error_variance = float(np.var(pca3_error, ddof=1))
            block_rows.append(
                {
                    "test_block": block_id,
                    "test_start": test_dates.iloc[0].strftime("%Y-%m-%d"),
                    "test_end": test_dates.iloc[-1].strftime("%Y-%m-%d"),
                    "portfolio": portfolio,
                    "total_pnl_variance_usd_squared": total_variance,
                    "oos_explained_variance_pct": 100 * (1 - reconstruction_error_variance / total_variance),
                    "residual_variance_usd_squared": float(np.var(residual, ddof=1)),
                    "reconstruction_error_variance_usd_squared": reconstruction_error_variance,
                    "reconstruction_rmse_usd": float(np.sqrt(np.mean(pca3_error**2))),
                    "max_add_up_error_usd": float(np.max(np.abs(total_pnl - (pc123_pnl + residual)))),
                }
            )

            factor_variances = np.var(factor_pnl, axis=0, ddof=1)
            factor_row: dict[str, float | str] = {"portfolio": portfolio}
            for factor, variance in enumerate(factor_variances, start=1):
                factor_row[f"pc{factor}_variance_usd_squared"] = float(variance)
            factor_row["block_total_pnl_variance_usd_squared"] = float(np.var(total_pnl, ddof=1))
            factor_rows.append(factor_row)

    daily = pd.concat(daily_rows, ignore_index=True)
    block_summary = pd.DataFrame(block_rows)
    factor_by_block = pd.DataFrame(factor_rows)

    summary_rows: list[dict[str, float | int | str]] = []
    for portfolio, frame in daily.groupby("portfolio", sort=False):
        total_pnl = frame["total_pnl_usd"].to_numpy()
        error = frame["pca3_reconstruction_error_usd"].to_numpy()
        residual = frame["residual_pnl_usd"].to_numpy()
        total_variance = float(np.var(total_pnl, ddof=1))
        error_variance = float(np.var(error, ddof=1))
        factor_sum = frame[["pc1_pnl_usd", "pc2_pnl_usd", "pc3_pnl_usd"]].sum(axis=1).to_numpy()
        summary_rows.append(
            {
                "portfolio": portfolio,
                "test_observations": len(frame),
                "test_blocks": n_blocks,
                "net_dv01_usd_per_bp": float(PORTFOLIOS[portfolio].sum()),
                "total_pnl_std_usd": float(np.std(total_pnl, ddof=1)),
                "total_pnl_variance_usd_squared": total_variance,
                "oos_explained_variance_pct": 100 * (1 - error_variance / total_variance),
                "pc123_variance_share_pct": 100 * np.var(factor_sum, ddof=1) / total_variance,
                "residual_variance_usd_squared": float(np.var(residual, ddof=1)),
                "reconstruction_error_variance_usd_squared": error_variance,
                "residual_variance_pct": 100 * np.var(residual, ddof=1) / total_variance,
                "reconstruction_rmse_usd": float(np.sqrt(np.mean(error**2))),
                "max_add_up_error_usd": float(np.max(np.abs(total_pnl - (factor_sum + residual)))),
            }
        )
    summary = pd.DataFrame(summary_rows)

    factor_summary_rows: list[dict[str, float | str]] = []
    for portfolio, frame in daily.groupby("portfolio", sort=False):
        block_factor = factor_by_block[factor_by_block["portfolio"] == portfolio]
        # Block-level factor P&Ls are sufficient for a variance profile; the
        # average across rolling fits avoids treating a changing PC sign as a
        # directional economic signal.
        total_variances = block_factor["block_total_pnl_variance_usd_squared"]
        row: dict[str, float | str] = {"portfolio": portfolio}
        for factor in range(1, len(TENORS) + 1):
            block_shares = 100 * block_factor[f"pc{factor}_variance_usd_squared"] / total_variances
            row[f"pc{factor}_variance_share_pct"] = float(block_shares.mean())
        row["pc4_pc7_factor_variance_share_pct"] = sum(
            row[f"pc{factor}_variance_share_pct"] for factor in range(4, 8)
        )
        factor_summary_rows.append(row)
    factor_summary = pd.DataFrame(factor_summary_rows)

    verification = pd.DataFrame(
        [
            {
                "input_observations": total,
                "excluded_input_observations": excluded,
                "complete_observations": len(changes),
                "training_window": TRAINING_WINDOW,
                "test_block_length": TEST_BLOCK,
                "full_test_blocks": n_blocks,
                "test_observations_used": used_test_rows,
                "trailing_observations_not_tested": trailing_rows,
                "first_test_date": dates.iloc[TRAINING_WINDOW].strftime("%Y-%m-%d"),
                "last_test_date": dates.iloc[TRAINING_WINDOW + used_test_rows - 1].strftime("%Y-%m-%d"),
                "max_daily_add_up_error_usd": float(summary["max_add_up_error_usd"].max()),
            }
        ]
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(results_dir / "walk_forward_daily_attribution.csv", index=False)
    summary.to_csv(results_dir / "walk_forward_summary.csv", index=False)
    block_summary.to_csv(results_dir / "walk_forward_block_summary.csv", index=False)
    factor_summary.to_csv(results_dir / "walk_forward_factor_variance.csv", index=False)
    verification.to_csv(results_dir / "walk_forward_verification.csv", index=False)
    _save_figures(summary, block_summary, factor_summary, figures_dir)
    return daily, summary, factor_summary, verification, {"n_blocks": n_blocks, "trailing_rows": trailing_rows}


def _save_figures(
    summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Save Step 4 comparison and butterfly-diagnostic charts."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(summary["portfolio"], summary["oos_explained_variance_pct"], color="#4472C4")
    ax.set_ylabel("Out-of-sample variance explained (%)")
    ax.set_title("Walk-Forward PC1–PC3 P&L Variance Explained")
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(0, color="black", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures_dir / "walk_forward_variance_explained.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(summary["portfolio"], summary["reconstruction_rmse_usd"], color="#C00000")
    ax.set_ylabel("RMSE ($)")
    ax.set_title("Walk-Forward PC1–PC3 P&L Reconstruction RMSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "walk_forward_reconstruction_rmse.png", dpi=160)
    plt.close(fig)

    factor_columns = [f"pc{factor}_variance_share_pct" for factor in range(1, 8)]
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(factor_summary))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(factor_columns)))
    for factor, (column, color) in enumerate(zip(factor_columns, colors), start=1):
        values = factor_summary[column].to_numpy()
        ax.bar(factor_summary["portfolio"], values, bottom=bottom, color=color, label=f"PC{factor}")
        bottom += values
    ax.set_ylabel("Average factor variance share (%)")
    ax.set_title("Where Each Portfolio's Walk-Forward P&L Variance Sits")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figures_dir / "walk_forward_factor_variance_profile.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for portfolio in summary["portfolio"]:
        frame = block_summary[block_summary["portfolio"] == portfolio]
        ax.plot(frame["test_end"], frame["reconstruction_rmse_usd"], marker=".", linewidth=0.8, label=portfolio)
    ax.set_ylabel("Block reconstruction RMSE ($)")
    ax.set_xlabel("Test block end")
    ax.set_title("Walk-Forward Reconstruction Error by Test Block")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "walk_forward_block_rmse.png", dpi=160)
    plt.close(fig)

