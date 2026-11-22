"""Synthetic Treasury portfolio P&L attribution using the Step 2 PCA model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pca_model import TENORS, fit_pca, load_daily_changes


# Positive DV01 means a long-duration exposure: a +1 bp yield move produces
# approximately -DV01 dollars of first-order P&L.
PORTFOLIOS = pd.DataFrame(
    {
        "Bullet_10Y": [0, 0, 0, 0, 100_000, 0, 0],
        "Barbell_2Y_30Y": [50_000, 0, 0, 0, 0, 0, 50_000],
        # Long 2Y / short 10Y: profits when the 2Y cheapens relative to 10Y.
        "Steepener_2Y10Y": [50_000, 0, 0, 0, -50_000, 0, 0],
        # Long wings / short belly, with zero net DV01.
        "Butterfly_2Y5Y10Y": [50_000, 0, -100_000, 0, 50_000, 0, 0],
    },
    index=TENORS,
)
PORTFOLIOS.index.name = "maturity"


def calculate_attribution(
    changes: pd.DataFrame,
    dates: pd.Series,
    pca_results: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate exact daily P&L and its PC1-PC3 plus residual attribution."""
    pca = pca_results.pca
    x = changes.to_numpy(dtype=float)
    scores = pca.transform(x)
    component_changes = [scores[:, i, None] * pca.components_[i] for i in range(3)]
    mean_change = np.asarray(pca.mean_, dtype=float)
    xhat3 = pca.mean_ + sum(component_changes)

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    for portfolio, dv01_series in PORTFOLIOS.items():
        dv01 = dv01_series.to_numpy(dtype=float)
        total_pnl = -(x @ dv01)
        pc_pnl = [-(component @ dv01) for component in component_changes]
        mean_pnl = float(-(mean_change @ dv01))
        reconstructed_pnl = -(xhat3 @ dv01)
        reconstruction_error = total_pnl - reconstructed_pnl
        residual = mean_pnl + reconstruction_error
        pc123 = pc_pnl[0] + pc_pnl[1] + pc_pnl[2]

        portfolio_daily = pd.DataFrame(
            {
                "Date": dates.dt.strftime("%Y-%m-%d"),
                "portfolio": portfolio,
                "total_pnl_usd": total_pnl,
                "pc1_pnl_usd": pc_pnl[0],
                "pc2_pnl_usd": pc_pnl[1],
                "pc3_pnl_usd": pc_pnl[2],
                "residual_pnl_usd": residual,
                "pc123_reconstructed_pnl_usd": reconstructed_pnl,
                "pca3_reconstruction_error_usd": reconstruction_error,
            }
        )
        daily_frames.append(portfolio_daily)

        total_variance = float(np.var(total_pnl, ddof=1))
        pc_variances = [float(np.var(value, ddof=1)) for value in pc_pnl]
        pc123_variance = float(np.var(pc123, ddof=1))
        residual_variance = float(np.var(residual, ddof=1))
        summary_rows.append(
            {
                "portfolio": portfolio,
                "net_dv01_usd_per_bp": float(dv01.sum()),
                "total_pnl_mean_usd": float(np.mean(total_pnl)),
                "total_pnl_std_usd": float(np.std(total_pnl, ddof=1)),
                "total_pnl_variance_usd_squared": total_variance,
                "pc1_variance_usd_squared": pc_variances[0],
                "pc2_variance_usd_squared": pc_variances[1],
                "pc3_variance_usd_squared": pc_variances[2],
                "pc123_variance_usd_squared": pc123_variance,
                "residual_variance_usd_squared": residual_variance,
                "pc123_variance_explained_pct": 100 * pc123_variance / total_variance,
                "residual_variance_pct": 100 * residual_variance / total_variance,
                "pca3_reconstruction_rmse_usd": float(np.sqrt(np.mean(reconstruction_error**2))),
                "max_daily_attribution_add_up_error_usd": float(
                    np.max(np.abs(total_pnl - (pc123 + residual)))
                ),
            }
        )

    return pd.concat(daily_frames, ignore_index=True), pd.DataFrame(summary_rows)


def save_attribution_outputs(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
) -> None:
    """Save Step 3 CSVs and informative attribution charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    PORTFOLIOS.reset_index().to_csv(output_dir / "synthetic_portfolio_dv01.csv", index=False)
    daily.to_csv(output_dir / "daily_pnl_attribution.csv", index=False)
    summary.to_csv(output_dir / "pnl_attribution_summary.csv", index=False)

    components = ["pc1_pnl_usd", "pc2_pnl_usd", "pc3_pnl_usd", "residual_pnl_usd"]
    labels = ["PC1", "PC2", "PC3", "Residual"]
    variance = summary.set_index("portfolio")[[
        "pc1_variance_usd_squared",
        "pc2_variance_usd_squared",
        "pc3_variance_usd_squared",
        "residual_variance_usd_squared",
    ]]
    percentages = variance.div(summary.set_index("portfolio")["total_pnl_variance_usd_squared"], axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(percentages))
    for column, label in zip(percentages.columns, labels):
        values = percentages[column].to_numpy()
        ax.bar(percentages.index, values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylabel("Share of P&L variance (%)")
    ax.set_title("Step 3 P&L Variance Attribution")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pnl_variance_attribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, portfolio in zip(axes.flat, PORTFOLIOS.columns):
        frame = daily[daily["portfolio"] == portfolio]
        ax.plot(pd.to_datetime(frame["Date"]), frame["total_pnl_usd"], color="black", linewidth=0.7, label="Actual")
        ax.plot(
            pd.to_datetime(frame["Date"]),
            frame["pc123_reconstructed_pnl_usd"],
            color="#C00000",
            linewidth=0.7,
            label="PC1-PC3 reconstruction",
        )
        ax.set_title(portfolio)
        ax.set_ylabel("Daily P&L ($)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Actual vs PC1-PC3 Reconstructed Daily P&L", y=1.01)
    fig.tight_layout()
    fig.savefig(figures_dir / "daily_pnl_vs_pca_reconstruction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, portfolio in zip(axes.flat, PORTFOLIOS.columns):
        frame = daily[daily["portfolio"] == portfolio].copy()
        dates = pd.to_datetime(frame["Date"])
        cumulative = frame[components].cumsum()
        for column, label in zip(components, labels):
            ax.plot(dates, cumulative[column], linewidth=0.8, label=label)
        ax.set_title(portfolio)
        ax.set_ylabel("Cumulative P&L ($)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Cumulative P&L Attribution by PCA Factor", y=1.01)
    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_pnl_attribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_attribution(data_path: Path, output_dir: Path, figures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit the existing PCA model and run the in-sample portfolio attribution."""
    dates, changes, excluded, total = load_daily_changes(data_path)
    pca_results = fit_pca(changes, dates, excluded, total)
    daily, summary = calculate_attribution(changes, dates, pca_results)
    save_attribution_outputs(daily, summary, output_dir, figures_dir)
    return daily, summary, pca_results

