"""Covariance-based PCA for Treasury yield changes.

The primary analysis uses daily yield changes in basis points without
standardizing maturities to unit variance. Missing rows are excluded
explicitly; no values are imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


TENORS = ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


@dataclass
class PCAResults:
    """All numerical results needed by Step 2 reporting and output files."""

    dates: pd.Series
    complete_changes: pd.DataFrame
    pca: PCA
    scores: np.ndarray
    covariance: np.ndarray
    reconstruction_errors: pd.DataFrame
    excluded_observations: int
    total_observations: int


def load_daily_changes(path: Path) -> tuple[pd.Series, pd.DataFrame, int, int]:
    """Load and validate daily changes, excluding incomplete rows explicitly."""
    raw = pd.read_csv(path)
    expected_columns = ["Date", *TENORS]
    missing_columns = [column for column in expected_columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    dates = pd.to_datetime(raw["Date"], errors="coerce")
    changes = raw[TENORS].apply(pd.to_numeric, errors="coerce")
    valid = dates.notna() & changes.notna().all(axis=1)
    total_observations = len(raw)
    excluded_observations = int((~valid).sum())
    if valid.sum() < 2:
        raise ValueError("At least two complete daily observations are required for PCA")
    return (
        dates.loc[valid].reset_index(drop=True),
        changes.loc[valid].reset_index(drop=True),
        excluded_observations,
        total_observations,
    )


def fit_pca(changes: pd.DataFrame, dates: pd.Series, excluded: int, total: int) -> PCAResults:
    """Fit covariance-based PCA and calculate diagnostics."""
    x = changes.to_numpy(dtype=float)
    pca = PCA(n_components=len(TENORS), svd_solver="full")
    scores = pca.fit_transform(x)
    centered = x - pca.mean_
    covariance = np.cov(centered, rowvar=False, ddof=1)

    errors = []
    for component_count in range(1, len(TENORS) + 1):
        reduced_scores = np.zeros_like(scores)
        reduced_scores[:, :component_count] = scores[:, :component_count]
        reconstructed = pca.inverse_transform(reduced_scores)
        residual = x - reconstructed
        mse = float(np.mean(residual**2))
        errors.append(
            {
                "components": component_count,
                "mse_bp_squared": mse,
                "rmse_bp": float(np.sqrt(mse)),
            }
        )

    return PCAResults(
        dates=dates,
        complete_changes=changes,
        pca=pca,
        scores=scores,
        covariance=covariance,
        reconstruction_errors=pd.DataFrame(errors),
        excluded_observations=excluded,
        total_observations=total,
    )


def save_results(results: PCAResults, output_dir: Path) -> None:
    """Save PCA numerical outputs and verification metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pca = results.pca
    components = [f"PC{i}" for i in range(1, len(TENORS) + 1)]

    explained = pd.DataFrame(
        {
            "component": components,
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(pca.components_.T, index=TENORS, columns=components)
    loadings.index.name = "maturity"
    loadings.to_csv(output_dir / "pca_loadings.csv")

    scores = pd.DataFrame(results.scores, columns=components)
    scores.insert(0, "Date", results.dates.dt.strftime("%Y-%m-%d"))
    scores.to_csv(output_dir / "pca_factor_scores.csv", index=False)

    covariance = pd.DataFrame(results.covariance, index=TENORS, columns=TENORS)
    covariance.index.name = "maturity"
    covariance.to_csv(output_dir / "pca_covariance_matrix.csv")

    results.reconstruction_errors.to_csv(output_dir / "pca_reconstruction_errors.csv", index=False)

    orthogonality = pca.components_ @ pca.components_.T
    off_diagonal = orthogonality - np.eye(len(TENORS))
    pd.DataFrame(
        [
            {
                "total_observations": results.total_observations,
                "excluded_observations": results.excluded_observations,
                "used_observations": len(results.complete_changes),
                "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
                "max_component_orthogonality_error": float(np.abs(off_diagonal).max()),
            }
        ]
    ).to_csv(output_dir / "pca_verification.csv", index=False)


def save_figures(results: PCAResults, figures_dir: Path) -> None:
    """Create the Step 2 scree, loading, and historical score charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    pca = results.pca
    component_numbers = np.arange(1, len(TENORS) + 1)
    ratios = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratios)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(component_numbers, ratios, color="#4472C4", label="Individual")
    ax.plot(component_numbers, cumulative, marker="o", color="#C00000", label="Cumulative")
    ax.set_xticks(component_numbers)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Treasury Yield-Change PCA Scree Plot")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_scree_plot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index in range(min(3, len(TENORS))):
        ax.plot(TENORS, pca.components_[index], marker="o", label=f"PC{index + 1}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Loading")
    ax.set_title("PCA Loadings Across Treasury Maturities")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_loadings_pc1_pc3.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for index in range(min(3, len(TENORS))):
        ax.plot(results.dates, results.scores[:, index], linewidth=0.8, label=f"PC{index + 1}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Factor score (bp scale)")
    ax.set_title("Historical Treasury PCA Factor Scores")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_factor_scores.png", dpi=160)
    plt.close(fig)


def run_analysis(data_path: Path, results_dir: Path, figures_dir: Path) -> PCAResults:
    """Load data, fit PCA, and write all Step 2 outputs."""
    dates, changes, excluded, total = load_daily_changes(data_path)
    results = fit_pca(changes, dates, excluded, total)
    save_results(results, results_dir)
    save_figures(results, figures_dir)
    return results

