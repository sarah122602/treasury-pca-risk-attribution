"""Run Step 2 covariance-based Treasury yield-curve PCA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pca_model import TENORS, run_analysis  # noqa: E402


def main() -> None:
    """Execute PCA and print beginner-friendly diagnostics."""
    results = run_analysis(
        ROOT / "data" / "treasury_daily_changes_bp.csv",
        ROOT / "results",
        ROOT / "figures",
    )
    pca = results.pca
    cumulative = pca.explained_variance_ratio_.cumsum()
    component_matrix = pca.components_ @ pca.components_.T
    orthogonality_error = abs(component_matrix - np.eye(len(TENORS))).max()
    if not np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-10):
        raise AssertionError("Explained variance ratios do not sum to 100%")
    if orthogonality_error > 1e-10:
        raise AssertionError("PCA component vectors are not orthogonal")
    if not np.all(np.diff(results.reconstruction_errors["rmse_bp"]) <= 1e-10):
        raise AssertionError("PCA reconstruction RMSE did not decrease monotonically")

    print("Treasury Yield Curve PCA — Step 2")
    print("=================================")
    print(f"Input units: daily yield changes in basis points; maturities: {', '.join(TENORS)}")
    print(f"Date dtype after parsing: {results.dates.dtype}")
    print(f"Maturity dtypes: {', '.join(f'{tenor}={dtype}' for tenor, dtype in results.complete_changes.dtypes.items())}")
    print(f"Total input rows: {results.total_observations:,}")
    print(f"Excluded rows with missing/invalid values: {results.excluded_observations:,}")
    print(f"Rows used in PCA: {len(results.complete_changes):,}")
    print("\nExplained variance:")
    for index, (ratio, total) in enumerate(zip(pca.explained_variance_ratio_, cumulative), start=1):
        print(f"  PC{index}: {ratio:.2%} individually; {total:.2%} cumulative")
    print("\nPCA loading vectors (eigenvectors; signs are arbitrary):")
    for index, loading in enumerate(pca.components_[:3], start=1):
        formatted = ", ".join(f"{tenor}={value:+.4f}" for tenor, value in zip(TENORS, loading))
        print(f"  PC{index}: {formatted}")
    print("\nVerification:")
    print(f"  Explained variance sum: {pca.explained_variance_ratio_.sum():.12f}")
    print(f"  Maximum component orthogonality error: {orthogonality_error:.3e}")
    print("  Reconstruction RMSE by component count (bp):")
    print(results.reconstruction_errors.to_string(index=False))
    print("\nMathematical connection:")
    print("  PCA centers the change matrix, forms its covariance matrix, and finds")
    print("  covariance eigenvectors (directions) and eigenvalues (variance along")
    print("  those directions). sklearn PCA reports the same eigenvalues as")
    print("  explained_variance_, and its component rows are the orthonormal vectors.")
    print("\nSaved numerical outputs to results/ and figures to figures/.")


if __name__ == "__main__":
    main()

