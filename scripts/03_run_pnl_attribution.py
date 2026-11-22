"""Run Step 3 synthetic Treasury portfolio P&L attribution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.portfolio_attribution import PORTFOLIOS, run_attribution  # noqa: E402


def main() -> None:
    """Execute in-sample P&L attribution and print the research summary."""
    daily, summary, pca_results = run_attribution(
        ROOT / "data" / "treasury_daily_changes_bp.csv",
        ROOT / "results",
        ROOT / "figures",
    )
    if not np.all(summary["max_daily_attribution_add_up_error_usd"] < 1e-6):
        raise AssertionError("Daily P&L attribution does not add up to total P&L")

    print("Synthetic Treasury Portfolio P&L Attribution — Step 3")
    print("=====================================================")
    print(f"Rows used: {pca_results.total_observations - pca_results.excluded_observations:,}; excluded rows: {pca_results.excluded_observations:,}")
    print("P&L convention: -sum(DV01 in USD/bp × yield change in bp)")
    print("\nPortfolio DV01 definitions (USD per bp):")
    print(PORTFOLIOS.to_string())
    print("\nVariance attribution summary:")
    columns = [
        "portfolio",
        "net_dv01_usd_per_bp",
        "total_pnl_std_usd",
        "pc123_variance_explained_pct",
        "residual_variance_pct",
        "pca3_reconstruction_rmse_usd",
        "max_daily_attribution_add_up_error_usd",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nInterpretation: residual includes the sample-mean P&L plus PC4-PC7 and")
    print("the PC1-PC3 reconstruction error. It is not a hedge or forecast result.")
    print("\nSaved Step 3 CSVs in results/ and charts in figures/.")


if __name__ == "__main__":
    main()

