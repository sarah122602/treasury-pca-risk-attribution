"""Run Step 6 portfolio-specific PCA factor selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.factor_selection import run_factor_selection  # noqa: E402


def main() -> None:
    """Execute Step 6 and print the actual strategy comparison."""
    results = run_factor_selection(
        ROOT / "data" / "treasury_daily_changes_bp.csv",
        ROOT / "results",
        ROOT / "figures",
    )
    print("Portfolio-Specific PCA Factor Selection — Step 6")
    print("================================================")
    print("\nVerification:")
    print(results["verification"].to_string(index=False))
    print("\nStrategy summary:")
    columns = [
        "portfolio",
        "strategy",
        "explained_pnl_variance_pct",
        "rmse_usd",
        "normalized_rmse",
        "oos_r2_vs_training_mean",
        "average_selected_factor_count",
        "median_selected_factor_count",
        "selection_stability_pct",
    ]
    print(results["summary"][columns].to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nButterfly regime comparison:")
    print(results["regime"].query("portfolio == 'Butterfly_2Y5Y10Y'").to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nSaved Step 6 CSVs in results/ and charts in figures/.")


if __name__ == "__main__":
    main()
