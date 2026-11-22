"""Run Step 4 walk-forward out-of-sample PCA P&L attribution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.walk_forward_attribution import (  # noqa: E402
    TRAINING_WINDOW,
    TEST_BLOCK,
    run_walk_forward,
)


def main() -> None:
    """Run the rolling-window analysis and print its actual findings."""
    daily, summary, factor_summary, verification, counts = run_walk_forward(
        ROOT / "data" / "treasury_daily_changes_bp.csv",
        ROOT / "results",
        ROOT / "figures",
    )
    if verification["max_daily_add_up_error_usd"].iloc[0] >= 1e-6:
        raise AssertionError("Walk-forward P&L attribution does not reconstruct total P&L")

    print("Treasury Walk-Forward PCA P&L Attribution — Step 4")
    print("=================================================")
    print(f"Training window: {TRAINING_WINDOW} trading days")
    print(f"Non-overlapping test block: {TEST_BLOCK} trading days")
    print(f"Full test blocks: {counts['n_blocks']}; trailing rows not tested: {counts['trailing_rows']}")
    print(f"Test observations: {verification['test_observations_used'].iloc[0]:,}")
    print("\nOut-of-sample results:")
    columns = [
        "portfolio",
        "oos_explained_variance_pct",
        "pc123_variance_share_pct",
        "residual_variance_pct",
        "reconstruction_rmse_usd",
        "max_add_up_error_usd",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nAverage factor variance profile for butterfly diagnosis:")
    butterfly = factor_summary[factor_summary["portfolio"] == "Butterfly_2Y5Y10Y"]
    print(butterfly.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nThe butterfly residual is larger when its 2Y/5Y/10Y DV01 pattern")
    print("has relatively more variance in PCs 4–7 than the other portfolios.")
    print("This is a diagnostic, not evidence of a trading opportunity.")
    print("\nSaved Step 4 CSVs in results/ and charts in figures/.")


if __name__ == "__main__":
    main()

