"""Run Step 5 rolling PCA stability and market-regime analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regime_stability import run_stability_analysis  # noqa: E402


def main() -> None:
    """Run Step 5 and print the verified numerical findings."""
    stability, regime_summary, portfolio_regime, verification = run_stability_analysis(
        ROOT / "data" / "treasury_daily_changes_bp.csv",
        ROOT / "results",
        ROOT / "figures",
    )
    max_error = float(verification["max_daily_add_up_error_usd"].iloc[0])
    if max_error >= 1e-6:
        raise AssertionError("Step 5 P&L attribution does not reconstruct total P&L")

    print("Treasury PCA Factor Stability and Market Regime Analysis — Step 5")
    print("===============================================================")
    print(verification.to_string(index=False))
    print("\nRegime-level PCA diagnostics:")
    print(regime_summary.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nPortfolio attribution by regime:")
    print(portfolio_regime.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nOverall loading-stability means:")
    print(
        stability[["pc1_loading_similarity", "pc2_loading_similarity", "pc3_loading_similarity", "subspace_similarity_pc123"]]
        .mean()
        .to_string(float_format=lambda value: f"{value:,.4f}")
    )
    print("\nSaved Step 5 CSVs in results/ and charts in figures/.")


if __name__ == "__main__":
    main()
