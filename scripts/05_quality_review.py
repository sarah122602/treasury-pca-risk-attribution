"""Run the independent Step 5 research-quality and robustness audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_review import run_quality_review  # noqa: E402


def main() -> None:
    """Execute the audit and print its main numerical findings."""
    results = run_quality_review(ROOT)
    print("Treasury PCA Research Quality Review and Robustness Audit")
    print("=========================================================")
    print("\nVerification checks:")
    print(results["verification"].to_string(index=False))
    print("\nFactor-stability comparison:")
    print(results["stability_summary"].to_string(index=False, float_format=lambda value: f"{value:,.6f}"))
    print("\nCommon-sample window robustness for PC1-PC3:")
    print(results["component_summary"].to_string(index=False, float_format=lambda value: f"{value:,.6f}"))
    print("\nBaseline regime metrics:")
    print(results["regime_summary"].query("regime_quantile == 0.5").to_string(index=False, float_format=lambda value: f"{value:,.6f}"))
    print("\nSaved quality-review CSVs in results/ and charts in figures/.")
    print("The numerical outputs are ready for the accompanying RESEARCH_VALIDATION.md report.")


if __name__ == "__main__":
    main()
