"""Run Step 7 model-complexity, stability, and risk-coverage analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.step7_complexity import run_step7  # noqa: E402


def main() -> None:
    results = run_step7(ROOT / "results", ROOT / "figures")
    print("Treasury PCA Step 7 — Complexity, Stability, and Risk Coverage")
    print("===============================================================")
    print("\nVerification:")
    print(results["verification"].to_string(index=False))
    print("\nStrategy comparison:")
    columns = ["portfolio", "strategy", "average_selected_factor_count", "explained_pnl_variance_pct", "normalized_rmse", "explained_variance_improvement_pp_vs_fixed_3", "explained_pp_per_additional_factor"]
    print(results["summary"][columns].to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nStability:")
    print(results["stability"].to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nPareto-efficient strategies:")
    print(results["pareto"].query("pareto_efficient").to_string(index=False))
    print("\nSaved Step 7 CSVs in results/ and charts in figures/.")


if __name__ == "__main__":
    main()
