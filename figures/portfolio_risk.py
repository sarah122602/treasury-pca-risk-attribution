
import numpy as np
import pandas as pd

# Maturities in years
tenors = ["2Y", "5Y", "10Y", "30Y"]

# Synthetic key-rate DV01 exposures ($ per bp)
portfolio_a = np.array([25000, 25000, 25000, 25000])
portfolio_b = np.array([50000, -10000, 40000, 20000])

# Illustrative yield-curve scenarios (basis points)
scenarios = pd.DataFrame(
    {
        "Parallel +10bp": [10, 10, 10, 10],
        "Steepening": [-5, 0, 10, 20],
        "Flattening": [15, 10, 0, -5],
    },
    index=tenors,
)

def calculate_pnl(dv01, yield_changes):
    """First-order P&L approximation in USD."""
    return -np.dot(dv01, yield_changes)

print("Portfolio A total DV01:", portfolio_a.sum())
print("Portfolio B total DV01:", portfolio_b.sum())

results = []

for scenario in scenarios.columns:
    changes = scenarios[scenario].to_numpy()

    pnl_a = calculate_pnl(portfolio_a, changes)
    pnl_b = calculate_pnl(portfolio_b, changes)

    results.append(
        {
            "Scenario": scenario,
            "Portfolio A P&L": pnl_a,
            "Portfolio B P&L": pnl_b,
            "Difference": pnl_b - pnl_a,
        }
    )

results_df = pd.DataFrame(results)

print("\nScenario P&L Comparison")
print(results_df.to_string(index=False))
