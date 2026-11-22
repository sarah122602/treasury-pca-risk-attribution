# Treasury PCA Risk Attribution

An independent quantitative research study of U.S. Treasury yield-curve risk attribution.

## Research question

Can a low-dimensional PCA model explain the P&L risk of every Treasury portfolio equally well, and does portfolio-specific factor selection provide enough additional out-of-sample coverage to justify greater complexity?

The study uses public U.S. Treasury par yields, daily yield changes, and four synthetic DV01 portfolios. It evaluates historical reconstruction—not returns, trading profitability, or genuine forecasting.

## Key findings

- PC1–PC3 explain **98.91%** of overall seven-tenor yield-change variance, but only **65.86%** of the 2Y/5Y/10Y butterfly’s out-of-sample P&L variance.
- Portfolio-aware 95% selection raises butterfly P&L variance coverage to **98.62%**, using **5.64 factors on average**—a large reconstruction gain with a meaningful complexity cost.
- Portfolio-aware selection is not uniformly superior: simpler three-factor models remain effective for the other portfolios, so the appropriate complexity depends on the portfolio’s exposure pattern. These are historical reconstruction results, not prospective model-selection claims.

### Figure 1 — Portfolio-level PCA coverage gap

![Portfolio-level PCA coverage gap](./figures/showcase_portfolio_coverage_gap.png)

The fixed three-factor baseline covers the bullet, barbell, and steepener well, but leaves a clear portfolio-specific coverage gap for the butterfly.

### Figure 2 — Butterfly risk coverage improvement

![Butterfly risk coverage improvement](./figures/showcase_butterfly_coverage_improvement.png)

Portfolio-aware selection addresses that gap: 95% selection captures 98.62% of butterfly P&L variance with 5.64 average factors, while the 99% rule adds only 1.16 percentage points with 0.66 more factors.

### Figure 3 — Model complexity versus historical risk coverage

![Model complexity versus historical risk coverage](./figures/showcase_complexity_vs_coverage.png)

The small multiples show why factor count is portfolio-dependent: the butterfly has a steep coverage gain from additional factors, while the other portfolios are already near 100% with simpler models.

## Baseline portfolio coverage

| Synthetic portfolio | OOS P&L variance explained | OOS normalized RMSE |
| --- | ---: | ---: |
| 10Y bullet | 98.86% | 0.107 |
| 2Y/30Y barbell | 99.71% | 0.054 |
| 2Y/10Y steepener | 97.65% | 0.153 |
| 2Y/5Y/10Y butterfly | **65.86%** | **0.584** |

“Yield variance explained” is a property of the Treasury yield panel. “Portfolio P&L variance explained” is calculated from out-of-sample actual P&L and PCA reconstruction error. Neither metric is a forecast of future returns.

## Methodology and data

- Public Treasury par yields for 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, and 30Y.
- Covariance PCA on daily changes in basis points; no standardization or imputation.
- Four synthetic DV01 portfolios: bullet, barbell, steepener, and butterfly.
- Rolling **250-day training windows** followed by **non-overlapping 21-day test blocks**.
- PCA fitting, volatility-regime labeling, and factor selection use training data only.
- Step 7 compares fixed PC1–PC3, fixed PC1–PC5, yield-variance thresholds, portfolio-P&L thresholds, and the saturated all-seven reconstruction check.

## Limitations

This is a historical reconstruction study using synthetic exposures and a linear first-order DV01 approximation. It omits carry, roll-down, convexity, transaction costs, liquidity, and position management. PCA factors are sample-dependent; bootstrap intervals preserve within-block dependence but do not eliminate regime persistence, structural breaks, or selection risk. Portfolio-aware thresholds improve attribution in this sample, especially for the butterfly, but do not establish predictive power or investment performance.

Detailed methodology, numerical tables, block-bootstrap intervals, regime analysis, reconciliation checks, and limitations are in [RESEARCH_VALIDATION.md](./RESEARCH_VALIDATION.md).

## Reproducibility and attribution

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/01_explore_data.py --offline
python scripts/02_run_pca.py
python scripts/03_run_pnl_attribution.py
python scripts/04_walk_forward_pca.py
python scripts/05_regime_stability.py
python scripts/05_quality_review.py
python scripts/06_portfolio_factor_selection.py
python scripts/07_complexity_stability.py
python scripts/08_create_showcase_figures.py
~~~

The offline option reuses the checked-in public Treasury dataset without rewriting it. Omit it in Step 1 to refresh data directly from the U.S. Treasury website. Numerical outputs are in [results/](./results/), and research charts are in [figures/](./figures/). The original [figures/portfolio_risk.py](./figures/portfolio_risk.py) example is preserved.

The workflow was inspired by the public [yield-curve-pca repository by GasparCoquet](https://github.com/GasparCoquet/yield-curve-pca). This project’s implementation and portfolio attribution are independent.
