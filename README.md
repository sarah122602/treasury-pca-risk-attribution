# Treasury PCA Risk Attribution

This project analyzes official U.S. Treasury par yields across 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, and 30Y maturities. Step 1 downloads and explores the data. Step 2 models daily yield changes with covariance-based principal component analysis (PCA). Portfolio hedging, regression, and out-of-sample testing are not yet implemented.

## Data source

The exploration script downloads the public **Daily Treasury Par Yield Curve Rates** CSV published by the U.S. Department of the Treasury. It requests each calendar year from 2015 through the current year, then combines the responses and keeps these maturities:

- 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, and 30Y

No proprietary data or copied external source code is used. Missing observations are retained as missing; the script does not forward-fill, backward-fill, interpolate, or otherwise silently impute values.

## Setup and run

```bash
python -m pip install -r requirements.txt
python scripts/01_explore_data.py
```

The script writes:

- `data/treasury_par_yields.csv`: cleaned daily yields in percent.
- `data/treasury_daily_changes_bp.csv`: first differences in basis points.
- `figures/treasury_yield_history.png`: historical yield chart.
- `figures/treasury_yield_correlation.png`: correlation heatmap of daily changes.

The pre-existing `figures/portfolio_risk.py` is preserved and is not used by this Step 1 pipeline.

## Step 2: yield-curve PCA

Run:

```bash
python scripts/02_run_pca.py
```

The analysis reads daily yield changes in basis points. It excludes rows with an invalid date or any missing maturity and prints the exclusion count. The primary analysis is covariance-based: maturities are centered but are not standardized to unit variance. This preserves the fact that a maturity with larger variance contributes more to the covariance structure.

For a centered change matrix `X`, the sample covariance matrix is `C = X'X / (n - 1)`. PCA finds eigenvectors `v` and eigenvalues `lambda` such that `C v = lambda v`. Each eigenvector is a maturity loading pattern, while its eigenvalue is the variance explained in that direction. The explained variance ratio is `lambda_i / sum(lambda)`. The factor scores are the daily centered observations projected onto those eigenvectors.

PCA component signs are arbitrary: multiplying an entire loading vector and its corresponding score series by -1 produces the same reconstruction and variance. Interpret loading shapes and relative signs, not an isolated sign convention. PC1–PC3 are not assumed to be level, slope, or curvature; inspect `results/pca_loadings.csv` and the loading chart before assigning economic names.

Step 2 writes these numerical results:

- `results/pca_explained_variance.csv`
- `results/pca_loadings.csv`
- `results/pca_factor_scores.csv`
- `results/pca_covariance_matrix.csv`
- `results/pca_reconstruction_errors.csv`
- `results/pca_verification.csv`

It also writes `figures/pca_scree_plot.png`, `figures/pca_loadings_pc1_pc3.png`, and `figures/pca_factor_scores.png`.

The pre-existing `figures/portfolio_risk.py` is preserved and is unrelated to the PCA analysis.

