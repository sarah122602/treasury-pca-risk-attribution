# Step 5 Research Quality Review and Robustness Audit

## Scope and conclusion

This report independently audits Steps 1–5 using the cleaned Treasury files and fresh rolling PCA fits. It does not modify the main research methodology, prior outputs, or the original portfolio example.

Overall assessment: the core calculations are reproducible and internally consistent. The strongest portfolio-level conclusion is supported: the 2Y/5Y/10Y butterfly is much harder to reconstruct with PC1–PC3 than the bullet, barbell, or steepener, and its omitted PC4–PC7 exposure remains high out of sample. The original stability conclusion requires qualification: adjacent overlapping windows produce similarity near 0.999, but non-overlapping and widely separated windows show materially lower similarity. The adjacent-window result should not be presented as evidence that the factor structure is globally stable across the entire sample.

## 1. Audit findings

The audit independently verified:

- 2,973 cleaned yield rows and 2,972 usable daily-change rows.
- Yield dates and change dates align exactly, are unique, and are sorted.
- The yield panel has no missing maturity observations; the seven first-difference values in the initial change row are missing by construction.
- Daily changes equal 100 times the first differences of the yield CSV, with maximum absolute discrepancy 0 bp.
- Yields are stored in percent and changes in basis points.
- Walk-forward tests use 129 non-overlapping 21-day blocks. Each training window ends exactly at the test start.
- Training PCA components are orthogonal; the largest orthogonality error is `1.78e-15`.
- Seven-factor P&L reconstruction error is at most `2.33e-9` USD.
- PC1–PC3 plus the training-mean P&L and residual reconcile to actual P&L with maximum error `6.98e-10` USD.
- Regime labels are assigned from the preceding training window only.

Detailed checks are in [`results/quality_review_verification.csv`](./results/quality_review_verification.csv) and [`results/quality_review_data_checks.csv`](./results/quality_review_data_checks.csv). The independent implementation is [`src/quality_review.py`](./src/quality_review.py), run by [`scripts/05_quality_review.py`](./scripts/05_quality_review.py).

No calculation error was found in the audited PCA projection, P&L sign convention, reconstruction, or baseline regime classification. The main issue is interpretive: adjacent rolling windows overlap heavily and therefore overstate how much independent evidence supports global factor stability.

## 2. Factor stability and overlapping-window bias

The baseline Step 5 value of mean PC1–PC3 subspace similarity, 0.9991, is reproduced for adjacent rolling windows. Those windows share 229 of 250 observations, so this statistic is expected to be high and is not an independent-period stability test.

| Comparison | Pairs | PC1 loading similarity | PC2 loading similarity | PC3 loading similarity | PC1–PC3 subspace similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adjacent, overlapping | 128 | 0.9997 | 0.9990 | 0.9967 | 0.9991 |
| Non-overlapping 250-day windows | 10 | 0.9861 | 0.9795 | 0.9490 | 0.9848 |
| Widely separated windows | 2 | 0.9218 | 0.8294 | 0.7902 | 0.9485 |

Similarity is sign- and permutation-invariant. No PC1–PC3 component switches were detected in these comparisons, but PC3 is clearly less stable than PC1. The result supports local continuity of the PCA basis, not unconditional stability across history. The widest-period comparisons are based on only two pairs and should be treated as descriptive rather than precise estimates.

## 3. Training-window robustness

For a fair comparison, 125-, 250-, and 500-day windows were evaluated on the same 2,457 test observations beginning after the 500-day burn-in. The table reports PC1–PC3 attribution.

| Window | Portfolio | Explained P&L variance | RMSE ($) | Normalized RMSE | OOS R² vs training-mean benchmark |
| ---: | --- | ---: | ---: | ---: | ---: |
| 125 | 10Y bullet | 98.86% | 57,202 | 0.107 | 0.989 |
| 250 | 10Y bullet | 98.89% | 56,467 | 0.105 | 0.989 |
| 500 | 10Y bullet | 98.89% | 56,527 | 0.105 | 0.989 |
| 125 | 2Y/30Y barbell | 99.70% | 25,618 | 0.055 | 0.997 |
| 250 | 2Y/30Y barbell | 99.71% | 25,183 | 0.054 | 0.997 |
| 500 | 2Y/30Y barbell | 99.72% | 24,764 | 0.053 | 0.997 |
| 125 | 2Y/10Y steepener | 97.55% | 28,818 | 0.157 | 0.976 |
| 250 | 2Y/10Y steepener | 97.72% | 27,799 | 0.151 | 0.977 |
| 500 | 2Y/10Y steepener | 97.69% | 27,970 | 0.152 | 0.977 |
| 125 | 2Y/5Y/10Y butterfly | 64.21% | 78,786 | 0.598 | 0.643 |
| 250 | 2Y/5Y/10Y butterfly | 65.65% | 77,193 | 0.586 | 0.657 |
| 500 | 2Y/5Y/10Y butterfly | 66.15% | 76,631 | 0.582 | 0.662 |

The conclusion is not driven by a single selected window length. The butterfly remains the least well reconstructed under all three pre-specified windows. Differences are modest, so there is no evidence here to optimize the window length using test results.

## 4. Butterfly higher-order-factor investigation

Using the full Step 4 test sample and a 250-day training window:

| Retained components | Explained P&L variance | RMSE ($) | OOS R² vs training mean |
| ---: | ---: | ---: | ---: |
| PC1 | 15.57% | 120,346 | 0.157 |
| PC1–PC2 | 15.79% | 120,194 | 0.160 |
| PC1–PC3 | 65.86% | 76,528 | 0.659 |
| PC1–PC4 | 70.37% | 71,290 | 0.704 |
| PC1–PC5 | 86.19% | 48,668 | 0.862 |
| PC1–PC7 | 100.00% | approximately 0 | 1.000 |

PC3 is the main improvement for the butterfly, but PC4 and especially PC5 add material explanatory power. The all-seven result is a numerical reconstruction check, not a useful low-dimensional model: seven components span the seven-tenor input space.

The result is not caused by a small number of extreme days. Removing the largest 1% of absolute butterfly P&L observations—28 days—changes PC1–PC3 explained variance from 65.86% to 63.12%. The conclusion therefore persists, although the exact percentage is sample-sensitive.

## 5. Volatility-regime reassessment

The baseline regime uses the existing median threshold: high volatility when the trailing 21-day curve RMS is at least the median of all available trailing 21-day RMS values inside the preceding 250-day training window. This produces 61 high-volatility and 68 low-volatility test blocks.

| Portfolio | Regime | Explained variance | RMSE ($) | Normalized RMSE | OOS R² | Mean residual ($) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bullet | High | 98.82% | 58,545 | 0.109 | 0.988 | -1,171 |
| Bullet | Low | 98.89% | 54,659 | 0.105 | 0.989 | 686 |
| Barbell | High | 99.69% | 25,902 | 0.055 | 0.997 | -98 |
| Barbell | Low | 99.72% | 23,981 | 0.053 | 0.997 | 351 |
| Steepener | High | 97.28% | 28,961 | 0.165 | 0.973 | 252 |
| Steepener | Low | 97.94% | 26,438 | 0.143 | 0.980 | -267 |
| Butterfly | High | 63.82% | 80,315 | 0.601 | 0.639 | -2,071 |
| Butterfly | Low | 67.82% | 72,964 | 0.567 | 0.679 | 666 |

Absolute errors are larger in high volatility, partly because actual P&L volatility is larger. Normalized RMSE also deteriorates for the steepener and butterfly, so the high-volatility difference is not only a scale effect. The butterfly remains the clear outlier in both regimes.

As a sensitivity check, a pre-specified 75th-percentile training-window threshold produces 36 high-volatility and 93 low-volatility blocks. Butterfly explained variance is 60.15% in the high group versus 68.09% in the low group; normalized RMSE is 0.632 versus 0.565. The direction is consistent with the baseline regime rule, though the alternative high-volatility sample is smaller.

## 6. Time-series-aware uncertainty

A 2,000-repetition bootstrap resampled complete, non-overlapping 21-day test blocks rather than individual daily observations. This preserves within-block dependence but assumes blocks are exchangeable, so the intervals do not account for all possible regime persistence or structural breaks.

| Quantity | Estimate | 95% block-bootstrap interval |
| --- | ---: | ---: |
| Bullet explained variance | 98.86% | [98.72%, 98.97%] |
| Barbell explained variance | 99.71% | [99.65%, 99.75%] |
| Steepener explained variance | 97.65% | [97.21%, 98.05%] |
| Butterfly explained variance | 65.86% | [62.68%, 68.65%] |
| Other-three mean minus butterfly explained variance | 32.88 pp | [30.08 pp, 36.05 pp] |
| Butterfly RMSE minus other-three mean | $40,166 | [$37,088, $43,446] |

The uncertainty analysis supports a large butterfly gap under this block-bootstrap design. It does not turn the historical reconstruction into a forecast or establish causal mechanisms.

## 7. Step 6: portfolio-specific factor selection

### Research question and design

Step 6 tests whether the number of factors should be selected for the portfolio being measured rather than from overall Treasury yield variance. It compares a fixed three-factor baseline with four training-only selection rules: the minimum consecutive PC1–PCk set explaining at least 95% or 99% of yield variance, and the minimum consecutive PC1–PCk set explaining at least 95% or 99% of each portfolio's training-window P&L variance. An all-seven-factor model is retained only as a numerical reconstruction check.

Every method uses the same 129 non-overlapping 21-trading-day test blocks, a 250-trading-day rolling training window, training-window means, and PCA fits estimated before the corresponding test block. Thus, factor selection does not use test-period observations. The analysis covers the four hypothetical DV01 portfolios used in Steps 3–5. The complete outputs are [`results/step6_strategy_summary.csv`](./results/step6_strategy_summary.csv), [`results/step6_training_factor_selection.csv`](./results/step6_training_factor_selection.csv), [`results/step6_factor_count_distribution.csv`](./results/step6_factor_count_distribution.csv), [`results/step6_regime_strategy_summary.csv`](./results/step6_regime_strategy_summary.csv), [`results/step6_paired_bootstrap_ci.csv`](./results/step6_paired_bootstrap_ci.csv), and [`results/step6_verification.csv`](./results/step6_verification.csv).

### Verified out-of-sample comparison

The table reports explained portfolio P&L variance, reconstruction RMSE, and average factor count. Explained P&L variance is calculated from test-period actual P&L and reconstruction error; it is not the PCA yield-variance ratio.

| Portfolio | Strategy | Explained P&L variance | RMSE ($) | Average factors |
| --- | --- | ---: | ---: | ---: |
| 10Y bullet | Fixed 3 | 98.86% | 56,530 | 3.00 |
| 10Y bullet | Yield ≥95% | 98.52% | 64,268 | 2.00 |
| 10Y bullet | Yield ≥99% | 99.04% | 51,694 | 3.93 |
| 10Y bullet | Portfolio P&L ≥95% | 96.95% | 92,288 | 1.16 |
| 10Y bullet | Portfolio P&L ≥99% | 99.22% | 46,687 | 4.36 |
| 2Y/30Y barbell | Fixed 3 | 99.71% | 24,907 | 3.00 |
| 2Y/30Y barbell | Yield ≥95% | 96.58% | 84,827 | 2.00 |
| 2Y/30Y barbell | Yield ≥99% | 99.79% | 21,270 | 3.93 |
| 2Y/30Y barbell | Portfolio P&L ≥95% | 97.36% | 74,584 | 1.46 |
| 2Y/30Y barbell | Portfolio P&L ≥99% | 99.71% | 24,800 | 3.02 |
| 2Y/10Y steepener | Fixed 3 | 97.65% | 27,660 | 3.00 |
| 2Y/10Y steepener | Yield ≥95% | 83.66% | 72,958 | 2.00 |
| 2Y/10Y steepener | Yield ≥99% | 97.99% | 25,565 | 3.93 |
| 2Y/10Y steepener | Portfolio P&L ≥95% | 97.65% | 27,660 | 3.00 |
| 2Y/10Y steepener | Portfolio P&L ≥99% | 99.41% | 13,845 | 6.08 |
| 2Y/5Y/10Y butterfly | Fixed 3 | 65.86% | 76,528 | 3.00 |
| 2Y/5Y/10Y butterfly | Yield ≥95% | 15.79% | 120,194 | 2.00 |
| 2Y/5Y/10Y butterfly | Yield ≥99% | 72.36% | 68,863 | 3.93 |
| 2Y/5Y/10Y butterfly | Portfolio P&L ≥95% | 98.62% | 15,358 | 5.64 |
| 2Y/5Y/10Y butterfly | Portfolio P&L ≥99% | 99.79% | 6,007 | 6.30 |

The butterfly is the clearest case where yield-variance selection underrepresents portfolio risk. The 95% yield rule selects two factors in every window and explains only 15.79% of butterfly P&L variance. The portfolio-aware rules select five to seven factors on average and raise explained P&L variance to 98.62% and 99.79%. This is a portfolio-specific reconstruction result, not evidence of forecasting or trading profitability.

The comparison is not a universal superiority claim. Portfolio-aware 95% selection is worse than fixed three factors for the bullet (96.95% versus 98.86%) and barbell (97.36% versus 99.71%), and is effectively the same as fixed three factors for the steepener. Even when portfolio-aware selection improves reconstruction, it generally does so by adding complexity: the butterfly averages 5.64 factors at the 95% threshold and 6.30 at 99%, versus three fixed factors. The all-seven model is an identity-span check, not a parsimonious attribution model.

### Regimes and uncertainty

The portfolio-aware improvement for the butterfly appears in both ex-ante volatility groups. Under the existing training-window regime rule, portfolio-aware 95% explains 98.48% of high-volatility and 98.76% of low-volatility butterfly P&L variance; portfolio-aware 99% explains 99.75% and 99.83%, respectively. Its normalized RMSE is also lower than fixed three factors in both groups. These are descriptive comparisons, not causal evidence that the selection rule causes better future attribution.

Paired block-bootstrap intervals used 2,000 repetitions, a fixed seed of 20261122, and complete non-overlapping 21-day test blocks. For the butterfly relative to fixed three factors, portfolio-aware 95% reduced RMSE by $61,170 (95% interval [$-66,222, $-56,576]) and increased explained variance by 32.77 percentage points ([$29.98, $35.97]). Portfolio-aware 99% reduced RMSE by $70,521 ([$-75,149, $-66,308]) and increased explained variance by 33.93 points ([$31.17, $37.13]). The intervals support a large paired difference for this portfolio under this resampling design, but the method assumes test blocks are exchangeable and does not resolve regime persistence, structural breaks, or selection risk.

### Interpretation and limitations

Step 6 supports a narrower conclusion than “more factors are better”: yield-curve variance thresholds can be poorly aligned with the P&L risk of a portfolio whose DV01 profile emphasizes higher-order curve shapes. Portfolio-aware selection can address that mismatch in this sample, especially for the butterfly, but it selects more factors and is not uniformly better across portfolios. Factor counts are themselves sample-dependent, and choosing factors using training P&L does not establish prospective forecasting ability. The results remain conditional on synthetic DV01 exposures, linear first-order P&L, the seven-tenor par-yield panel, the 250-day window, and the 21-day block design.

## 8. Findings that qualify the existing README

The existing README’s portfolio attribution findings are supported: the butterfly has materially lower PC1–PC3 explained P&L variance and higher residual exposure. The Step 5 regime conclusion is also supported, with the additional observation that normalized error deteriorates in high volatility.

The statement that PCA is “highly stable” needs narrower wording. The reported 0.9991 similarity is the adjacent-overlapping-window statistic. Independent-window comparisons are lower: 0.9848 for non-overlapping windows and 0.9485 for the two widely separated comparisons. The exact README correction should be:

> Adjacent rolling PCA windows are very similar, partly because neighboring 250-day windows share 229 observations. Non-overlapping and widely separated windows show lower PC1–PC3 similarity, especially for PC3, so the factor basis is locally persistent but not globally invariant.

No README changes were made during this audit.

## 9. Reproducibility and output map

Run:

```bash
python scripts/05_quality_review.py
```

The audit writes:

- [`quality_review_verification.csv`](./results/quality_review_verification.csv)
- [`quality_review_data_checks.csv`](./results/quality_review_data_checks.csv)
- [`quality_review_factor_stability.csv`](./results/quality_review_factor_stability.csv)
- [`quality_review_factor_stability_summary.csv`](./results/quality_review_factor_stability_summary.csv)
- [`quality_review_window_robustness.csv`](./results/quality_review_window_robustness.csv)
- [`quality_review_regime_metrics.csv`](./results/quality_review_regime_metrics.csv)
- [`quality_review_extreme_days.csv`](./results/quality_review_extreme_days.csv)
- [`quality_review_bootstrap_ci.csv`](./results/quality_review_bootstrap_ci.csv)

Charts are prefixed `quality_review_` in [`figures/`](./figures/). The audit implementation is [`src/quality_review.py`](./src/quality_review.py). No files were committed or pushed.
