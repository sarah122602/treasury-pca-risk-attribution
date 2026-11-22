# Research Validation, Robustness & Factor Selection

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

Step 6 tests whether the number of factors should be selected for the portfolio being measured rather than from overall Treasury yield variance. It compares fixed PC1–PC3 and fixed PC1–PC5 baselines with four training-only selection rules: the minimum consecutive PC1–PCk set explaining at least 95% or 99% of yield variance, and the minimum consecutive PC1–PCk set explaining at least 95% or 99% of each portfolio's training-window P&L variance. An all-seven-factor model is retained only as a numerical reconstruction check.

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

The README was subsequently updated with the Step 7 executive finding and figure.

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

## 10. Step 7: model complexity, stability, and risk coverage

### Methodology

Step 7 evaluates whether portfolio-specific factor selection provides enough additional out-of-sample P&L coverage to justify added complexity. It uses the same cleaned public Treasury daily-change data, four synthetic DV01 portfolios, 250-day training windows, 21-day non-overlapping test blocks, training-only PCA fitting and selection, and 129 common test blocks as Step 6. The comparison now explicitly includes fixed PC1–PC3 and fixed PC1–PC5, yield-variance 95% and 99%, portfolio-P&L variance 95% and 99%, and all-seven as a saturated numerical check. Selection criteria are training-window quantities; explained variance, RMSE, and normalized RMSE are test-period metrics.

For each portfolio and strategy, Step 7 reports average factor count, out-of-sample explained P&L variance, reconstruction RMSE, normalized RMSE, marginal improvement versus fixed PC1–PC3, improvement per added factor, count distributions, count-change frequency, average absolute count changes, and high/low-volatility differences. Pareto efficiency is defined retrospectively using average factor count and out-of-sample normalized RMSE: a strategy is efficient when no alternative uses fewer or equal factors and has equal or lower normalized RMSE with at least one strict improvement. This is a descriptive frontier, not a test-data optimization claim.

### Verified results

| Portfolio | Strategy | Avg factors | OOS explained variance | Normalized RMSE | Change vs fixed 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| 10Y bullet | Fixed PC1–PC3 | 3.00 | 98.86% | 0.107 | baseline |
| 10Y bullet | Fixed PC1–PC5 | 5.00 | 99.31% | 0.083 | +0.45 pp |
| 10Y bullet | Portfolio 95% | 1.16 | 96.95% | 0.175 | -1.90 pp |
| 10Y bullet | Portfolio 99% | 4.36 | 99.22% | 0.088 | +0.36 pp |
| 2Y/30Y barbell | Fixed PC1–PC3 | 3.00 | 99.71% | 0.054 | baseline |
| 2Y/30Y barbell | Fixed PC1–PC5 | 5.00 | 99.87% | 0.036 | +0.17 pp |
| 2Y/30Y barbell | Portfolio 95% | 1.46 | 97.36% | 0.162 | -2.35 pp |
| 2Y/30Y barbell | Portfolio 99% | 3.02 | 99.71% | 0.054 | +0.00 pp |
| 2Y/10Y steepener | Fixed PC1–PC3 | 3.00 | 97.65% | 0.153 | baseline |
| 2Y/10Y steepener | Fixed PC1–PC5 | 5.00 | 98.49% | 0.123 | +0.84 pp |
| 2Y/10Y steepener | Portfolio 95% | 3.00 | 97.65% | 0.153 | +0.00 pp |
| 2Y/10Y steepener | Portfolio 99% | 6.08 | 99.41% | 0.077 | +1.76 pp |
| 2Y/5Y/10Y butterfly | Fixed PC1–PC3 | 3.00 | 65.86% | 0.584 | baseline |
| 2Y/5Y/10Y butterfly | Fixed PC1–PC5 | 5.00 | 86.19% | 0.372 | +20.33 pp |
| 2Y/5Y/10Y butterfly | Portfolio 95% | 5.64 | 98.62% | 0.117 | +32.77 pp |
| 2Y/5Y/10Y butterfly | Portfolio 99% | 6.30 | 99.79% | 0.046 | +33.93 pp |

The butterfly benefits most from additional factors. Fixed PC1–PC5 adds 20.33 percentage points of explained variance, and portfolio-aware 95% adds 32.77 points with 5.64 factors on average. Portfolio-aware 99% adds only 1.17 further points over portfolio-aware 95% while increasing average complexity to 6.30 factors. This makes portfolio-aware 95% the clearest historical trade-off for the butterfly: it captures almost all of the 99% coverage gain without using nearly all seven factors. For the bullet and barbell, the fixed three-factor model is already highly accurate; portfolio-aware 95% is worse, and portfolio-aware 99% provides only 0.36 and 0.00 percentage points of improvement. The steepener gains 1.76 points at portfolio-aware 99%, but at 6.08 factors, so the operational value of that detail must justify the complexity.

All-seven reconstruction is numerically exact but saturated and is not a practical attribution recommendation. The fixed PC1–PC5 comparison is a meaningful pre-specified complexity benchmark, while yield-99 remains a yield-variance-based adaptive comparator. No strategy is declared universally best solely because it has the highest historical accuracy.

### Stability and volatility regimes

Portfolio-aware 95% count changes between adjacent blocks occurred in 4.7% of transitions for the bullet, 3.9% for the barbell, 0% for the steepener, and 16.4% for the butterfly. Portfolio-aware 99% changed more often for the bullet (24.2%), steepener (19.5%), and butterfly (26.6%), but not the barbell (3.1%). Average absolute count changes under portfolio-aware 95% were 0.05, 0.06, 0.00, and 0.17 factors, respectively; under 99% they were 0.26, 0.03, 0.21, and 0.30.

The high-minus-low volatility difference in average selected count was small for the butterfly (-0.01 factors for both thresholds). It was +0.10 and +0.16 for the bullet at 95% and 99%, +0.19 and +0.03 for the barbell, and 0.00 and -0.18 for the steepener. Therefore, volatile periods do not create a broad, substantial increase in selected factor count in this sample, although the 99% rule is less stable through time for several portfolios. Regime labels remain ex ante training-window classifications; the 129 blocks are not independent regime experiments.

### Uncertainty and verification

The paired block-bootstrap comparisons from Step 6 are carried into [`results/step7_paired_bootstrap_ci.csv`](./results/step7_paired_bootstrap_ci.csv). For the butterfly versus fixed PC1–PC3, portfolio-aware 95% improves explained variance by 32.77 percentage points with a 95% block-bootstrap interval of [29.98, 35.97] and reduces RMSE by about $61,170 with an interval of [$-66,222, $-56,576]. Portfolio-aware 99% improves explained variance by 33.93 points with an interval of [31.17, 37.13] and reduces RMSE by about $70,521 with an interval of [$-75,149, $-66,308]. These intervals preserve within-block dependence but assume exchangeable blocks and do not resolve regime persistence, structural breaks, or selection risk; they are not claims of prospective statistical significance.

The Step 7 verification table confirms identical 2,709 test observations across strategies, exact actual-P&L/reconstruction/residual reconciliation (maximum absolute error 0.0 USD in the saved calculations), all inherited Step 6 training-only/block checks, and all factor counts within PC1–PC7. Outputs are [`results/step7_strategy_comparison.csv`](./results/step7_strategy_comparison.csv), [`results/step7_factor_count_stability.csv`](./results/step7_factor_count_stability.csv), [`results/step7_factor_count_distribution.csv`](./results/step7_factor_count_distribution.csv), [`results/step7_regime_strategy_summary.csv`](./results/step7_regime_strategy_summary.csv), [`results/step7_pareto_strategies.csv`](./results/step7_pareto_strategies.csv), [`results/step7_paired_bootstrap_ci.csv`](./results/step7_paired_bootstrap_ci.csv), and [`results/step7_verification.csv`](./results/step7_verification.csv). Charts are `step7_accuracy_vs_complexity.png`, `step7_marginal_improvement_per_factor.png`, `step7_factor_count_stability.png`, and `step7_complexity_risk_coverage.png` in [`figures/`](./figures/). The implementation is [`src/step7_complexity.py`](./src/step7_complexity.py), run by [`scripts/07_complexity_stability.py`](./scripts/07_complexity_stability.py).

### Practical decision framework

For the bullet, retain fixed PC1–PC3 unless the small historical gain from PC1–PC5 or portfolio-aware 99% is worth added operational complexity. For the barbell, fixed PC1–PC3 is preferred because coverage is already 99.71% and portfolio-aware 99% is effectively tied. For the steepener, fixed PC1–PC3 is the simple stable baseline; portfolio-aware 99% is a defensible higher-detail retrospective alternative, not an automatic choice. For the butterfly, portfolio-aware 95% is the most reasonable historical compromise: fixed PC1–PC3 undercovers risk, fixed PC1–PC5 improves materially but remains incomplete, and portfolio-aware 99% adds relatively little coverage beyond portfolio-aware 95% at higher complexity. These recommendations are conditional on synthetic portfolios, linear DV01 P&L, the seven-tenor par-yield panel, the selected historical sample, and the stated walk-forward design.
