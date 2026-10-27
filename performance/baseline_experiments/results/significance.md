# Paired significance tests

Reference model: **KGP-DC**. Bootstrap resamples per fold: 1000. Seed: 42.

| comparison | metric | mean diff (pp) | folds won | paired t-test p | Wilcoxon p | bootstrap 95% CI (pp) | CI excludes 0 |
|---|---|---:|---:|---:|---:|---:|:--:|
| KGP-DC vs KGNN | AUROC | +4.719 | 5/5 | 4.59e-08 | 0.0625 | [+4.263, +5.099] | yes |
| KGP-DC vs KGNN | AUPRC | +3.359 | 5/5 | 1.58e-07 | 0.0625 | [+2.998, +3.816] | yes |
| KGP-DC vs GraRep | AUROC | -0.221 | 0/5 | 4.30e-04 | 0.0625 | [-0.389, -0.089] | yes |
| KGP-DC vs GraRep | AUPRC | -0.307 | 0/5 | 3.68e-03 | 0.0625 | [-0.585, +0.059] | no |
| KGP-DC vs DeepDDS | AUROC | +4.863 | 5/5 | 7.64e-06 | 0.0625 | [+4.171, +5.738] | yes |
| KGP-DC vs DeepDDS | AUPRC | +5.757 | 5/5 | 5.58e-06 | 0.0625 | [+4.956, +6.873] | yes |
| KGP-DC vs DeepTraSynergy | AUROC | +0.187 | 5/5 | 1.02e-03 | 0.0625 | [+0.047, +0.365] | yes |
| KGP-DC vs DeepTraSynergy | AUPRC | +0.195 | 5/5 | 3.92e-02 | 0.0625 | [-0.155, +0.558] | no |

Notes

* With 5 folds the smallest attainable two-sided Wilcoxon p-value is 2/2^5 = 0.0625; the signed-rank test therefore cannot reject at the 0.05 level and the paired t-test is the primary test.
* The bootstrap interval is the union of the per-fold intervals of the resampled test pairs; it answers whether the metric difference is stable with respect to the finite test sample.
* Differences are reported as reference minus baseline; a positive value means the reference model performs better.
