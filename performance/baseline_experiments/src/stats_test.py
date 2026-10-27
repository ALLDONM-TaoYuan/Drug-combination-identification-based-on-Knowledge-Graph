"""Paired significance tests for the baseline comparison.

The comparison table in ``results/table.csv`` reports the mean and standard
deviation over the cross-validation folds.  This script adds the statistical
evidence that the table alone cannot carry:

* a paired t-test over the fold-level metric differences,
* a Wilcoxon signed-rank test over the same differences (reported together
  with its resolution limit, because with five folds the smallest attainable
  two-sided p-value is 2/2^5 = 0.0625 and the test can never reject at the
  0.05 level),
* a paired bootstrap confidence interval for the AUROC difference, resampling
  the test pairs of each fold with replacement.

Only the validation split is used for model selection and thresholding; the
test labels are used here exclusively for the final evaluation of already
fixed models.

Usage
-----
    python src/stats_test.py
    python src/stats_test.py --bootstrap 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common

METRICS: tuple[str, ...] = ("AUROC", "AUPRC")
REFERENCE: str = "KGP-DC"
OUTPUT_CSV: Path = common.RESULTS_DIR / "significance.csv"
OUTPUT_MD: Path = common.RESULTS_DIR / "significance.md"


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC; identical to sklearn.metrics.roc_auc_score."""
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-interpolated average precision.

    Reproduces sklearn.metrics.average_precision_score, including its
    handling of tied scores: the curve is evaluated once per distinct score
    value, exactly as ``precision_recall_curve`` does.  This matters for the
    bootstrap, where resampling with replacement creates many tied scores.
    """
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    # last index of every run of equal scores -> count of samples >= threshold
    distinct = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    true_positive = np.cumsum(y)[distinct]
    false_positive = np.cumsum(1 - y)[distinct]
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = true_positive / n_pos
    # sklearn orders the curve from the lowest threshold upwards, i.e. recall
    # decreasing and precision increasing, and appends the (1, 0) end point.
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def paired_bootstrap(labels: np.ndarray, score_a: np.ndarray,
                     score_b: np.ndarray, metric: str, n_boot: int,
                     rng: np.random.RandomState) -> tuple[float, float, float]:
    """Paired bootstrap over test pairs; returns (mean, low, high) of a - b."""
    scorer = fast_auc if metric == "AUROC" else fast_ap
    n = len(labels)
    diffs = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.randint(0, n, n)
        y = labels[sample]
        if len(np.unique(y)) < 2:
            diffs[index] = np.nan
            continue
        diffs[index] = scorer(y, score_a[sample]) - scorer(y, score_b[sample])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap resamples per fold (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", default=REFERENCE,
                        help=f"model the others are compared against (default: {REFERENCE})")
    args = parser.parse_args()

    from scipy.stats import ttest_rel, wilcoxon

    records = common.load_run_records()
    folds = sorted(set(int(f) for f in records["fold"]))
    seeds = sorted(set(int(s) for s in records["seed"]))
    print(f"records: {len(records)}  folds: {folds}  seeds: {seeds}")
    if len(folds) < 2:
        raise SystemExit("need at least two folds for a paired test")

    rng = np.random.RandomState(args.seed)
    rows: list[dict] = []
    lines: list[str] = []

    for seed in seeds:
        for other in [key for key in common.MODEL_KEYS if key != args.reference]:
            for metric in METRICS:
                reference_values, other_values = [], []
                per_fold_bootstrap = []
                for fold in folds:
                    ref = common.load_predictions(args.reference, fold, seed)
                    cmp_ = common.load_predictions(other, fold, seed)
                    if not np.array_equal(ref["true_label"].to_numpy(),
                                          cmp_["true_label"].to_numpy()):
                        raise SystemExit(f"fold {fold}: prediction files disagree on labels")
                    labels = ref["true_label"].to_numpy(dtype=np.int32)
                    ref_score = ref["predicted_probability"].to_numpy(dtype=np.float64)
                    cmp_score = cmp_["predicted_probability"].to_numpy(dtype=np.float64)
                    scorer = fast_auc if metric == "AUROC" else fast_ap
                    reference_values.append(scorer(labels, ref_score))
                    other_values.append(scorer(labels, cmp_score))
                    mean, low, high = paired_bootstrap(
                        labels, ref_score, cmp_score, metric, args.bootstrap, rng)
                    per_fold_bootstrap.append((fold, mean, low, high))

                reference_values = np.asarray(reference_values)
                other_values = np.asarray(other_values)
                difference = reference_values - other_values
                t_stat, p_ttest = ttest_rel(reference_values, other_values)
                try:
                    p_wilcoxon = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_wilcoxon = float("nan")
                row = {
                    "seed": seed, "comparison": f"{args.reference} vs {other}",
                    "metric": metric,
                    "reference_mean": float(reference_values.mean()),
                    "baseline_mean": float(other_values.mean()),
                    "mean_diff_pp": float(difference.mean() * 100),
                    "std_diff_pp": float(difference.std(ddof=1) * 100),
                    "folds_won": int((difference > 0).sum()),
                    "folds_total": int(len(difference)),
                    "t_stat": float(t_stat), "p_paired_ttest": float(p_ttest),
                    "p_wilcoxon": p_wilcoxon,
                    "wilcoxon_resolution_limit": float(2.0 / (2 ** len(difference))),
                    "bootstrap_mean_pp": float(np.mean([b[1] for b in per_fold_bootstrap]) * 100)
                    if per_fold_bootstrap else float("nan"),
                    "bootstrap_ci_low_pp": float(np.min([b[2] for b in per_fold_bootstrap]) * 100)
                    if per_fold_bootstrap else float("nan"),
                    "bootstrap_ci_high_pp": float(np.max([b[3] for b in per_fold_bootstrap]) * 100)
                    if per_fold_bootstrap else float("nan"),
                }
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_CSV, index=False)

    lines.append("# Paired significance tests")
    lines.append("")
    lines.append(f"Reference model: **{args.reference}**. Bootstrap resamples per fold: "
                 f"{args.bootstrap}. Seed: {args.seed}.")
    lines.append("")
    lines.append("| comparison | metric | mean diff (pp) | folds won | paired t-test p | "
                 "Wilcoxon p | bootstrap 95% CI (pp) | CI excludes 0 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|:--:|")
    for _, row in frame.iterrows():
        excludes = "yes" if (row["bootstrap_ci_low_pp"] > 0 or row["bootstrap_ci_high_pp"] < 0) else "no"
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_diff_pp']:+.3f} | "
            f"{row['folds_won']}/{row['folds_total']} | {row['p_paired_ttest']:.2e} | "
            f"{row['p_wilcoxon']:.4f} | "
            f"[{row['bootstrap_ci_low_pp']:+.3f}, {row['bootstrap_ci_high_pp']:+.3f}] | {excludes} |")
    lines.append("")
    lines.append("Notes")
    lines.append("")
    lines.append(f"* With {len(folds)} folds the smallest attainable two-sided Wilcoxon "
                 f"p-value is 2/2^{len(folds)} = {frame['wilcoxon_resolution_limit'].iloc[0]:.4f}; "
                 "the signed-rank test therefore cannot reject at the 0.05 level and the "
                 "paired t-test is the primary test.")
    lines.append("* The bootstrap interval is the union of the per-fold intervals of the "
                 "resampled test pairs; it answers whether the metric difference is stable "
                 "with respect to the finite test sample.")
    lines.append("* Differences are reported as reference minus baseline; a positive value "
                 "means the reference model performs better.")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(frame.to_string(index=False))
    print(f"\n-> {OUTPUT_CSV}")
    print(f"-> {OUTPUT_MD}")


if __name__ == "__main__":
    main()
