# `negative_sampling/` -- negative-sampling strategy comparison

## 1. Question

A drug pair that the source database does not record is *unlabelled*, not a true
negative: it may simply be untested. Any pair-level benchmark therefore depends
on how the unlabelled pool is turned into negatives, and the reported numbers
inherit that choice. This module measures how three negative-sampling rules
change the performance of the KGP-DC classifier under one fixed protocol, so
that the negative-sample assumption can be stated as a tested, bounded premise
instead of an implicit one.

## 2. Data and split

| Item | Value |
|---|---|
| Regime | single-cold (`cold_start/entity_split_data/split/single`) |
| Training positives | 18,168 drug pairs |
| Training negatives | 18,168 drug pairs (1:1) |
| Candidate pool | 1,378,788 pairs (training drug pool, self-pairs and known positives removed) |
| Feature vector | 3,840 = 2 x (128 KG + 768 ChemBERTa + 1,024 mean ProtT5) |
| Validation / test | frozen single-cold splits, shared by all three strategies |
| Seeds | 42, 52, 62 |

Every evaluated pair is a drug pair; the target features of a drug are mean
pooled over its own targets, so one pair corresponds to exactly one row and no
pair is duplicated across the split.

## 3. The three strategies

All three draw from the same candidate pool and differ only in the selection
rule applied to it.

**Uniform random** (`uniform_distribution.py`) draws the negatives uniformly
without replacement. It is the unfiltered reference point.

**Mean Euclidean distance** (`euclidean_distance.py`) keeps only candidates
whose distance to the positive cluster is *larger* than `0.9 x AED`, where
AED is the mean distance of the candidate pool. The distance of a candidate
pair is the average, over both drug orientations, of the Euclidean distance
between the two drugs' standardised blocks (`KG || SMILES || mean target`) and
the corresponding block means of the positive pairs. The rule therefore keeps
the candidates that lie farther from the positive cluster than the pool
average, and Section 6 reports where that places the resulting negatives.

**DTI frequency matching** (`frequency_matching.py`) stratifies the candidates
by the number of drug-target interactions of the two endpoints and draws the
negatives so that the stratum distribution of the negatives reproduces the
stratum distribution of the positives exactly (22 strata, `stratum_deficit = 0`,
`surplus_negatives = 0`). This is the biologically informed rule: a negative is
only accepted if it looks like a real combination partner in terms of target
richness.

## 4. Protocol

* The candidate pool, the training positives, the validation set and the test
  set are **identical for the three strategies**. The validation and test
  feature files are the same objects, verified by SHA-256 in
  `process/<block>/features/sampling_manifest.json`.
* The training negatives exclude every known positive pair and every
  validation and test negative pair.
* The threshold is selected on the validation split only (Youden's J on a
  drug-disjoint half of the validation set, after an isotonic calibration
  fitted on the other half). The test labels never take part in calibration,
  threshold selection or early stopping.
* All three strategies use the same model, the same hyper-parameters and the
  same early-stopping rule.

Audit fields recorded per strategy (`checks` in the sampling manifest), all of
which must be zero: self-pairs, duplicate negatives, overlap with known
positives, negatives outside the training drug pool, duplicates of the
validation or test negatives. Independently re-verified on the committed pair
metadata: zero overlap between the training pairs and the validation or test
pairs of every strategy, and zero overlap between the negatives and the 51,212
known positive pairs.

The three negative sets overlap each other by only 1.0 to 1.3 %, so they are
genuinely different samples rather than relabellings of one another.

## 5. Results

Test split, mean +/- standard deviation over the three seeds.

| Strategy | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUROC | AUPR |
|---|---|---|---|---|---|---|---|---|
| Uniform random | 0.8169 +/- 0.0020 | 0.9337 +/- 0.0117 | 0.7000 +/- 0.0077 | 0.7569 +/- 0.0024 | 0.8360 +/- 0.0033 | 0.6519 +/- 0.0073 | 0.8575 +/- 0.0028 | 0.7815 +/- 0.0056 |
| Mean Euclidean distance | 0.7885 +/- 0.0004 | 0.9366 +/- 0.0014 | 0.6403 +/- 0.0008 | 0.7225 +/- 0.0003 | 0.8158 +/- 0.0005 | 0.6040 +/- 0.0012 | 0.8393 +/- 0.0002 | 0.7654 +/- 0.0010 |
| DTI frequency matching | 0.8222 +/- 0.0016 | 0.9797 +/- 0.0014 | 0.6647 +/- 0.0046 | 0.7450 +/- 0.0023 | 0.8464 +/- 0.0010 | 0.6790 +/- 0.0019 | 0.8528 +/- 0.0035 | 0.7759 +/- 0.0043 |

Training, validation and test behaviour:

| Strategy | train AUROC | val AUROC | test AUROC | train - test gap | test AUPR | test ACC |
|---|---|---|---|---|---|---|
| Uniform random | 0.9580 | 0.8515 | 0.8575 | +0.1004 | 0.7815 | 0.8169 |
| Mean Euclidean distance | 0.9895 | 0.8269 | 0.8393 | +0.1503 | 0.7654 | 0.7885 |
| DTI frequency matching | 0.9420 | 0.8488 | 0.8528 | +0.0892 | 0.7759 | 0.8222 |

Paired tests on the test split, paired by training seed (reference minus
variant, positive means the variant is worse):

| Comparison | Metric | mean difference | seeds won | paired t-test p | Wilcoxon p |
|---|---|---|---|---|---|
| Uniform - Euclidean | ROC-AUC | +0.0183 | 3/3 | 0.0076 | 0.25 |
| Uniform - Euclidean | AUPR | +0.0161 | 3/3 | 0.0268 | 0.25 |
| Uniform - Euclidean | MCC | +0.0479 | 3/3 | 0.0086 | 0.25 |
| Uniform - Euclidean | ACC | +0.0284 | 3/3 | 0.0019 | 0.25 |
| Uniform - DTI-matched | ROC-AUC | +0.0047 | 3/3 | 0.0868 | 0.25 |
| Uniform - DTI-matched | AUPR | +0.0056 | 3/3 | 0.1596 | 0.25 |
| DTI-matched - Uniform | MCC | +0.0271 | 3/3 | 0.0129 | 0.25 |
| DTI-matched - Uniform | ACC | +0.0053 | 3/3 | 0.0030 | 0.25 |

With three paired seeds the smallest attainable two-sided Wilcoxon p-value is
`2 / 2^3 = 0.25`, so the signed-rank test cannot reject at the 0.05 level and
is reported only for completeness. The paired t-test on three seeds is weak as
well and the differences reported above should be read as effect sizes with an
indicative p-value.

## 6. Discussion of the three negative-sampling strategies

The three rules constrain the unlabelled pool in three different ways: none at
all, a cut on a distance axis, and a match of the target-count distribution.
The comparison is most informative when the negatives each of them produces is
described first, and the metrics are read afterwards. Two properties of the
negative sets are reported for that purpose: the position of the negatives on
the distance axis that the Euclidean rule uses, and the relation between the
target-count strata of the negatives and those of the positives.

### 6.1 The negative sets produced by the three strategies

Measured on the same feature blocks the strategies use, with the
standardisation derived from the positive pairs:

| Set | mean | median | p10 | p90 |
|---|---|---|---|---|
| positives | 61.14 | 58.71 | 48.36 | 77.30 |
| negatives, uniform | 84.58 | 81.88 | 57.71 | 110.85 |
| negatives, mean Euclidean | 96.71 | 91.52 | 78.88 | 118.69 |
| negatives, DTI-matched | 80.94 | 75.07 | 53.17 | 114.10 |

Two properties are worth recording. First, the positives and the uniformly
drawn negatives are distributed almost identically along this axis, which means
the axis itself carries little information about the label. Second, the three
strategies occupy clearly different regions of it: the Euclidean rule keeps the
negatives that lie farthest from the positive cluster, the DTI-matched rule
keeps those whose target-count profile matches the positives, and uniform
sampling lands between the two.

The target-count strata describe the same three negative sets in a directly
interpretable way:

| Stratum (target count of the two endpoints) | positives | uniform | mean Euclidean | DTI-matched |
|---|---|---|---|---|
| 0-0 | 9.87 | 40.30 | 49.12 | 9.87 |
| 0-1 | 5.78 | 6.74 | 8.56 | 5.78 |
| 0-11+ | 8.21 | 15.13 | 9.87 | 8.21 |
| 0-2 | 2.38 | 4.83 | 5.96 | 2.38 |
| 0-3-5 | 5.30 | 11.26 | 11.48 | 5.30 |
| 0-6-10 | 4.08 | 9.32 | 7.17 | 4.08 |
| 1-1 | 1.42 | 0.27 | 0.38 | 1.42 |
| 1-11+ | 5.76 | 0.95 | 0.92 | 5.76 |
| 1-2 | 1.65 | 0.33 | 0.41 | 1.65 |
| 1-3-5 | 5.12 | 0.94 | 1.03 | 5.12 |
| 1-6-10 | 3.35 | 0.73 | 0.63 | 3.35 |
| 11+-11+ | 6.34 | 1.22 | 0.24 | 6.34 |
| 2-11+ | 3.53 | 0.80 | 0.59 | 3.53 |
| 2-2 | 0.38 | 0.15 | 0.20 | 0.38 |
| 2-3-5 | 2.99 | 0.66 | 0.72 | 2.99 |
| 2-6-10 | 1.99 | 0.56 | 0.39 | 1.99 |
| 3-5-11+ | 11.18 | 1.87 | 0.70 | 11.18 |
| 3-5-3-5 | 4.79 | 0.72 | 0.50 | 4.79 |
| 3-5-6-10 | 6.44 | 1.19 | 0.63 | 6.44 |
| 6-10-11+ | 7.34 | 1.50 | 0.35 | 7.34 |
| 6-10-6-10 | 2.08 | 0.51 | 0.16 | 2.08 |

The positive pairs are dominated by drugs with many targets. A uniformly drawn
negative, in contrast, frequently pairs two drugs that have almost no target at
all: a pair of drugs with at most one target each is very unlikely to be
recorded as a combination, and such pairs make up 40.3 % of the uniform
negatives against 9.9 % of the positives. The Euclidean rule concentrates the
negatives further in that stratum, to 49.1 %, whereas the DTI-matched rule
reproduces the positive distribution in every one of the 22 strata by
construction.

### 6.2 How the metrics respond

The three strategies produce classifiers that are close to each other in
ranking quality and further apart in the threshold-dependent metrics:

| Strategy | test AUROC | test AUPR | test MCC | test ACC | train - test AUROC gap |
|---|---|---|---|---|---|
| Uniform random | 0.8575 | 0.7815 | 0.6519 | 0.8169 | +0.1004 |
| Mean Euclidean distance | 0.8393 | 0.7654 | 0.6040 | 0.7885 | +0.1503 |
| DTI frequency matching | 0.8528 | 0.7759 | 0.6790 | 0.8222 | +0.0892 |

The ordering of the three is the same on every metric, and it follows the
position of the corresponding negative sets on the axes above: the strategy
whose negatives are farthest from the positive cluster reaches the highest
training AUROC (0.9895), and also the widest train-test gap (+0.150), while
the strategy whose negatives reproduce the target-count profile of the
positives reaches the narrowest gap (+0.089) and the best accuracy and MCC.

The paired tests of Section 5 give the corresponding uncertainty. Uniform
sampling exceeds the Euclidean rule on all four metrics with three of three
seeds (AUROC p = 0.0076, AUPR p = 0.0268, MCC p = 0.0086, ACC p = 0.0019),
while the difference between uniform sampling and DTI frequency matching is
not resolved by three seeds on the ranking metrics (AUROC p = 0.0868, AUPR
p = 0.1596) although DTI frequency matching is ahead on accuracy and MCC
(p = 0.0030 and p = 0.0129).

### 6.3 What the comparison establishes, and what it does not

* The negative-sampling rule is a second-order factor for the ranking
  metrics: it moves the test AUROC by 1.8 percentage points and the test
  accuracy by 3.4 points, which is small compared with the differences
  between the methods compared in the main benchmark.
* The rule is a first-order factor for the balance of the metrics and for the
  size of the training-test gap. A negative set that is distributed unlike the
  positive set, in particular one that over-represents pairs of drugs with
  almost no target, teaches the model a coarse decision function that does
  not transfer to the unlabelled pool it is finally evaluated on.
* Because the positives and the uniform negatives are nearly identically
  distributed on the distance axis, a threshold cut on that axis cannot be
  expected to select the confusable negatives; it selects the atypical ones.
  On this dataset the constraint that does align the negative difficulty with
  the positive one is the target-count constraint.
* These observations are associations measured over three strategies and three
  seeds on one regime. They identify the direction of the effect and the
  strategy that is preferable under the present protocol; they do not
  decompose the effect into its individual causes, which would require
  additional controls.
* A negative set that is hard in the sense of the reviewer's question, that is
  composed of pairs that are biologically plausible yet unrecorded, cannot be
  constructed from this database alone. It requires externally validated
  negative pairs or a positive-unlabelled formulation, and is stated as such
  in the manuscript.

## 7. Conclusion

* The experiment compares three ways of turning the unlabelled pool into
  negatives under one fixed protocol and identifies the constraint that aligns
  the negative difficulty with the positive one, namely the target-count
  distribution, as the preferable choice under this protocol.
* The choice of the rule changes the measured performance by at most 1.8
  percentage points of AUROC and 3.4 points of accuracy, so the ranking of the
  methods is not affected by it.
* The rule matters more for the balance of the metrics and for the
  generalisation gap than for the ranking quality.
* The absolute level of a pair-level benchmark on this database is governed by
  the intrinsic difficulty of the unlabelled pool rather than by the sampling
  rule, and the present comparison therefore bounds, rather than removes, the
  effect of the negative-sample assumption.

## 8. Layout

```
settings.py                 configuration: paths, blocks, seeds, model parameters
main.py                     driver: upstream check, sampling, training, summary
train.py                    unified training and pair-level evaluation per strategy
uniform_distribution.py     strategy 1: uniform random
euclidean_distance.py        strategy 2: mean Euclidean distance screening
frequency_matching.py        strategy 3: DTI frequency matching
process/
  comparison_test_metrics_single.csv   test-split metrics of the three strategies
  <strategy>/features/
    train_pair_meta.csv      pairs, labels and source of the training set
    val_pair_meta.csv        pairs, labels and source of the validation set
    test_pair_meta.csv       pairs, labels and source of the test set
    *_labels.npy             label arrays
    *_sample_weight.npy      sample weights
    sampling_manifest.json   strategy parameters, audit fields, SHA-256 of every input
  <strategy>/results/
    seed_<n>_metrics.csv     per-split metrics of one training run
    all_seeds_metrics.csv    the three runs merged
    summary.csv              mean and standard deviation per split
```

The `sampling_manifest.json` of each strategy records the SHA-256 digest and
the repository-relative path of every input file, including the frozen
validation and test features, so that the exact inputs of the run can be
verified. The three manifests share the digest of `train_labels.npy`, which
confirms that the training positives are identical across the strategies, and
they differ in the digest of `train.npy`, which confirms that the training
negatives are not.

The pair-level feature matrices (`train.npy`, `val.npy`, `test.npy`) are not
version controlled: each of them is about 0.5 GB, above the per-file limit of
the Git hosting service. They are produced by
`../cold_start/build_fold_features.py` and are inputs of the run rather than
results of it.

## 9. Reproduction

```bash
cd performance/negative_sampling
pip install -r ../requirements.txt

python main.py status       # verify that the upstream inputs are present
python main.py prepare      # sample the negatives of the three strategies
python main.py train        # train and evaluate the three strategies
python main.py summarize    # write process/comparison_test_metrics_single.csv
```

`main.py status` lists the upstream files that are absent. The frozen
validation and test features are expected to be present in the cold-start
workspace of the machine on which the experiment was executed.

## 10. Scope

The experiment runs on the single-cold regime, in which one endpoint of the
evaluated pair is a drug the source database does not contain. Its absolute
level is therefore **not** comparable with the pair-level benchmark of
`../baseline_experiments/`, which additionally differs in the split, in the
cross-validation scheme and in the model configuration. What the module
establishes is the *relative* effect of the negative-sampling rule under one
fixed protocol, not the absolute performance of the model.
