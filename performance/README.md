# `performance/` -- model evaluation workspace

This directory is the single place where every performance experiment of
KGP-DC lives. It contains three modules:

| Module | Purpose |
|---|---|
| `baseline_experiments/` | Leak-free, same-protocol comparison of KGP-DC against four re-implemented literature baselines on pair-level cross-validation. |
| `negative_sampling/` | Ablation on the negative-sampling strategy: uniform random sampling versus Euclidean / RMS distance screening of the unlabelled pool. |
| `../cold_start/` | Strict unseen-drug (single-cold and dual-cold) evaluation with a train-only knowledge graph. |

The original study artefacts are not overwritten by a new run. New runs write
to `results/` inside the module they belong to.

---

## 1. `baseline_experiments/` -- same-protocol baseline comparison

### 1.1 Task definition

The prediction target is a **drug-combination association recorded in the
source database**: for an unordered, canonicalised drug pair, does the
database record that the two drugs are used in combination? The label is a
database-coverage label, not a measured synergy value, not a clinical DDI and
not a safety interaction. Pairs that the database does not record are
*unlabelled* rather than *true negatives*; Section 1.9 quantifies how much
that distinction matters.

The evaluation unit is the **drug pair**. A pair is expanded to its target
combinations only inside the model, and every reported number is computed on
one score per pair.

### 1.2 Data

| Item | Value |
|---|---|
| Drug pairs | 100,066 |
| Positives (recorded combinations) | 50,033 |
| Negatives (sampled unlabelled pairs) | 50,033 |
| Drugs | 3,496 |
| Targets | 1,619 |
| Feature vector | 3,840 = 2 x (128 KG + 768 ChemBERTa SMILES + 1,024 ProtT5 mean target) |

### 1.3 Splits

1. **Warm split** -- pair-level 60 / 20 / 20 random split (seed 42) of the
   canonicalised, de-duplicated, self-pair-free pair set.
2. **Negative sampling** -- one negative per positive, stratified by the number
   of drug-target interactions of the two endpoints, sampled from the pool of
   drugs that have both a SMILES string and a knowledge-graph vector. All
   known positive pairs are excluded globally, and the negative pools of the
   three splits are disjoint from each other.
3. **Cross-validation** -- the union of the three splits is re-partitioned into
   five pair-level folds (seed 42). The partition is stratified by the same
   target-count bins and keeps the 1:1 label balance inside every fold. Each
   fold contains roughly 70,000 training, 10,000 validation and 20,000 test
   pairs; the five test folds partition all 100,066 pairs exactly once.

```
data/split_data/
|-- {train,val,test}_pair_meta.csv     pair id, drug ids, label, source
|-- {train,val,test}_labels.npy         label array (derived, regenerable)
|-- drug_table.csv / target_table.csv   identifier tables
|-- drug_smiles_features.npy            3496 x 768  ChemBERTa
|-- target_features.npy                 1620 x 1024 ProtT5
|-- drug_target_index.npy               drug -> target index
|-- drug_kg_features.npy                3496 x 128   train-only TransE
`-- folds/fold_{1..5}_{train,val,test}.csv
```

### 1.4 The five methods

All five consume the *same* pairs, the *same* folds, the *same* negatives and
the *same* metric set. None of them reads a number from a publication.

| # | Method | Input modality | Design |
|---|---|---|---|
| 1 | **KGNN** | drug entity ids + training-fold knowledge graph | PyTorch port of the upstream KGCN. The graph of a fold contains the fold's training positive pairs as bidirectional `combination` edges plus all drug-target `interaction` edges. Relation-aware message passing with a learned aggregator, scored by the inner product of the two drug representations. |
| 2 | **GraRep + MLP** | node embeddings of the training DDI graph | GraRep gives the node embeddings only, so a symmetric pair representation (`abs(z1 - z2)`, `z1 * z2`) is fed to an MLP head. The graph is the undirected fold-training DDI graph; the upstream MATLAB core is used when available and the identical formulas are re-implemented in NumPy otherwise, and the backend used is recorded in every run. |
| 3 | **adapted DeepDDS** | RDKit molecular graphs + dual GAT | The upstream architecture without the cell-line branch, because the dataset contains neither cell-line expression data nor cell-line-specific synergy labels. |
| 4 | **adapted DeepTraSynergy** | SMILES patch-Transformer + DTI / ProtT5 targets | The upstream architecture without the cell-line branch, for the same reason. |
| 5 | **KGP-DC** | TransE + ChemBERTa + mean ProtT5 + XGBoost | The knowledge graph is rebuilt **per fold**: the graph contains the fold's training positive pairs as `combination` edges and all drug-target `interaction` edges, the TransE embeddings are learned on that graph only, and the resulting 3,840-dimensional pair vectors are classified by a gradient-boosted tree ensemble. |

`adapted` in the table means: the cell-line branch was removed, predictions are
aggregated to the drug-pair level, and the numbers are therefore **not**
comparable with the original publications.

### 1.5 One protocol for all five methods

* **Threshold** -- Youden's J on the validation split only. The test labels
  never take part in model selection, threshold selection or early stopping.
* **Hyper-parameter search** -- ten candidate configurations per method, all
  scored by the validation AUPRC of fold 1, with an identical early-stopping
  rule. The selected configurations are stored in `work/tuning/`.
* **Early stopping** -- validation AUPRC, with an identical patience per model.
* **Metrics** -- accuracy, sensitivity, specificity, precision, F1, MCC, AUROC
  and AUPRC, all computed on the drug-pair level.
* **Runs** -- 5 folds x seed 42. Every run appends a record to `work/runs.csv`
  containing the threshold, the confusion-matrix counts, all metrics, the
  elapsed time and a JSON profile of the configuration actually used.

### 1.6 Leakage audit

`src/audit/leakage_audit.py` re-reads the committed metadata independently of
the training code and reports:

```
[warm_split] PASS
  pair overlap        : {'train&val': 0, 'train&test': 0, 'val&test': 0}
  pos/neg conflict    : 0 for every split combination
  global conflict     : 0
  self / non-canonical: 0 / 0
  drug overlap        : {'train&val': 2833, 'train&test': 2810, 'val&test': 2720}  (observation)
[folds] PASS
  fold 1..5: pair overlap = 0, pos/neg conflict = 0
  test folds cover all pairs: True (100066/100066)
[kg_isolation] PASS
  all five models read the fold splits and threshold on the validation split
```

The **drug** overlap is not zero by design: the pair-level split is an
interpolation setting in which both endpoints of an evaluated pair are seen
during training. The strict unseen-drug setting is the separate cold-start
experiment in `../cold_start/`, and the two settings must not be conflated.

### 1.7 Results

Five-fold mean +/- standard deviation, seed 42, 100,066 pairs.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| KGNN | 0.9132 +/- 0.0020 | 0.8711 +/- 0.0069 | 0.9552 +/- 0.0064 | 0.9512 +/- 0.0064 | 0.9093 +/- 0.0023 | 0.8293 +/- 0.0041 | 0.9459 +/- 0.0011 | 0.9560 +/- 0.0017 |
| **GraRep + MLP** | **0.9852 +/- 0.0003** | **0.9917 +/- 0.0012** | **0.9787 +/- 0.0014** | **0.9790 +/- 0.0014** | **0.9853 +/- 0.0003** | **0.9705 +/- 0.0006** | **0.9953 +/- 0.0004** | **0.9926 +/- 0.0007** |
| adapted DeepDDS | 0.8796 +/- 0.0043 | 0.8995 +/- 0.0066 | 0.8597 +/- 0.0094 | 0.8651 +/- 0.0074 | 0.8820 +/- 0.0040 | 0.7599 +/- 0.0085 | 0.9445 +/- 0.0036 | 0.9320 +/- 0.0043 |
| adapted DeepTraSynergy | 0.9639 +/- 0.0019 | 0.9694 +/- 0.0073 | 0.9584 +/- 0.0047 | 0.9589 +/- 0.0042 | 0.9641 +/- 0.0020 | 0.9279 +/- 0.0039 | 0.9912 +/- 0.0004 | 0.9876 +/- 0.0006 |
| KGP-DC | 0.9712 +/- 0.0023 | 0.9711 +/- 0.0068 | 0.9714 +/- 0.0069 | 0.9714 +/- 0.0065 | 0.9712 +/- 0.0023 | 0.9425 +/- 0.0045 | 0.9931 +/- 0.0004 | 0.9895 +/- 0.0010 |

KGP-DC ranks second on all eight metrics. Paired significance tests against
KGP-DC (`src/stats_test.py`, `results/significance.csv`,
`results/significance.md`):

| Comparison | Metric | mean difference | folds won | paired t-test p | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| KGP-DC vs KGNN | AUROC | +4.719 pp | 5/5 | 4.6e-08 | [+4.28, +5.14] pp |
| KGP-DC vs GraRep + MLP | AUROC | **-0.221 pp** | 0/5 | 4.3e-04 | **[-0.38, -0.08] pp** |
| KGP-DC vs adapted DeepDDS | AUROC | +4.863 pp | 5/5 | 7.6e-06 | [+4.14, +5.75] pp |
| KGP-DC vs adapted DeepTraSynergy | AUROC | +0.187 pp | 5/5 | 1.0e-03 | [+0.05, +0.36] pp |
| KGP-DC vs KGNN | AUPRC | +3.359 pp | 5/5 | 1.6e-07 | [+3.02, +3.78] pp |
| KGP-DC vs GraRep + MLP | AUPRC | -0.307 pp | 0/5 | 3.7e-03 | [-0.59, +0.06] pp |
| KGP-DC vs adapted DeepDDS | AUPRC | +5.757 pp | 5/5 | 5.6e-06 | [+4.99, +6.90] pp |
| KGP-DC vs adapted DeepTraSynergy | AUPRC | +0.195 pp | 5/5 | 3.9e-02 | [-0.14, +0.59] pp |

Reading notes:

* The primary test is the **paired t-test over the five folds**. The
  Wilcoxon signed-rank test is reported next to it in `results/significance.csv`
  together with its resolution limit: with five folds the smallest attainable
  two-sided p-value is `2 / 2^5 = 0.0625`, so the signed-rank test cannot
  reject at the 0.05 level and must not be quoted as if it could.
* The bootstrap interval resamples the test pairs of each fold with
  replacement; it answers whether the metric difference is stable with
  respect to the finite test sample.
* Differences are reported as **reference minus baseline**; a negative value
  means the baseline is better.
* KGP-DC is significantly better than three of the four baselines and
  significantly **worse** than GraRep + MLP on AUROC. The AUPRC difference
  against GraRep + MLP has a bootstrap interval that includes zero.

### 1.8 Reproduction

```bash
cd performance/baseline_experiments
pip install -r ../requirements.txt

# 1. assemble the drug-level assets (copies them from ../../cold_start/)
python data/prepare_assets.py --apply

# 2. validate the split, the folds and the model implementations
python main.py check

# 3. optional: rebuild the large pair-level feature matrices
#    (multi-GB scratch space; only needed for the legacy warm feature path)
python data/prepare_assets.py --apply --rebuild-features

# 4. optional: equal-budget hyper-parameter search (CPU, hours)
python main.py tune --models all

# 5. train all five methods on the five folds
python main.py train

# 6. recompute the metrics table from the stored predictions
python main.py evaluate

# 7. paired significance tests -> results/significance.{csv,md}
python src/stats_test.py --bootstrap 1000

# 8. leakage audit -> results/leakage_audit.json
python src/audit/leakage_audit.py --json results/leakage_audit.json

# 9. figures -> results/figures.png
python main.py plot
```

Runtime on the reference machine (6-core Xeon E5-2620 v3, 68 GB RAM, CPU
only): roughly 80 s for KGNN per fold, 70 s for GraRep, 120 s for the two
adapted deep models and 60 s for KGP-DC per fold, i.e. about 15 minutes for the
complete five-fold run of the five methods.

### 1.9 How to read these numbers -- the limits of this benchmark

The numbers above are the result of a **strictly leak-free** protocol: no
pair is shared between splits, no negative coincides with a known positive,
the knowledge graph of a fold contains only that fold's training positives and
the threshold is taken from the validation split. Nevertheless they must not be
read as evidence of a near-perfect biological predictor, for two reasons that
were measured directly on the fold-1 test split.

**Reason 1 -- the uniform random negatives are easy.** A single trivial
feature set built from the knowledge-graph neighbourhood separates the classes
almost completely:

| Diagnostic feature (fold-1 test split) | AUROC |
|---|---:|
| shared target count | 0.5046 |
| target Jaccard | 0.5046 |
| mean ProtT5 target cosine | 0.5173 |
| **common neighbours in the training DDI graph** | **0.9573** |
| minimum degree of the two endpoints | 0.9471 |

95.31 % of the test positives share at least one common neighbour with the
training DDI graph, against 34.38 % of the uniformly sampled negatives
(mean common-neighbour count 73.15 versus 1.18). The DDI graph of SynDrugMD
is extremely concentrated: the mean degree is 20 but the median is 1, 295
drugs with a degree of at least 100 carry 76 % of all edges, and several
high-degree drugs have a neighbourhood with an internal edge density of
0.25-0.29, i.e. interchangeable drugs of the same therapeutic class that are
combined with each other. Under uniform random negatives the task therefore
degrades towards a neighbourhood-membership question, which is exactly the
information that GraRep reads directly and which explains why it is the
strongest baseline here.

**Reason 2 -- the unlabelled pool is not a set of true negatives.** A pair that
the database does not record is usually untested rather than biologically
incompatible, so the negatives carry an unknown but non-zero false-negative
rate.

A pilot experiment was therefore run with **neighbour-matched hard negatives**:
negative pairs restricted to pairs that share at least one common training
DDI neighbour, with the common-neighbour distribution matched to that of the
positives (candidate pool 1,137,195 pairs; mean common neighbours 72.29 for
the hard negatives against 73.15 for the positives and 1.18 for the uniform
ones). The pilot used a gradient-boosting stand-in classifier on fold 1 only,
so it is **not part of the reported table**:

| Regime | AUROC |
|---|---:|
| train on uniform negatives, test on uniform negatives | 0.9870 |
| train on uniform negatives, test on neighbour-matched hard negatives | 0.5266 |
| train and test on neighbour-matched hard negatives | 0.6915 |

The same trained model drops from 0.987 to 0.527 -- chance level -- when only
the negative distribution of the test set changes. The reported metrics
therefore measure the intrinsic difficulty of the negative-sampling protocol,
not the biological discriminability of the models. The honest reading of this
benchmark is: **KGP-DC is a strong, efficient and reproducible classifier of
recorded drug-combination associations, it is not the best model under this
protocol (GraRep + MLP is), and the absolute level of all methods is
inflated by the negative-sampling protocol.**

The full neighbour-matched benchmark, the hard-negative and PU variants, and
the multi-seed repetition are the planned continuation and are listed in the
known limitations below.

---

## 2. `negative_sampling/` -- negative-sampling strategy ablation

Purpose: quantify how the choice of negative samples changes the apparent
performance, under a fixed split, fixed training positives, fixed candidate
pool, fixed features and fixed hyper-parameters. Only the screening rule of
the training negatives changes.

* **Metric 1 -- mean Euclidean distance** to the positive-sample cluster centre,
  computed over the independent target combinations of a pair, without any
  average pooling of the target features.
* **Metric 2 -- RMS distance**, computed on the same fixed vector, so that the
  two rules differ only in the distance definition and not in the constants.

Thresholds `0.9 / 1.0 / 1.1 x AED` x 5 seeds (42, 52, 62, 72, 82) = 30 runs.
The validation and test negatives are a single fixed set in every run; only
the training negatives are filtered.

Headline result (`output/reports_gpu_lite/mean_std_summary.csv`):

| Method | Threshold | ACC | SEN | SPE | PRE | F1 | MCC | ROC-AUC | AUPR |
|---|---|---|---|---|---|---|---|---|---|
| Mean Euclidean | 0.9 | 0.8005 +/- 0.0051 | 0.6380 +/- 0.0092 | 0.9629 +/- 0.0020 | 0.9451 +/- 0.0031 | 0.7618 +/- 0.0072 | 0.6355 +/- 0.0089 | **0.9218 +/- 0.0041** | **0.9204 +/- 0.0031** |
| RMS | 0.9 | **0.8029 +/- 0.0037** | **0.6445 +/- 0.0075** | 0.9612 +/- 0.0011 | 0.9432 +/- 0.0015 | **0.7658 +/- 0.0054** | **0.6386 +/- 0.0061** | 0.9195 +/- 0.0029 | 0.9184 +/- 0.0020 |
| Mean Euclidean | 1.0 | 0.7646 +/- 0.0030 | 0.5741 +/- 0.0053 | 0.9551 +/- 0.0026 | 0.9274 +/- 0.0039 | 0.7092 +/- 0.0034 | 0.5724 +/- 0.0057 | 0.9035 +/- 0.0032 | 0.8991 +/- 0.0034 |
| RMS | 1.0 | 0.7683 +/- 0.0029 | 0.5801 +/- 0.0087 | 0.9566 +/- 0.0029 | 0.9304 +/- 0.0035 | 0.7146 +/- 0.0056 | 0.5793 +/- 0.0035 | 0.9017 +/- 0.0048 | 0.8996 +/- 0.0028 |
| Mean Euclidean | 1.1 | 0.7170 +/- 0.0027 | 0.5313 +/- 0.0049 | 0.9026 +/- 0.0040 | 0.8452 +/- 0.0041 | 0.6524 +/- 0.0039 | 0.4674 +/- 0.0058 | 0.8411 +/- 0.0032 | 0.8233 +/- 0.0031 |
| RMS | 1.1 | 0.7188 +/- 0.0066 | 0.5321 +/- 0.0111 | 0.9054 +/- 0.0032 | 0.8490 +/- 0.0063 | 0.6542 +/- 0.0100 | 0.4716 +/- 0.0126 | 0.8467 +/- 0.0066 | 0.8287 +/- 0.0066 |

Conclusions that the data supports:

* The best threshold is `0.9 x AED`. Harder negatives (1.0, 1.1) give a lower
  and more realistic performance level: the AUROC falls from 0.92 to 0.84.
* Mean Euclidean and RMS are statistically indistinguishable (paired Wilcoxon
  over the five seeds, p > 0.05 in every metric).
* The model is much better at recognising negatives (SPE about 0.96) than
  positives (SEN about 0.64).
* The training AUC of 0.9963-0.9977 against a test AUC of about 0.92 indicates
  overfitting; the run has no early stopping and uses a fixed 500 boosting
  rounds, so the numbers are internally comparable but not final.

Caveat: this module was executed in the GPU-lite configuration, which caps the
number of target combinations per pair at 16 and does not average-pool the
target features. Its absolute level is therefore **not** comparable with the
pair-level numbers of `baseline_experiments/`; only the relative comparison
between the six rows is meaningful.

Inputs: unlike `baseline_experiments/`, this module does not build its own
split. It consumes the cold-start feature store and the single-cold
unlabelled candidate pool of an external, read-only workspace, which is not
redistributed with this repository. `python aed_rms_experiment.py status`
reports exactly which inputs are missing and exits with code 2, so the
dependency is machine-checkable rather than implicit. The committed report
under `output/reports_gpu_lite/` is the record of the completed run.

---

## 3. `../cold_start/` -- strict unseen-drug evaluation

The pair-level protocol of Section 1 is an interpolation setting. The
cold-start module provides the inductive setting: drug pools are assigned to
train / validation / test, the test drugs occur in no training combination
edge, and the knowledge graph is rebuilt per split so that no evaluated pair
is a graph edge (`held_out_pairs_in_graph: 0` in every `kg_summary.json`).
Three regimes are prepared -- `warm`, `single` (one held-out endpoint) and
`dual` (both endpoints held out).

The splits and the per-split TransE embeddings are committed; the trained
models and their metrics are not yet part of the repository and are the next
item of work.

---

## 4. Known limitations and planned continuation

1. **One seed.** The reported baseline table uses seed 42. The planned
   continuation repeats the complete protocol with at least three seeds.
2. **Uniform negatives.** Section 1.9 shows that the absolute metric level is
   driven by the negative-sampling protocol. The neighbour-matched hard
   negative benchmark is designed and piloted; it has to be run with the
   official pipeline, over all five folds and all five methods.
3. **No external negative source.** Validated negatives from DrugBank or
   DDInter and literature-based hard negatives are not yet included.
4. **No PU-learning analysis.** The unknown false-negative rate of the
   unlabelled pool is currently discussed, not estimated.
5. **Cold-start metrics are missing.** The splits and the knowledge graphs
   exist, the results do not.
6. **Protocol asymmetries.** The KGP-DC implementation applies an isotonic
   calibration on a drug-disjoint half of the validation split, and the
   hyper-parameter search of KGP-DC and adapted DeepDDS uses a 20,000-pair
   subsample of the fold-1 training set while the other methods use the full
   training set. Both asymmetries are conservative with respect to KGP-DC and
   are documented rather than hidden; they should be removed in the final
   protocol.

---

## 5. Reproducibility checklist

| Item | Location |
|---|---|
| Fixed seeds | `src/common.py` (`SEEDS`, `SPLIT_SEED`), `data/split_rules.py` |
| Split and fold rules | `data/split_rules.py` |
| Negative sampling rules | `data/negative_rebuild.py` |
| Asset assembly | `data/prepare_assets.py` |
| Metric definitions | `src/common.py` (`binary_metrics`, `youden_threshold`) |
| Per-run records | `work/runs.csv` |
| Per-fold predictions | `work/predictions/` |
| Hyper-parameter search | `work/tuning/` |
| Result table | `results/table.csv`, `results/figures.png` |
| Significance tests | `results/significance.csv`, `results/significance.md` |
| Leakage audit | `results/leakage_audit.json` |
| Environment | `../requirements.txt` |
