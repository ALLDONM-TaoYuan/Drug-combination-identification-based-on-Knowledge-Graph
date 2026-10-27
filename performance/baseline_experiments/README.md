# `baseline_experiments/` -- same-protocol baseline comparison

Five methods are trained and evaluated on the same drug-pair dataset, the
same pair-level splits, the same negative samples, the same metrics and the
same decision-threshold rule.

| # | Method | Input modality |
|---|---|---|
| 1 | KGNN | drug entity IDs + training-fold knowledge graph |
| 2 | GraRep + MLP | node embeddings of the training DDI graph |
| 3 | adapted DeepDDS (without cell-line branch) | RDKit molecular graphs + dual GAT |
| 4 | adapted DeepTraSynergy (without cell-line branch) | SMILES patch-Transformer + DTI / ProtT5 targets |
| 5 | KGP-DC | TransE + ChemBERTa + mean ProtT5 + XGBoost |

No number is copied from the literature: every baseline is retrained locally.

The full protocol, the results, the significance tests and the discussion of
the limits of the benchmark are documented in `../README.md`, Section 1.

## Protocol

* **Samples**: 100,066 drug pairs (50,033 positives, 50,033 negatives).
* **Positives**: approved combinations from the KGP-DC database, canonicalised
  (`min,max`), deduplicated, self-pairs and label conflicts removed.
* **Negatives**: generated once, 1:1, stratified by the number of DTI targets,
  and **disjoint across splits** (no pair appears in two splits).
* **Splits**: pair-level 5-fold cross-validation (seed 42); each fold contains
  about 70k / 10k / 20k pairs in train / validation / test, with exact 1:1
  label balance in every split.
* **Knowledge graph**: per fold, `combination` edges are the **training
  positives only**; `interaction` edges are all drug-target pairs. Validation
  and test combination edges are never graph edges.
* **Threshold**: Youden's J on the validation split only; test data is never
  used for model selection or threshold selection.
* **Tuning**: equal budget for all five methods (10 configurations, fold-1
  validation AUPRC, identical early-stopping rule, no test data). The selected
  configurations are in `.work/tuning/`.
* **Hardware**: CPU only (6-core Xeon E5-2620 v3, 68 GB RAM); no GPU was usable.

## Results (5-fold mean +/- std, seed 42)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| KGNN | 0.9132 +/- 0.0020 | 0.8711 +/- 0.0069 | 0.9552 +/- 0.0064 | 0.9512 +/- 0.0064 | 0.9093 +/- 0.0023 | 0.8293 +/- 0.0041 | 0.9459 +/- 0.0011 | 0.9560 +/- 0.0017 |
| GraRep + MLP | 0.9852 +/- 0.0003 | 0.9917 +/- 0.0012 | 0.9787 +/- 0.0014 | 0.9790 +/- 0.0014 | 0.9853 +/- 0.0003 | 0.9705 +/- 0.0006 | 0.9953 +/- 0.0004 | 0.9926 +/- 0.0007 |
| adapted DeepDDS | 0.8796 +/- 0.0043 | 0.8995 +/- 0.0066 | 0.8597 +/- 0.0094 | 0.8651 +/- 0.0074 | 0.8820 +/- 0.0040 | 0.7599 +/- 0.0085 | 0.9445 +/- 0.0036 | 0.9320 +/- 0.0043 |
| adapted DeepTraSynergy | 0.9639 +/- 0.0019 | 0.9694 +/- 0.0073 | 0.9584 +/- 0.0047 | 0.9589 +/- 0.0042 | 0.9641 +/- 0.0020 | 0.9279 +/- 0.0039 | 0.9912 +/- 0.0004 | 0.9876 +/- 0.0006 |
| KGP-DC | 0.9712 +/- 0.0023 | 0.9711 +/- 0.0068 | 0.9714 +/- 0.0069 | 0.9714 +/- 0.0065 | 0.9712 +/- 0.0023 | 0.9425 +/- 0.0045 | 0.9931 +/- 0.0004 | 0.9895 +/- 0.0010 |

KGP-DC ranks second on all eight metrics. It is significantly better than
KGNN, adapted DeepDDS and adapted DeepTraSynergy (paired t-test over the five
folds, p < 0.005 for all three) and significantly **worse** than GraRep + MLP
on AUROC (-0.221 pp, p = 4.3e-04, bootstrap 95% CI [-0.38, -0.08] pp). See
`results/significance.md` for the full table.

## Repository layout

```
data/
  prepare_assets.py     assemble the drug-level assets from ../../cold_start/
  split_rules.py        split, negative-sampling and 5-fold rules
  negative_rebuild.py   rebuild the negatives and the warm split
  split_data/           pair metadata, fold definitions, identifier tables
models/
  KGNN/model.py
  GraRep/model.py
  DeepDDS/model.py
  DeepTraSynergy/model.py
  KGP-DC/model.py
src/
  common.py             paths, data access, metrics, result I/O
  train.py              5-fold training
  tune.py               equal-budget, validation-only hyper-parameter search
  evaluate.py           metrics and results/table.csv
  plot_results.py       results/figures.png
  stats_test.py         paired t-test, Wilcoxon and paired bootstrap
  audit/leakage_audit.py  pair overlap and label conflict audit
results/
  table.csv             model + eight metrics (mean +/- std over the folds)
  significance.csv      paired significance tests
  significance.md       the same, formatted for the manuscript
  leakage_audit.json    machine-readable leakage audit
  figures.png           panel A (metrics) and panel B (ROC / PRC)
work/
  tuning/               10-configuration search per model (validation only)
  runs.csv              per-fold run records (threshold, metrics, runtime)
  predictions/          per-fold test predictions of every model
main.py                 command-line entry point
```

## Data provenance

`data/split_data/` holds the pair metadata, the fold definitions and the
identifier tables. The drug-level feature arrays are **not** distributed:
they are copies of the files that live under
`../../cold_start/entity_split_data/features/` and
`../../cold_start/kg_embedding/warm/train/`, and they are assembled with

```bash
python data/prepare_assets.py --apply
```

The large pair-level feature matrices (3,840 features per drug pair) are
reproducible from the same drug-level assets with

```bash
python data/prepare_assets.py --apply --rebuild-features
```

They are excluded from version control because of their size, not because they
are irreproducible. The drug-level TransE embedding used for the warm split is
read from `$KGP_DC_KG_FEATURES` (default `data/split_data/drug_kg_features.npy`).
The MATLAB executable, if the upstream GraRep core is available, is read from
`$MATLAB_EXE` (default `matlab` on `PATH`); otherwise the same formulas are
re-implemented in NumPy and the backend is recorded in the run profile.

## Running

```bash
pip install -r ../requirements.txt

python data/prepare_assets.py --apply   # assemble the drug-level assets
python main.py check                     # validate splits and model files
python main.py tune --models all         # optional: equal-budget search (CPU, hours)
python main.py train                     # 5 folds x seed 42
python main.py evaluate                  # results/table.csv
python src/stats_test.py --bootstrap 1000  # results/significance.{csv,md}
python src/audit/leakage_audit.py --json results/leakage_audit.json
python main.py plot                      # results/figures.png
```

`main.py train` reads `work/tuning/<model>_best.json` when present; otherwise
the default configuration in each `model.py` is used.

## Implementation notes

* **KGNN** is a PyTorch port of the upstream Keras/TF 1.13 implementation (that
  stack is not installable on the available Python versions). Neighbour
  sampling, depth, aggregators and scoring follow the upstream code.
* **GraRep** itself only provides node embeddings. The reported pipeline adds a
  symmetric pair representation and a small MLP head, hence the name
  `GraRep + MLP`. The upstream MATLAB core is used when available; otherwise the
  same formulas are reimplemented in NumPy.
* **adapted DeepDDS** and **adapted DeepTraSynergy** remove the cell-line
  branch because the dataset contains no cell-line or expression data. They are
  not full reproductions of the original models, and their numbers are not
  comparable to the original papers.
* The negative-sample overlap of an earlier data version was found and fixed;
  `data/negative_rebuild.py` documents the correction (30,431 unique negatives
  for 50,033 required pairs before the fix).
