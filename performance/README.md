# SynDrugMD Classifier Benchmark

## 1. Representation

- Drug-pair-level split
- train-only knowledge graph (train DDI + full DTI) TransE 128-dimensional, Xavier initialisation for isolated drugs
- DTI frequency stratification, cross-split mutually exclusive negative sampling (1:1)
- Every drug pair is expanded into all target-combination rows (n1 x n2), `sample_weight = 1/(n1 x n2)`
- Evaluation is carried out at the **drug-pair level**: the combination rows are pooled by weight into one row per drug pair before the metrics are computed

## 2. Directory layout and path conventions

```
diffierent_classifical/
├── main.py                Main entry point (the only run entry point)
├── config.py              Global configuration and path constants (all scripts read their paths from here)
├── features.py            Representation data generation + ComboDataLoader
├── classifiers.py         Classifier registry and unified training/prediction interface
├── evaluation.py          Drug-pair level metric computation, plotting and result export
├── embeddings/            Input data: knowledge graph triples and entity mapping
├── smiles/                Input data: ChemBERTa drug features (768)
├── targets/               Input data: ProtT5 target features (1024)
├── feature_batches/       Intermediate artefacts: {train,val,test}_combo_index.npz + kg/ (generated automatically)
└── benchmark_results/     Result output (generated automatically)
```

## 3. Script responsibilities and call relationships

```
main.py (orchestration)
  ├─ ensure_dataset()        checks {train,val,test}_combo_index.npz; if missing, calls
  │                            features.DrugDatasetGenerator.generate_dataset()
  ├─ load_splits()           → features.ComboDataLoader (dynamic concatenation of the 3,840-dimensional features)
  │                            + build_pair_features (pooling to the drug-pair level by sample_weight)
  └─ evaluation.run_benchmark()
       ├─ classifiers.init_classifiers / train / predict_proba
       └─ evaluation.compute_pair_metrics / _export_reference_metrics / _plot_reference_figure
```

| Module             | Responsibility                                                                               | Not responsible              |
| ------------------ | -------------------------------------------------------------------------------------------- | ---------------------------- |
| `main.py`        | Process orchestration, CLI arguments, data loading and pooling                               | Metric computation, plotting |
| `features.py`    | Representation generation (split/KG/negative sampling/combination index) and feature loading | Classification, evaluation   |
| `classifiers.py` | Classifier definitions and the unified training/prediction interface                         | Data, metrics                |
| `evaluation.py`  | Drug-pair level metrics, plotting, result export                                             | Training data preparation    |
| `config.py`      | Seed, paths, pooling budget, threshold                                                       | —                           |

## 4. How to run

```bash
cd performance/diffierent_classifical
python main.py                              # trains directly when the features and intermediate artefacts are present
python main.py 						        # subsample the training drug pairs (memory protection, optional)
python main.py               			    # override the random seed
```

Dependency tiers:

| Tier                   | Packages                                   | Description                                                                                                                    |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Required               | numpy / pandas / scikit-learn / matplotlib | Feature construction, training, metrics and plotting                                                                           |
| Optional               | xgboost                                    | if missing, that classifier is skipped and the remaining classifiers run unchanged, and the metrics CSV then has fewer columns |
| Conditionally required | torch / pykeen                             | only needed when `feature_batches/kg/` is missing and the train-only TransE has to be retrained                              |

## 5. Output files

| File                                        | Content                                                                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark_results/SynDrugMD_metrics.csv` | Test-set metrics with the classifiers as columns and the metrics as rows (Acc/Sen/Spe/Pre/Mcc/AUROC/AUPRC/F1, at the drug-pair level) |
| `benchmark_results/SynDrugMD_figure.png`  | Panel A: grouped bar chart of the 8 metrics x classifiers; Panel B: combined ROC and PRC curves                                       |

Intermediate artefacts (generated automatically; they can be deleted and rebuilt by the scripts):

| File                                                 | Content                                                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `feature_batches/{train,val,test}_combo_index.npz` | Weighted target-combination index (pair_index / drug index / target index / label / weight) |
| `feature_batches/kg/drug_kg_features.npy`          | train-only TransE drug embeddings (3496 x 128)                                              |
| `feature_batches/kg/drug_smiles_features.npy`      | ChemBERTa drug features (3496 x 768)                                                        |
| `feature_batches/kg/target_features.npy`           | ProtT5 target features (1620 x 1024)                                                        |

## 6. Protocol notes

- Drug-pair level evaluation: the combination rows are pooled by `sample_weight = 1/(n1 x n2)` into one row per drug pair,
  so the sample granularity of training and evaluation is identical to the mean-pooling protocol of `performance/baseline_experiments`.
- The pooling row budget (`pooling_row_budget = 200000`) is only a memory protection measure, not part of the protocol itself, and it can be adjusted in `config.py`.
- All classifiers use the same training interface and the same drug-pair level threshold (`config.CONFIG['threshold'] = 0.5`),
  so the metrics are strictly comparable; when `xgboost` is missing that classifier is skipped, which does not affect the remaining results.
- The drug-pair split is completed before the feature construction; the knowledge graph is rebuilt from the training positives only, and the combination edges of the validation and test sets never enter the graph.
