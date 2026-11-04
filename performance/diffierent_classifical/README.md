# Different classifiers on two datasets

**Experiment design idea.** This module discusses how much of the outcome is decided by the learning algorithm itself when the feature representation is held fixed.

Implementation path: The same benchmark -- same features, same splits, same negatives, same metric definitions -- is executed with a set of standard classifiers on the two datasets `DrugCombDB` and `SynDrugMD KG`, so that the classifier used in the published workflow can be placed in context and the contribution of the feature representation can be separated from the contribution of the model.

---

## 1. Layout

```
diffierent_classifical/
|-- README.md                        this document (English)
|-- README.zh-CN.md                  this document (Chinese)
|-- DrugCombDB/
|   |-- fea.py                       stage 1: pair feature construction
|   |-- classifier.py                stage 2: classifier benchmark
|   |-- embeddings/                  KG embeddings and mapped triples
|   |-- smiles/                      SMILES features and identifiers
|   `-- proteins/                    target (protein) features and identifiers
`-- SynDrugMD KG/
    |-- fea.py                       stage 1: pair feature construction
    |-- classifier.py                stage 2: classifier benchmark
    |-- embeddings/                  KG embeddings and mapped triples
    |-- smiles/                      SMILES features and identifiers
    `-- targets/                     target features and identifiers
```

---

## 2. Datasets

| Item              | `DrugCombDB`                                                                                                                                                            | `SynDrugMD KG`                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Knowledge graph   | `embeddings/entity_emb.npy`, `embeddings/entity_mapping.csv`, `embeddings/mapped_triples.npy`, `embeddings/relation_emb.npy`, `embeddings/relation_mapping.csv` | `embeddings/entity_embeddings.npy`, `embeddings/entity_mapping.csv`, `embeddings/mapped_triples.npy`, `embeddings/relation_embeddings.npy`, `embeddings/relation_mapping.csv` |
| Drug features     | `smiles/smiles_features.npy`, `smiles/smiles_ids.npy`                                                                                                                 | `smiles/smiles_features.npy`, `smiles/smiles_ids.npy`                                                                                                                               |
| Target features   | `proteins/protein_features.npy`, `proteins/protein_ids.npy`                                                                                                           | `targets/target_features.npy`, `targets/target_ids.npy`                                                                                                                             |
| Positive pairs    | triples with relation id `0` (drug-drug combination)                                                                                                                    | triples with relation id `0` (drug-drug combination)                                                                                                                                  |
| Drug-target links | triples with relation id `1`                                                                                                                                            | triples with relation id `1`                                                                                                                                                          |
| Relative size     | small, workstation-sized                                                                                                                                                  | large, server-sized                                                                                                                                                                     |

Both datasets expose the same three feature modalities for every drug and every target: a knowledge-graph embedding, a SMILES embedding and a target-protein embedding. The two runs differ only in the scale of the inputs and in the resource settings, not in the protocol.

---

## 3. Classifier benchmark (`classifier.py`)

### 3.1 Classifiers under comparison

| Classifier          | Configuration                                                                   | Rationale                                                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Random Forest       | 100 trees, `max_depth=15`, `max_samples=0.5`                                | bagged trees, insensitive to the scale of the features                                                                                                                                                                         |
| Logistic Regression | `lbfgs`, `max_iter` 500 (DrugCombDB) / 1000 (SynDrugMD KG)                  | linear baseline on the same features                                                                                                                                                                                           |
| Decision Tree       | `max_depth=15`                                                                | single-tree baseline, axis-aligned decision surfaces                                                                                                                                                                           |
| Gaussian NB         | default                                                                         | generative baseline, cheap and assumption-free                                                                                                                                                                                 |
| MLP                 | `(100, 50)` (DrugCombDB) / `(128, 64)` (SynDrugMD KG), Adam, early stopping | non-linear reference, `early_stopping` on the internal validation fraction                                                                                                                                                   |
| XGBoost             | 100 trees, `max_depth=8`, `tree_method='hist'`                              | gradient boosting, enabled in the `DrugCombDB` script when the package is importable; the slot is reserved but switched off in the `SynDrugMD KG` script, where the run is executed with the scikit-learn classifiers only |

Every estimator receives `random_state=42`, so a rerun on the same inputs reproduces the same model.

### 3.2 Resource settings

The two folders were produced under different constraints and keep their own settings, which is the reason why the two `classifier.py` files differ in the thread caps and in the sampling ratios:

| Setting                           | `DrugCombDB`                                                    | `SynDrugMD KG`                      |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------------- |
| Sampling ratio                    | 0.20                                                              | 0.15                                  |
| Maximum rows (train / val / test) | 50,000 / 10,000 / 10,000                                          | 100,000 / 15,000 / 15,000             |
| Threads                           | 2 (`n_jobs=2`, `OMP_NUM_THREADS=2`)                           | 32 (`MAX_THREADS=32`)               |
| Platform guard                    | `SetThreadExecutionState` on Windows, `PYTHONHASHSEED` pinned | `CUDA_VISIBLE_DEVICES=-1`, CPU only |

These caps are a memory guard, not part of the protocol: the benchmark has to run to completion on one machine, and the caps make the peak footprint predictable for a matrix of this width. `psutil` prints the resident memory before and after the training loop, and the garbage collector is invoked after every classifier.

### 3.3 Perturbation probe (`DrugCombDB` only)

The `DrugCombDB` script trains on a deliberately corrupted copy of the training split and evaluates on the untouched test split. The corruption is composed of four independent perturbations:

| Perturbation          | Setting                                            | Purpose                                               |
| --------------------- | -------------------------------------------------- | ----------------------------------------------------- |
| Gaussian noise        | `sigma = 0.05` on every feature                  | tolerance to measurement noise in the embeddings      |
| Salt-and-pepper noise | probability 0.1, amplitude `5 x sigma`           | tolerance to single-feature outliers                  |
| Feature shuffling     | 30 % of the rows get their feature blocks permuted | insensitivity to the order of the concatenated blocks |
| Feature dropout       | probability 0.2, applied as a multiplicative mask  | reliance on a few dominant feature blocks             |

The probe answers a design question rather than a performance question: it shows which classifier families keep their ranking when the representation is partly destroyed, and therefore which of them exploit the representation rather than the training protocol. The test split is never perturbed.

### 3.4 Metrics and outputs

The metric set is identical in both folders and is defined once, in`compute_metrics`:

| Key       | Meaning                                                    |
| --------- | ---------------------------------------------------------- |
| `Acc`   | accuracy                                                   |
| `Sen`   | sensitivity (recall)                                       |
| `Spe`   | specificity, computed from the confusion matrix counts     |
| `Pre`   | precision                                                  |
| `Mcc`   | Matthews correlation coefficient                           |
| `AUROC` | area under the ROC curve, from the predicted probabilities |
| `AUPRC` | average precision, from the predicted probabilities        |
| `F1`    | F1 score                                                   |

The discrete metrics use the default decision threshold of 0.5, the ranking metrics use the predicted probability of the positive class, and the class balance is 1:1 by construction, so the random-guess line of the precision-recall plot coincides with the positive ratio of the test split.

The outputs of a run are written to `benchmark_results/`:

| Artefact                         | Content                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------- |
| `classifier_performance.csv`   | one row per dataset and classifier, all eight metrics (`DrugCombDB`)           |
| `<dataset>_metrics.csv`        | the same metrics in classifier-per-column layout (`SynDrugMD KG`)              |
| `<dataset>_ROC_PRC_curves.png` | ROC and precision-recall curves of all classifiers in one panel (`DrugCombDB`) |
| `<dataset>_curve_points.csv`   | the sampled points of both curves (`DrugCombDB`)                               |
| `<dataset>_radar.png`          | radar chart of the eight metrics (`SynDrugMD KG`)                              |
| `<dataset>_curves.png`         | ROC and precision-recall panels side by side (`SynDrugMD KG`)                  |
| console summary                  | the metric table and the classifier with the highest AUROC                       |

The `SynDrugMD KG` script additionally loads the validation split and reports its size and class balance. It is the split that any future threshold selection or hyper-parameter search has to use, so that the test split stays untouched.

---

## 4. Reproduction

```bash
cd "performance/diffierent_classifical/DrugCombDB"
python fea.py
python classifier.py

cd "../SynDrugMD KG"
python fea.py
python classifier.py
```

---

## 5. What is committed and what is not

| Item                                                        | Status                                                                             |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `fea.py`, `classifier.py` of both folders               | committed                                                                          |
| `embeddings/`, `smiles/`, `proteins/` or `targets/` | committed; these are the pre-trained inputs and the evidence of the feature layout |
| `feature_batches/dataset.h5`                              | not committed; regenerated by `fea.py`, large                                    |
| `feature_batches/dataset_metadata.json`                   | not committed; regenerated by `fea.py`                                           |
| `benchmark_results/`                                      | not committed; regenerated by `classifier.py`                                    |

The small `entity_mapping.csv` and `relation_mapping.csv` files are committed on purpose: they document the identifier layout of the graph and are enough to verify that the feature construction reads the same identifiers that the mapping files describe.
