# KGP-DC: Knowledge Graph-Powered Drug Combination Prediction

[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for predicting synergistic drug combinations by integrating knowledge graphs, pre-trained molecular features, and protein sequence embeddings.

## Overview

![Overall Pipeline](Overview.png)

## Description

The process consists of five main steps: 1. Collect drug-drug combination information and drug-target interaction information from various databases, such as SynDrugMD, as well as from the literature. 2. Further organize the dataset and remove redundant information based on the collected drug CIDs, drug database names, and target IDs. 3. Based on the collected information, construct a knowledge graph and convert the chemical structure (SMILES) information of drugs and the sequence information of target proteins. 4. Utilize the pre-trained models ChemBert and Prot_Bert to extract feature vectors from drug chemical structure information and corresponding target sequence information, respectively, while employing the TransE model for embedding learning to represent drug-drug combinations. 5. Build an XGBoost model to identify potential drug-drug combinations.

## Project Structure

```
SynDrugMD/
├── Database/                  # DDI and DTI datasets
│   ├── DDI.csv
│   └── DTI.csv
├── Triples/                   # KG triple construction & embedding
│   ├── create triples.py
│   ├── get embeddings.py
│   └── embeddings/
├── Proformance/               # Original feature engineering & model training
│   ├── feature.py             # Drug pair feature vector construction
│   ├── model.py               # XGBoost training & evaluation
│   ├── smiles/                # Drug SMILES & ChemBERTa features
│   ├── targets/               # Protein sequences & ProtT5 features
│   ├── embeddings/            # TransE KG embeddings
│   └── model_results/         # Trained model & metrics
├── new Performance/
│   └── cold_start/            # Single-cold drug-pair experiment
│       ├── main.py            # XGBoost training, selection, and plotting
│       ├── feature.py         # Feature assembly, split preparation, and indexes
│       ├── smiles.py
│       ├── target.py
│       ├── data/              # CSV inputs and retained NPY feature arrays
│       └── KG/                # KG builder, triples, and pretrained features
├── Similarity/                # Drug & combo similarity analysis
│   ├── morgan.py              # Morgan fingerprint extraction
│   ├── sim.py                 # Cosine similarity computation & visualization
│   ├── DB_id_smiles.csv       # DrugCombDB drug SMILES data
│   ├── Syn_id_smiles.csv      # SynDrugMD drug SMILES data
│   ├── syn_morgan_features/   # Pre-computed SynDrugMD Morgan fingerprints
│   ├── db_morgan_features/    # Pre-computed DrugCombDB Morgan fingerprints
│   └── complete_analysis.png  # Similarity heatmap & distribution plot
├── requirements.txt
├── LICENSE
└── README.md
```

## Dependencies

Use **Python 3.10 or 3.11**. Install all dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Key packages:

| Package      | Version   | Purpose                               |
| ------------ | --------- | ------------------------------------- |
| pykeen       | 1.10.2    | Knowledge graph embedding (TransE)    |
| xgboost      | ≥3.0,<4   | Drug combination classifier           |
| scikit-learn | 1.3.2     | Metrics & data splitting              |
| rdkit        | ≥2023.03 | Morgan fingerprint computation        |
| transformers | ≥4.30.0  | ChemBERTa & ProtT5 feature extraction |
| torch        | ≥2.0.0   | Deep learning backend                 |

## Usage

### 1. Build Knowledge Graph Triples & Embeddings

```bash
cd Triples/
python "create triples.py"
python "get embeddings.py"
```

### 2. Generate Feature Vectors & Train Model

```bash
cd ../Proformance/
python feature.py
python model.py
```

### 3. Drug Similarity Analysis

```bash
cd ../Similarity/
# Generate Morgan fingerprints (if not pre-computed)
python morgan.py

# Compute drug & combination similarity matrices
python sim.py
```

### 4. Single-Cold Drug-Pair Evaluation

The single-cold experiment evaluates a drug pair containing one training drug
and one unseen validation or test drug. It uses SMILES and target-sequence
features only; KG features are disabled for this branch.

The experiment follows these rules:

1. Drugs are assigned to train/validation/test pools using a fixed 60/20/20
   split with seed 42.
2. Training positives contain two training drugs. Validation and test pairs
   contain one held-out drug and one training drug.
3. Random unlabeled pairs are sampled at a 1:1 pair-level ratio and exclude all
   known positive pairs and self-pairs.
4. Target features are expanded as the full Cartesian product. Every
   target-combination row receives weight `1 / (n1 * n2)`, giving every parent
   drug pair total weight 1.
5. Predictions are aggregated to one score per drug pair before ROC-AUC and
   AUPR are calculated. Validation selects the boosting round; test labels do
   not participate in selection.

This branch retains the original model's full target-target Cartesian
expansion. The cold-start revision changes the statistical unit from individual
target-combination rows to drug pairs by weighting each row and aggregating the
predictions at pair level; it does not replace target combinations with mean
pooling.

Run the following commands from `new Performance/cold_start/`.

```powershell
# Build the single-cold pair split.
python feature.py prepare-single-cold --seed 42

# Expand each labeled pair into its weighted target-target combinations.
python feature.py build-single-cold-index

# Or run both preprocessing stages in sequence.
python feature.py prepare-single-cold-all --seed 42

# Validate retained inputs and generated indexes.
python main.py status

# Train the finalized CPU configuration.
python main.py train --branch cold `
  --combo-dir data/combo_index_single_cold `
  --run-name single_cold `
  --rounds 1000 --batch-size 512 --nthread 4 `
  --learning-rate 0.02 --max-depth 2 `
  --subsample 0.70 --colsample-bytree 0.65 `
  --reg-alpha 3 --reg-lambda 20 --gamma 0.8 `
  --min-child-weight 8 --early-stopping-rounds 40 `
  --selection-step 1 --device cpu --seed 42 --force
```

The program creates `new Performance/cold_start/results/<run-name>/`
automatically. Models, pair-level predictions, metrics, selection records,
training history, and performance figures are all written beneath this single
results directory. Generated results are ignored by Git.

Tabular inputs and generated drug-pair splits use CSV. Dense feature matrices
and numeric training indexes use NPY; the cold-start project does not mix NPY
and NPZ storage. The following deterministic outputs are treated as caches and
are ignored by Git:

- `data/split_single_cold/`
- `data/labeled_pairs/`
- `data/combo_index_single_cold/`
- `data/combo_index/`

The retained SMILES, ProtT5, and pretrained TransE arrays are model inputs, not
temporary caches. They remain in the project because reproducing them requires
the exact pretrained checkpoints and preprocessing environment used by the
experiment.

For CUDA 13 execution, install the optional CuPy dependency separately:

```powershell
python -m pip install "cupy-cuda13x[ctk]"
```

The two preprocessing stages are integrated into `feature.py` but remain
separate subcommands. `prepare-single-cold` creates reproducible labeled drug
pairs and held-out drug pools; `build-single-cold-index` converts every positive
and negative pair into weighted target-combination indexes. This separation
allows the split to be audited before the potentially large Cartesian expansion
is built. Use `prepare-single-cold-all` only when a one-command rebuild is
preferred.

The reusable TransE features are stored in `KG/pretrained/`. A KG vector is
enabled only for a training drug with `has_kg_relation=True`; held-out drugs and
untrained isolated entities are masked to zero. The finalized single-cold model
uses only SMILES and target features. If `python KG/kg_feature.py train` creates
a complete KG feature set directly under `KG/`, that rebuilt set takes priority
over `KG/pretrained/`.

## Similarity Analysis

![Similarity Analysis](Similarity/complete_analysis.png)

The `Similarity/` module evaluates the redundancy of drug features and drug combination representations across two benchmark datasets (**SynDrugMD** and **DrugCombDB**). The analysis proceeds as follows:

1. **Morgan Fingerprint Extraction** — Convert drug SMILES strings into 1024-bit Morgan fingerprints (ECFP4) using RDKit.
2. **Drug Similarity Matrix** — Compute the pairwise cosine similarity matrix for all individual drugs in each dataset.
3. **Drug Combination Features** — For each known drug pair (extracted from KG triples with `rel_id == 0`), concatenate the Morgan fingerprints of the two constituent drugs to form a 2048-dimensional combination feature vector.
4. **Combination Similarity Matrix** — Compute the pairwise cosine similarity matrix across all drug combination feature vectors using chunked computation for memory efficiency.
5. **Visualization** — Generate a 2×3 panel figure with:
   - **Top row (SynDrugMD)**: Drug-drug similarity heatmap, combination-combination similarity heatmap, and side-by-side similarity distribution histogram.
   - **Bottom row (DrugCombDB)**: Corresponding heatmaps and histogram for the second dataset.

The resulting heatmaps and histograms reveal the similarity distribution patterns of drugs and their combinations, providing insight into feature redundancy and dataset diversity.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use KGP-DC in your research, please cite our work:

```bibtex
@article{kgpdc2026,
  title={KGP-DC: A Framework for Predicting Drug Combinations by Integrating Knowledge Graphs and Pre-trained Features},
  author={Tao Yuan},
  journal={},
  year={2026}
}
```
