# `legacy_published_workflow/` -- the original published workflow (read-only reference)

This directory is the code and the pre-trained assets of the **originally
published** KGP-DC workflow. It is kept unchanged as a read-only reference so
that the published pipeline can be inspected and re-executed. It is **not** the
source of any number reported in the revised manuscript.

## Contents

| Path | Origin | Purpose |
|---|---|---|
| `feature.py` | original | Row-level feature construction: the drug pair is expanded into the full Cartesian product of its target combinations (about 13 x 13 = 169 rows per pair on SynDrugMD), and the 3,840-dimensional pair matrix is written as HDF5 batches. |
| `model.py` | original | XGBoost training and evaluation on the expanded rows, with the grid-searched parameter set of the published paper. |
| `smiles/get smiles features.py` | original | ChemBERTa feature extraction over the SMILES strings of the dataset. |
| `smiles/get id.py` | original | Drug name / CID / SMILES mapping for the SMILES feature extraction. |
| `targets/get target features.py` | original | ProtT5 feature extraction over the target amino-acid sequences. |
| `targets/get id.py` | original | Target identifier and sequence mapping for the target feature extraction. |
| `embeddings/` | original | TransE entity and relation embeddings together with the mapped triple set of the full knowledge graph. |
| `smiles/smiles_features.npy`, `targets/target_features.npy` | original | The extracted ChemBERTa (768-dim) and ProtT5 (1,024-dim) feature matrices. |

The metrics that this workflow produced are kept separately in
`../reference_results/`.

## Why this directory is retained

The reviewers asked for the scripts that reproduce the published dataset
expansion and the published benchmarking, and for the knowledge-graph
construction code. This directory is the answer to that request.

## Why it is not used for the reported results

The published workflow expands a drug pair into its target combinations
**before** the train / validation / test split and then splits at the row
level. A drug pair therefore appears in several splits through different
target combinations, and the same pair can appear in the training set and in
the test set at the same time. The metrics that this produces are the ones
reported in the original manuscript (accuracy 0.9768, AUROC 0.9968) and they
must not be read as a leak-free evaluation.

The leak-free, pair-level protocol, the five-method comparison and the
statistical evidence are in `../baseline_experiments/`; see `../../README.md`
for the full discussion.

## Reference pipeline

```bash
cd performance/legacy_published_workflow

# 1. knowledge graph and its embedding (see also ../../triples/)
cd ../../triples
python "create triples.py"
python "get embeddings.py"

# 2. pre-trained features
cd ../performance/legacy_published_workflow
python "smiles/get id.py"
python "smiles/get smiles features.py"
python "targets/get id.py"
python "targets/get target features.py"

# 3. dataset expansion and the classifier
python feature.py
python model.py
```

Step 3 needs a large amount of scratch space: the expansion produces roughly
8.0 x 10^6 positive rows and about 1.6 x 10^7 rows in total at 3,840
dimensions, and the repository therefore does not distribute the generated
batches.
