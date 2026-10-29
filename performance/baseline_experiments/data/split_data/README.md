# `data/split_data/` -- the pair-level dataset

Everything in this directory describes the **drug pair** as the unit of
evaluation. No row-level target-combination expansion is stored here.

## Version controlled

| File | Content |
|---|---|
| `{train,val,test}_pair_meta.csv` | `drug1_id`, `drug2_id`, `label`, `source` for every pair of the warm 60/20/20 split. |
| `folds/fold_{1..5}_{train,val,test}.csv` | The pair-level stratified 5-fold cross-validation; `pair_id`, `drug1_id`, `drug2_id`, `label`, `source`. |
| `folds/folds_manifest.json` | Fold sizes, positive/negative counts and the seed. |
| `drug_table.csv` | `internal_id`, drug name, CID, SMILES availability. |
| `target_table.csv` | `internal_id`, target id, sequence availability. |
| `warm_split_summary.json` | Split sizes, the positive/negative counts and the knowledge-graph policy. |
| `rebuild_summary.json` | The record of the negative-sampling rebuild: policy, pool size, per-stratum quotas. |
| `{train,val,test}_build_summary.json` | Per-split build records. |

## Not version controlled

The following arrays are large and are therefore **not** in the repository.
They are part of the experiment workspace on the machine on which the
experiment was executed.

| File | Size | Content |
|---|---|---|
| `train.npy` | about 0.9 GB | `(60862, 3840)` float32 pair-level feature matrix |
| `val.npy` | about 0.3 GB | `(19590, 3840)` float32 pair-level feature matrix |
| `test.npy` | about 0.3 GB | `(19614, 3840)` float32 pair-level feature matrix |
| `drug_smiles_features.npy` | about 10 MB | `(3496, 768)` ChemBERTa drug features |
| `target_features.npy` | about 6 MB | `(1620, 1024)` ProtT5 target features |
| `drug_target_index.npy` | about 0.2 MB | drug to target index |
| `drug_ids.npy`, `target_ids.npy` | small | identifier orderings |
| `drug_kg_features.npy` | about 1.8 MB | train-only TransE drug vectors, also readable from `$KGP_DC_KG_FEATURES` |
| `{train,val,test}_labels.npy`, `{train,val,test}_sample_weight.npy` | small | label and weight arrays derived from the pair metadata |

The reason is the 100 MB per-file limit of the Git hosting service: the
training matrix alone is about 0.9 GB. The commit contains everything that is
needed to **audit** the experiment -- the pair membership, the labels and the
fold definition -- and the drug-level feature matrices, which are the actual
model inputs, are read from the experiment workspace.

## Reproducing the pairs

The pair list is fully determined by the committed metadata plus the
knowledge-graph availability filter:

```bash
cd performance/baseline_experiments
python data/negative_rebuild.py --apply      # rewrites the pair metadata and the arrays
python data/split_rules.py --folds           # rebuilds folds/fold_*.csv
python main.py check                         # validates the split and the folds
```

`data/split_rules.py` documents the exact layout and the rules, and
`data/negative_rebuild.py` documents the negative-sampling strategy and its
audit (50,033 unique negatives for 50,033 positive pairs, disjoint across
the three splits and disjoint from every known positive pair).
