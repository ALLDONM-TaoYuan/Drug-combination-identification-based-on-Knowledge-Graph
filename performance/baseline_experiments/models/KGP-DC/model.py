from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common

NUM_BOOST_ROUND: int = 800
EARLY_STOPPING_ROUNDS: int = 60
FEATURE_DIM: int = 3840

BASE_PARAMS: dict = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_bin": 256,
    "gamma": 0.0,
    "verbosity": 0,
}

DEFAULT_CONFIG: dict = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 10.0,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
}

TUNING_CANDIDATES: list[dict] = [
    {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
     "colsample_bytree": 0.5, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9,
     "colsample_bytree": 0.3, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.1},
    {"max_depth": 6, "learning_rate": 0.08, "subsample": 0.8,
     "colsample_bytree": 0.4, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"max_depth": 4, "learning_rate": 0.08, "subsample": 0.8,
     "colsample_bytree": 0.4, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"max_depth": 4, "learning_rate": 0.08, "subsample": 1.0,
     "colsample_bytree": 0.6, "min_child_weight": 1.0,
     "reg_lambda": 1.0, "reg_alpha": 0.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9,
     "colsample_bytree": 0.5, "min_child_weight": 5.0,
     "reg_lambda": 5.0, "reg_alpha": 0.0},
    {"max_depth": 3, "learning_rate": 0.10, "subsample": 0.9,
     "colsample_bytree": 0.5, "min_child_weight": 5.0,
     "reg_lambda": 5.0, "reg_alpha": 0.0},
    {"max_depth": 3, "learning_rate": 0.08, "subsample": 0.8,
     "colsample_bytree": 0.4, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"max_depth": 2, "learning_rate": 0.08, "subsample": 0.8,
     "colsample_bytree": 0.4, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
     "colsample_bytree": 0.5, "min_child_weight": 10.0,
     "reg_lambda": 10.0, "reg_alpha": 0.0},
]

_FEATURE_CACHE = common.WORK_DIR / "fold_features"
_ASSETS = None

def _assets():
    global _ASSETS
    if _ASSETS is not None:
        return _ASSETS
    smiles = np.load(common.WARM_DIR / "drug_smiles_features.npy").astype(np.float32)
    targets = np.load(common.WARM_DIR / "target_features.npy").astype(np.float32)
    n_drugs = common.n_drugs()
    n_targets = common.n_targets()
    index = np.load(common.WARM_DIR / "drug_target_index.npy")
    mask = index["target_index"] != n_targets
    sums = np.zeros((n_drugs, targets.shape[1]), dtype=np.float64)
    counts = np.zeros(n_drugs, dtype=np.float64)
    np.add.at(sums, index["drug_id"][mask].astype(np.int64),
              targets[index["target_index"][mask].astype(np.int64)].astype(np.float64))
    np.add.at(counts, index["drug_id"][mask].astype(np.int64), 1.0)
    mean_targets = np.zeros((n_drugs, targets.shape[1]), dtype=np.float32)
    has = counts > 0
    mean_targets[has] = (sums[has] / counts[has, None]).astype(np.float32)
    _ASSETS = (smiles, mean_targets, n_drugs)
    return _ASSETS

def _train_transe(fold: int) -> np.ndarray:
    import pykeen.losses as pykeen_losses
    import pykeen.models as pykeen_models
    import torch
    from pykeen.sampling import BasicNegativeSampler
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    train = common.load_fold_frame(fold, "train")
    val = common.load_fold_frame(fold, "val")
    test = common.load_fold_frame(fold, "test")
    dti = common.load_dti()

    def combination(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
        pairs = frame.loc[frame["label"] == 1, ["drug1_id", "drug2_id"]].to_numpy()
        return {(f"d{int(a)}", "combination", f"d{int(b)}") for a, b in pairs}

    train_triples = combination(train)
    held_out = combination(val) | combination(test)
    interaction = {(f"d{int(d)}", "interaction", f"t{int(t)}")
                   for d, t in dti[["drug_id", "target_id"]].to_numpy()}
    graph = sorted(train_triples | interaction)

    factory = TriplesFactory.from_labeled_triples(np.asarray(graph, dtype=object),
                                                 create_inverse_triples=False)
    filter_rows = set(graph) | held_out
    mapped = [(factory.entity_to_id[h], factory.relation_to_id[r], factory.entity_to_id[t])
              for h, r, t in filter_rows
              if h in factory.entity_to_id and t in factory.entity_to_id
              and r in factory.relation_to_id]
    sampler = BasicNegativeSampler(
        mapped_triples=torch.as_tensor(mapped, dtype=torch.long), filtered=True,
        num_entities=factory.num_entities, num_relations=factory.num_relations,
        num_negs_per_pos=10)
    torch.manual_seed(42)
    model = pykeen_models.TransE(triples_factory=factory, embedding_dim=128,
                                 random_seed=42, loss=pykeen_losses.SoftplusLoss())
    loop = SLCWATrainingLoop(model=model, triples_factory=factory, optimizer="Adam",
                             optimizer_kwargs={"lr": 1e-3}, negative_sampler=sampler)
    loop.train(triples_factory=factory, num_epochs=50, batch_size=256, use_tqdm=False)
    embeddings = (model.entity_representations[0]()
                  .detach().cpu().numpy().astype(np.float32))

    n_drugs = common.n_drugs()
    drug_kg = np.zeros((n_drugs, 128), dtype=np.float32)
    for drug_id in range(n_drugs):
        row = factory.entity_to_id.get(f"d{drug_id}")
        if row is not None:
            drug_kg[drug_id] = embeddings[row]
    return drug_kg

def _pair_matrix(frame: pd.DataFrame, kg: np.ndarray, smiles: np.ndarray,
                 mean_targets: np.ndarray) -> np.ndarray:
    d1 = frame["drug1_id"].to_numpy(dtype=np.int64)
    d2 = frame["drug2_id"].to_numpy(dtype=np.int64)
    matrix = np.empty((len(frame), FEATURE_DIM), dtype=np.float32)
    matrix[:, 0:128] = kg[d1]
    matrix[:, 128:896] = smiles[d1]
    matrix[:, 896:1920] = mean_targets[d1]
    matrix[:, 1920:2048] = kg[d2]
    matrix[:, 2048:2816] = smiles[d2]
    matrix[:, 2816:3840] = mean_targets[d2]
    return matrix

def fold_features(fold: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    _FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    paths = {split: _FEATURE_CACHE / f"fold{fold}_{split}.npy"
             for split in common.SPLITS}
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if all(path.exists() for path in paths.values()):
        for split, path in paths.items():
            frame = common.load_fold_frame(fold, split)
            output[split] = (np.load(path, mmap_mode="r"),
                             frame["label"].to_numpy(dtype=np.int32))
        return output

    smiles, mean_targets, _ = _assets()
    kg = _train_transe(fold)
    for split, path in paths.items():
        frame = common.load_fold_frame(fold, split)
        matrix = _pair_matrix(frame, kg, smiles, mean_targets)
        np.save(path, matrix)
        output[split] = (matrix, frame["label"].to_numpy(dtype=np.int32))
    return output

def _drug_disjoint_halves(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    drugs = np.unique(np.concatenate([frame["drug1_id"].to_numpy(),
                                      frame["drug2_id"].to_numpy()]))
    rng = np.random.RandomState(int(seed))
    chosen = set(rng.choice(drugs, size=max(1, len(drugs) // 2), replace=False).tolist())
    d1 = frame["drug1_id"].to_numpy()
    d2 = frame["drug2_id"].to_numpy()
    mask_a = np.asarray([(a in chosen) and (b in chosen) for a, b in zip(d1, d2)])
    mask_b = np.asarray([(a not in chosen) and (b not in chosen) for a, b in zip(d1, d2)])
    if mask_a.sum() < 200 or mask_b.sum() < 200:
        index = rng.permutation(len(frame))
        mask_a = np.zeros(len(frame), dtype=bool)
        mask_a[index[: len(frame) // 2]] = True
        mask_b = ~mask_a
    return mask_a, mask_b

def train_and_predict(fold: int, seed: int = 42, config: dict | None = None,
                      write: bool = True) -> dict:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    started = time.time()
    if config is None:
        config = common.load_best_config("KGP-DC")
    settings = {**DEFAULT_CONFIG, **(config or {})}
    subsample = settings.pop("train_subsample", None)
    features = fold_features(fold)
    (x_train, y_train), (x_val, y_val), (x_test, _) = (
        features["train"], features["val"], features["test"])
    if subsample:
        keep = common.subsample_indices(y_train, subsample, seed)
        x_train, y_train = x_train[keep], y_train[keep]

    params = {**BASE_PARAMS, **settings, "seed": int(seed),
              "eval_metric": ["logloss", "auc", "aucpr"]}
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_test)
    booster = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")], verbose_eval=False,
        callbacks=[xgb.callback.EarlyStopping(
            rounds=EARLY_STOPPING_ROUNDS, metric_name="aucpr", data_name="val",
            maximize=True, save_best=False)])
    best = int(getattr(booster, "best_iteration", NUM_BOOST_ROUND - 1)) + 1
    val_raw = booster.predict(dval, iteration_range=(0, best))
    test_raw = booster.predict(dtest, iteration_range=(0, best))

    val_frame = common.load_fold_frame(fold, "val")
    mask_a, mask_b = _drug_disjoint_halves(val_frame, seed)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        np.clip(val_raw[mask_a], 1e-6, 1 - 1e-6), y_val[mask_a])
    val_probability = np.clip(calibrator.predict(np.clip(val_raw, 1e-6, 1 - 1e-6)),
                              1e-6, 1 - 1e-6)
    test_probability = np.clip(calibrator.predict(np.clip(test_raw, 1e-6, 1 - 1e-6)),
                               1e-6, 1 - 1e-6)
    threshold = float(common.youden_threshold(y_val[mask_b], val_probability[mask_b]))

    if write:
        test_frame = common.load_fold_frame(fold, "test")
        frame = test_frame[["pair_id"]].copy()
        frame.insert(0, "model", common.DISPLAY["KGP-DC"])
        frame.insert(1, "fold", int(fold))
        frame.insert(2, "seed", int(seed))
        frame["true_label"] = test_frame["label"].to_numpy()
        frame["predicted_label"] = (test_probability >= threshold).astype(int)
        frame["predicted_probability"] = test_probability
        common.save_predictions("KGP-DC", fold, seed, frame)

    return {"val_probability": val_probability, "test_probability": test_probability,
            "threshold": threshold,
            "meta": {"profile": "faithful", "best_iteration": best,
                     "train_pairs": int(len(x_train)), "feature_dim": int(x_train.shape[1]),
                     "config": settings, "elapsed_sec": round(time.time() - started, 1),
                     "notes": "per-fold TransE (train positives only) + mean ProtT5"}}
