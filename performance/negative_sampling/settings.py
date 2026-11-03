"""Global configuration of the negative-sampling strategy comparison."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM_DIR = HERE.parent / "cold_start"
REPO_ROOT = UPSTREAM_DIR.parent.parent

DATA_ROOT = UPSTREAM_DIR / "entity_split_data"
FEATURE_DIR = DATA_ROOT / "features"
SPLIT_DIR = DATA_ROOT / "split" / "single"
KG_DIR = UPSTREAM_DIR / "kg_embedding" / "single" / "train"
FROZEN_DIR = UPSTREAM_DIR / "single_cold_start" / "features"
OUTPUT_ROOT = HERE / "process"

SPLITS = ("train", "val", "test")

BLOCKS = (
    {"name": "euclidean_distance", "strategy": "euclidean",
     "title": "Mean Euclidean distance negative sampling"},
    {"name": "frequency_matching", "strategy": "dti_matched",
     "title": "DTI frequency matching negative sampling"},
    {"name": "uniform_distribution", "strategy": "uniform",
     "title": "Uniform random negative sampling"},
)

MODE = "single"

SAMPLING_SEED = 42
NEGATIVE_RATIO = 1.0
EUCLIDEAN_THRESHOLD_MULTIPLIER = 0.9

SEEDS = (42, 52, 62)
ROUNDS = 4000
NTHREAD = 8
DEVICE = "cpu"
CALIBRATION = "isotonic"

XGB_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.005,
    "max_depth": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.1,
    "min_child_weight": 100.0,
    "reg_alpha": 5.0,
    "reg_lambda": 50.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": ["logloss", "auc", "aucpr"],
    "verbosity": 0,
}

METRICS = ("ACC", "SEN", "SPE", "PRE", "F1", "MCC", "ROC_AUC", "AUPR")


def get_block(name: str) -> dict:
    for block in BLOCKS:
        if block["name"] == name:
            return block
    raise KeyError(f"unknown block: {name}")


def block_features(name: str) -> Path:
    return OUTPUT_ROOT / name / "features"


def block_results(name: str) -> Path:
    return OUTPUT_ROOT / name / "results"
