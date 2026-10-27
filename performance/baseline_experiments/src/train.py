from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common

def load_model_module(model_key: str):
    path = common.MODELS_DIR / common.MODEL_SPECS[model_key]["dir"] / "model.py"
    if not path.exists():
        raise FileNotFoundError(f"missing model implementation: {path}")
    module_name = f"baseline_{common.model_slug(model_key)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_one(model_key: str, module, fold: int, seed: int) -> dict:
    started = time.time()
    common.set_seed(seed)
    result = module.train_and_predict(int(fold), int(seed))
    elapsed = time.time() - started

    test_frame = common.load_fold_frame(fold, "test")
    probabilities = np.asarray(result["test_probability"], dtype=np.float64)
    if len(probabilities) != len(test_frame):
        raise ValueError(f"{model_key} fold{fold}: {len(probabilities)} probabilities "
                         f"vs {len(test_frame)} test pairs")
    threshold = float(result["threshold"])

    labels = test_frame["label"].to_numpy(dtype=np.int32)
    prediction = test_frame[["pair_id"]].copy()
    prediction.insert(0, "model", common.DISPLAY[model_key])
    prediction.insert(1, "fold", int(fold))
    prediction.insert(2, "seed", int(seed))
    prediction["true_label"] = labels
    prediction["predicted_label"] = (probabilities >= threshold).astype(np.int32)
    prediction["predicted_probability"] = probabilities
    path = common.save_predictions(model_key, int(fold), int(seed), prediction)

    metrics = common.binary_metrics(labels, probabilities, threshold)
    meta = result.get("meta", {}) or {}
    record = {
        "model_key": model_key, "model": common.DISPLAY[model_key],
        "fold": int(fold), "seed": int(seed), "threshold": threshold,
        "elapsed_sec": round(elapsed, 1),
        "profile": json.dumps(meta, ensure_ascii=False, default=str),
        "pairs": metrics["pairs"], "positives": metrics["positives"],
        "negatives": metrics["negatives"], "tp": metrics["tp"], "fp": metrics["fp"],
        "tn": metrics["tn"], "fn": metrics["fn"],
        **{metric: metrics[metric] for metric in common.METRIC_ORDER},
    }
    common.append_run_record(record)
    print(f"[done] {model_key} fold={fold} seed={seed} {elapsed:.1f}s "
          f"AUROC={metrics['AUROC']:.4f} AUPRC={metrics['AUPRC']:.4f} "
          f"ACC={metrics['ACC']:.4f} -> {path.name}", flush=True)
    return record

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(common.MODEL_KEYS),
                        choices=list(common.MODEL_KEYS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(common.FOLDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    validation = common.validate_splits()
    folds = common.split_rules().validate_folds()
    if not validation["passed"] or not folds["passed"]:
        failed = [item for item in validation["checks"] + folds["checks"]
                  if not item["passed"]]
        raise SystemExit(f"split validation failed: {failed}")

    records = []
    for model_key in args.models:
        module = load_model_module(model_key)
        for fold in args.folds:
            if args.skip_existing and common.prediction_path(
                    model_key, fold, args.seed).exists():
                print(f"[skip] {model_key} fold={fold} seed={args.seed}")
                continue
            records.append(run_one(model_key, module, int(fold), int(args.seed)))
    print(json.dumps({"runs": len(records)}, indent=2))

if __name__ == "__main__":
    main()
