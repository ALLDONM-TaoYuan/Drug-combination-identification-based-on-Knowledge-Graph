from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common
import train as train_module

SEARCH_SUBSAMPLE_BY_MODEL: dict[str, int] = {
    "KGP-DC": 20_000,
    "KGNN": 0,
    "GraRep": 0,
    "DeepTraSynergy": 0,
    "DeepDDS": 20_000,
}

def tune_one(model_key: str, fold: int, seed: int, subsample: int) -> dict:
    module = train_module.load_model_module(model_key)
    candidates = getattr(module, "TUNING_CANDIDATES", None)
    if not candidates:
        raise NotImplementedError(f"{model_key} does not define TUNING_CANDIDATES")
    val_labels = common.load_fold_frame(fold, "val")["label"].to_numpy()
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        config = dict(candidate)
        config["train_subsample"] = int(subsample) or None
        started = time.time()
        result = module.train_and_predict(int(fold), int(seed), config=config, write=False)
        metrics = common.binary_metrics(val_labels, result["val_probability"], 0.5)
        rows.append({"candidate": index, **candidate,
                     "val_AUROC": metrics["AUROC"], "val_AUPRC": metrics["AUPRC"],
                     "val_ACC_at_0.5": metrics["ACC"],
                     "elapsed_sec": round(time.time() - started, 1)})
        print(f"[tune {model_key} {index}/{len(candidates)}] "
              f"val_AUROC={metrics['AUROC']:.4f} val_AUPRC={metrics['AUPRC']:.4f} "
              f"({rows[-1]['elapsed_sec']}s)", flush=True)

    table = pd.DataFrame(rows).sort_values(["val_AUPRC", "val_AUROC"], ascending=False)
    table.to_csv(common.TUNING_DIR / f"{common.model_slug(model_key)}.csv", index=False)
    best = table.iloc[0].to_dict()
    best_config = {key: candidate for key, candidate in candidates[
        int(best["candidate"]) - 1].items()}
    payload = {"model_key": model_key, "fold": int(fold), "seed": int(seed),
               "subsample": int(subsample), "selection_metric": "val_AUPRC",
               "val_AUROC": float(best["val_AUROC"]), "val_AUPRC": float(best["val_AUPRC"]),
               "config": best_config}
    common.write_json(common.TUNING_DIR / f"{common.model_slug(model_key)}_best.json",
                      payload)
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(common.MODEL_KEYS),
                        choices=list(common.MODEL_KEYS))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample", type=int, default=None,
                        help="override the per-model search size (0 = full fold-1 train)")
    args = parser.parse_args()

    results = []
    for model_key in args.models:
        size = args.subsample if args.subsample is not None \
            else SEARCH_SUBSAMPLE_BY_MODEL.get(model_key, 0)
        results.append(tune_one(model_key, int(args.fold), int(args.seed), int(size)))
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
