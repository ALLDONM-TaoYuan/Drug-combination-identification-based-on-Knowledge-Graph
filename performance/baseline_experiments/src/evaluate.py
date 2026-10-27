from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", default=True,
                        help="fail when a prediction file is inconsistent with its threshold")
    args = parser.parse_args()

    records = common.load_run_records()
    rows, problems = [], []
    for _, record in records.iterrows():
        model_key = str(record["model_key"])
        fold = int(record["fold"])
        seed = int(record["seed"])
        threshold = float(record["threshold"])
        frame = common.load_predictions(model_key, fold, seed)
        labels = frame["true_label"].to_numpy(dtype=np.int32)
        probabilities = frame["predicted_probability"].to_numpy(dtype=np.float64)
        predicted = frame["predicted_label"].to_numpy(dtype=np.int32)

        recomputed = (probabilities >= threshold).astype(np.int32)
        mismatch = recomputed != predicted
        boundary = np.abs(probabilities - threshold) <= 1e-9
        hard_mismatch = mismatch & (~boundary)
        if hard_mismatch.any():
            problems.append(f"{model_key} fold={fold} seed={seed}: "
                            f"{int(hard_mismatch.sum())} rows disagree with "
                            f"threshold {threshold}")

        metrics = common.binary_metrics(labels, probabilities, threshold)
        row = record.to_dict()
        row.update({metric: metrics[metric] for metric in common.METRIC_ORDER})
        row.update({"threshold": threshold, "pairs": metrics["pairs"],
                    "positives": metrics["positives"], "negatives": metrics["negatives"],
                    "tp": metrics["tp"], "fp": metrics["fp"],
                    "tn": metrics["tn"], "fn": metrics["fn"],
                    "mismatch_rows": int(mismatch.sum()),
                    "boundary_rows": int(boundary.sum())})
        rows.append(row)

    if not rows:
        raise SystemExit("no runs found; run `python main.py train` first")

    evaluated = pd.DataFrame(rows)
    evaluated.to_csv(common.RUNS_CSV, index=False)
    table = common.write_summary_table(evaluated)

    columns = ["model"] + [common.METRIC_LABELS[metric] for metric in common.METRIC_ORDER]
    print(table[columns].to_string(index=False))
    print(f"\n-> {common.TABLE_CSV} (each cell is mean +/- std over the 5 folds)")
    if problems:
        for problem in problems:
            print("[inconsistent]", problem)
        if args.strict:
            raise SystemExit(1)

if __name__ == "__main__":
    main()
