from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
PYTHON = sys.executable

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common

def call(script: str, extra: list[str]) -> int:
    command = [PYTHON, str(SRC / script), *extra]
    print("$", " ".join(command))
    return subprocess.call(command, cwd=str(PROJECT_ROOT))

def check() -> int:
    validation = common.validate_splits()
    folds = common.split_rules().validate_folds()
    print("warm split:", "PASS" if validation["passed"] else "FAIL")
    print("5-fold split:", "PASS" if folds["passed"] else "FAIL")
    for item in validation["checks"] + folds["checks"]:
        if item["passed"]:
            continue
        if item.get("optional"):
            print(f"  [optional] {item['check']} -- {item['detail']}")
        else:
            print(f"  [FAIL] {item['check']} -- {item['detail']}")
    print("\nmodels:")
    for model_key in common.MODEL_KEYS:
        path = common.MODELS_DIR / common.MODEL_SPECS[model_key]["dir"] / "model.py"
        print(f"  [{'OK' if path.exists() else 'MISSING'}] {common.DISPLAY[model_key]}"
              f" -> {path.relative_to(PROJECT_ROOT)}")
    return 0 if validation["passed"] and folds["passed"] else 1

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=("check", "tune", "train", "evaluate", "plot", "all"))
    parser.add_argument("--models", nargs="+", default=list(common.MODEL_KEYS),
                        choices=list(common.MODEL_KEYS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(common.FOLDS))
    parser.add_argument("--seed", type=int, default=common.SEEDS[0])
    parser.add_argument("--fold", type=int, default=1, help="tuning fold")
    parser.add_argument("--subsample", type=int, default=20000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model_args = ["--models", *args.models]
    fold_args = ["--folds", *[str(fold) for fold in args.folds]]
    seed_args = ["--seed", str(args.seed)]

    if args.command == "check":
        code = check()
    elif args.command == "tune":
        code = call("tune.py", model_args + ["--fold", str(args.fold),
                                             "--seed", str(args.seed),
                                             "--subsample", str(args.subsample)])
    elif args.command == "train":
        extra = model_args + fold_args + seed_args
        if args.skip_existing:
            extra.append("--skip-existing")
        code = call("train.py", extra)
    elif args.command == "evaluate":
        code = call("evaluate.py", [])
    elif args.command == "plot":
        code = call("plot_results.py", fold_args + seed_args)
    else:
        code = check()
        if code == 0:
            code = call("train.py", model_args + fold_args + seed_args)
        if code == 0:
            code = call("evaluate.py", [])
        if code == 0:
            code = call("plot_results.py", fold_args + seed_args)
    raise SystemExit(code)

if __name__ == "__main__":
    main()
