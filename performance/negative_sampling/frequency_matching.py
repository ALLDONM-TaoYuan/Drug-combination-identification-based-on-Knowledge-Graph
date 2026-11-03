"""DTI frequency matching negative sampling."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import settings as S

BLOCK_NAME = "frequency_matching"


def sample(ctx, candidates, needed, rng):
    bff = ctx["bff"]
    counts = ctx["target_counts"]
    quota = Counter(bff.target_stratum(counts.get(a, 0), counts.get(b, 0))
                    for a, b in ctx["positives"]["train"])
    pool = np.asarray(sorted(ctx["pools"]["train"] & ctx["usable"]), dtype=np.int64)
    negatives, report = bff.sample_negatives_matched(
        set(pool.tolist()), set(pool.tolist()), quota, counts,
        ctx["sample_forbidden"], ctx["n_drugs"], rng)
    deficit = int(report.get("stratum_deficit", 0))
    if deficit != 0 or len(negatives) != needed:
        raise RuntimeError(f"stratified sampling failed: deficit={deficit}, n={len(negatives)}")
    return np.asarray(sorted(negatives), dtype=np.int64), {
        "per_stratum": report.get("per_stratum", {}),
        "stratum_deficit": deficit,
    }


if __name__ == "__main__":
    import main

    main.prepare([BLOCK_NAME])
