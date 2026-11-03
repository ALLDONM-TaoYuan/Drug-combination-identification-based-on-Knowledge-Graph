"""Mean Euclidean distance negative sampling."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import settings as S

BLOCK_NAME = "euclidean_distance"


def sample(ctx, candidates, needed, rng):
    positives = np.asarray(ctx["positives"]["train"], dtype=np.int64)
    blocks = ctx["blocks"]
    pos_left = blocks[positives[:, 0]]
    pos_right = blocks[positives[:, 1]]
    mean1, std1 = pos_left.mean(0), np.sqrt(np.maximum(pos_left.var(0), 1e-8))
    mean2, std2 = pos_right.mean(0), np.sqrt(np.maximum(pos_right.var(0), 1e-8))
    left = (((blocks - mean1) / std1) ** 2).sum(1)
    right = (((blocks - mean2) / std2) ** 2).sum(1)
    distance = 0.5 * (np.sqrt(left[candidates[:, 0]] + right[candidates[:, 1]])
                      + np.sqrt(left[candidates[:, 1]] + right[candidates[:, 0]]))
    aed = float(distance.mean())
    threshold = S.EUCLIDEAN_THRESHOLD_MULTIPLIER * aed
    eligible = np.flatnonzero(distance > threshold)
    if eligible.size < needed:
        raise RuntimeError(f"not enough eligible candidates: {eligible.size} < {needed}")
    chosen = np.sort(rng.choice(eligible, size=needed, replace=False))
    return candidates[chosen], {"aed": aed, "threshold": threshold, "eligible": int(eligible.size)}


if __name__ == "__main__":
    import main

    main.prepare([BLOCK_NAME])
