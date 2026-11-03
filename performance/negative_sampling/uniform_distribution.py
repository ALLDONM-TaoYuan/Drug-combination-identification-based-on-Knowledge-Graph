"""Uniform random negative sampling."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BLOCK_NAME = "uniform_distribution"


def sample(ctx, candidates, needed, rng):
    if len(candidates) < needed:
        raise RuntimeError(f"not enough candidates: {len(candidates)} < {needed}")
    chosen = np.sort(rng.choice(len(candidates), size=needed, replace=False))
    return candidates[chosen], {}


if __name__ == "__main__":
    import main

    main.prepare([BLOCK_NAME])
