"""Helpers to evaluate estimates against the simulated truth."""

import itertools
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def total_variation(H_est: NDArray, H: NDArray) -> NDArray:
    """Total variation distance between spectra, per component."""
    return np.abs(H_est - H).sum(-1) / 2


def match_components(H_est: NDArray, H: NDArray, fixed=None) -> list[int]:
    """Permutation of the estimated components that best matches H.

    Blind NMF recovers components in arbitrary order. Known (fixed)
    components keep their position.
    """
    K = len(H)
    fixed = np.flatnonzero(np.zeros(K, bool) if fixed is None else fixed)
    keeps_fixed = lambda p: all(p[i] == i for i in fixed)
    candidates = map(list, filter(keeps_fixed, itertools.permutations(range(K))))
    return min(candidates, key=lambda p: np.abs(H_est[p] - H).sum())


def rmse(x: NDArray, truth: NDArray, mask: NDArray | None = None) -> float:
    e = (x - truth) ** 2
    return float(np.sqrt(e.mean() if mask is None else e[mask].mean()))


def save_results(script: str, results) -> Path:
    """Save results as JSON in results/<script name>.json."""
    path = Path(script).resolve().parents[1] / "results" / f"{Path(script).stem}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=1, default=_to_builtin))
    return path


def _to_builtin(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))
