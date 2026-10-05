"""Semi-blind unmixing, K=3: how much does knowing a subset of the spectra
help, with and without nearly pure pixels?"""

import itertools
import warnings

import numpy as np

from nnmf import (
    cov_fractions,
    fit_path,
    gaussian_spectra,
    match_components,
    measure,
    random_spectra,
    save_results,
    total_variation,
)

warnings.filterwarnings("ignore")

H = gaussian_spectra([480, 550, 620])
N, PIXELS, REPETITIONS = 1000, 1000, 5
AT = (100, 300, 1000, 3000)
DATASETS = {1.0: "dirichlet(1): has near-pure pixels", 5.0: "dirichlet(5): mixed only"}
KNOWN = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]


def repetition(seed: int, concentration: float, known: np.ndarray):
    """Returns rmse/crb per (iteration, component) and the H error."""
    rng = np.random.default_rng(seed)
    F = rng.dirichlet([concentration] * 3, PIXELS)
    crb = np.sqrt(np.diagonal(cov_fractions(H, F, N), axis1=-2, axis2=-1))
    X = measure(rng, F, H, N)
    H0 = np.where(known[:, None], H, random_spectra(rng, H.shape))
    W0 = rng.uniform(0, 1, (PIXELS, 3))
    Ws, Hs = fit_path(W0, H0, X, at=AT, loss="kl", fixed_h=known)
    order = match_components(Hs[-1], H, fixed=known)
    f = Ws[..., order] / Ws.sum(-1, keepdims=True)
    return np.sqrt((((f - F) / crb) ** 2).mean(1)), total_variation(Hs[-1][order], H)


def run(concentration: float, known: list[int]) -> dict:
    known = np.array(known, bool)
    rmse, h_error = zip(
        *(repetition(seed, concentration, known) for seed in range(REPETITIONS))
    )
    return {
        "dataset": DATASETS[concentration],
        "known": known.astype(int),
        "iterations": AT,
        "H_TV": np.mean(h_error, 0),
        "rmse/crb": np.mean(rmse, 0),  # (iterations, components)
    }


def report(r: dict) -> str:
    return (
        f"{r['dataset']:36s} known={r['known']} H_TV={r['H_TV'].round(3)} "
        f"rmse/crb per comp at it {AT}: {r['rmse/crb'].round(2).tolist()}"
    )


if __name__ == "__main__":
    results = list(itertools.starmap(run, itertools.product(DATASETS, KNOWN)))
    print("\n".join(map(report, results)))
    save_results(__file__, results)
