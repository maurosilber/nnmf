"""Binlets denoising before unmixing, K=2 with known spectra.

1. Calibration: on a uniform image, the level-0 test should reject a
   fraction alpha of neighbour pairs.
2. RMSE of f1 on a phantom with flat regions, edges and a smooth gradient,
   compared with no denoising (raw) and box filters.
"""

import itertools
import time
import warnings

import numpy as np
import scipy.ndimage

from nnmf import (
    binlets_unmix,
    compatibility_test,
    fractions,
    gaussian_spectra,
    measure,
    rmse,
    save_results,
    unmix,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

H = gaussian_spectra([500, 600])
SIZE = 128
CALIBRATION_PHOTONS = [10, 30, 100, 1000]
PHOTONS = [20, 100, 1000]


def phantom(n: int = SIZE) -> np.ndarray:
    y, x = np.mgrid[:n, :n] / n
    f1 = np.full((n, n), 0.2)
    f1[(x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.15**2] = 0.7
    f1[(abs(x - 0.72) < 0.12) & (abs(y - 0.3) < 0.12)] = 0.45
    band = y > 0.6
    f1[band] = 0.1 + 0.8 * x[band]  # smooth gradient
    f1[(y > 0.6) & (abs(x - 0.5) < 0.02)] = 0.0  # thin line with no component 1
    return np.stack([f1, 1 - f1], -1)


F = phantom()


def phantom_regions(F: np.ndarray) -> dict[str, np.ndarray]:
    n = len(F)
    y, x = np.mgrid[:n, :n] / n
    edges = (
        scipy.ndimage.morphological_gradient(np.round(F[..., 0], 2) * 100, size=3) > 0
    )
    gradient = ~edges & (y > 0.62) & (np.abs(x - 0.5) > 0.04)
    return {"flat": ~edges & ~gradient, "edges": edges, "gradient": gradient}


REGIONS = phantom_regions(F)


def rejection_rate(N: int, alpha: float = 0.05) -> dict:
    X = measure(rng, np.full((64, 64, 2), [0.3, 0.7]), H, N)
    X = np.moveaxis(X, -1, 0)
    compatible = compatibility_test(H, alpha=alpha)(X, np.roll(X, -1, axis=1))
    return {"N": N, "alpha": alpha, "rejection": 1 - compatible.mean()}


def box(size: int):
    return lambda X: unmix(
        scipy.ndimage.uniform_filter(X, size=(size, size, 1), mode="nearest"), H
    )


def binlets_method(alpha: float):
    return lambda X: binlets_unmix(X, H, alpha=alpha)[0]


METHODS = {
    "raw": lambda X: unmix(X, H),
    "box3": box(3),
    "box7": box(7),
    "binlets 0.2": binlets_method(0.2),
    "binlets 0.05": binlets_method(0.05),
    "binlets 0.01": binlets_method(0.01),
}


def timed(f, *args):
    start = time.perf_counter()
    return f(*args), time.perf_counter() - start


def run(N: int) -> dict:
    X = measure(rng, F, H, N)

    def evaluate(name: str) -> dict:
        W, seconds = timed(METHODS[name], X)
        f1 = fractions(W)[0][..., 0]
        errors = {region: rmse(f1, F[..., 0], mask) for region, mask in REGIONS.items()}
        return {"N": N, "method": name, "seconds": seconds, "rmse": errors, "f1": f1}

    return list(map(evaluate, METHODS))


def report(r: dict) -> str:
    errors = " ".join(f"{e:.4f}" for e in r["rmse"].values())
    return f"  {r['method']:13s}{errors}   ({r['seconds']:.1f}s)"


if __name__ == "__main__":
    calibration = list(map(rejection_rate, CALIBRATION_PHOTONS))
    print(
        "== calibration: uniform image f1=0.3, fraction of level-0 tests rejecting (expect alpha=0.05)"
    )
    print(
        "\n".join(
            f"  N={r['N']:5d} rejection={r['rejection']:.3f}" for r in calibration
        )
    )

    results = list(itertools.chain.from_iterable(map(run, PHOTONS)))
    print("== RMSE of f1 by region:", " | ".join(REGIONS))
    for N, rows in itertools.groupby(results, key=lambda r: r["N"]):
        print(f"N={N}")
        print("\n".join(map(report, rows)))

    maps = {f"{r['method']} N={r['N']}": r.pop("f1") for r in results}
    path = save_results(__file__, {"calibration": calibration, "rmse": results})
    np.savez_compressed(
        path.with_name("5_binlets_maps.npz"),
        truth=F[..., 0],
        **{f"region {k}": v for k, v in REGIONS.items()},
        **maps,
    )
