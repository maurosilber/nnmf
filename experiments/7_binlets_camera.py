"""Binlets on camera data (read noise), K=2 with known spectra.

1. Calibration per level: x and y sum n = pixels_per_bin(2, level) camera
   pixels with equal fractions; the test should reject a fraction alpha.
   Compared with the background of a single pixel (not scaled with n) and
   with the Poisson test on photoelectrons clipped at 0.
2. RMSE of f1 on the phantom of experiment 5, with camera noise.
"""

import importlib
import itertools
import time
import warnings

import numpy as np
import scipy.ndimage

from nnmf import (
    Camera,
    binlets_unmix,
    compatibility_test,
    fractions,
    measure,
    rmse,
    save_results,
    unmix,
)
from nnmf.binning import pixels_per_bin

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

phantom = importlib.import_module("5_binlets")
H, F, REGIONS = phantom.H, phantom.F, phantom.REGIONS

GAIN = 1.7

# 1. Calibration
LEVELS = [0, 1, 2, 3]
PHOTONS = [10, 100]  # per pixel
READ_NOISE = [1.0, 3.0]  # e⁻ per pixel and channel
ALPHA = 0.05
SIZE = 32


def binned(camera: Camera, N: int, n: int) -> np.ndarray:
    """(wavelength, SIZE, SIZE) sums of n camera pixels, in photoelectrons."""
    electrons = measure(rng, np.full((n, SIZE, SIZE, 2), [0.3, 0.7]), H, N)
    X = camera.to_electrons(camera.digitize(rng, electrons)).sum(0)
    return np.moveaxis(X, -1, 0)


def run_calibration(N: int, read_noise: float, level: int) -> dict:
    camera = Camera(gain=GAIN, read_noise=read_noise * GAIN)
    n = pixels_per_bin(2, level)
    x, y = binned(camera, N, n), binned(camera, N, n)
    test = lambda background: compatibility_test(H, alpha=ALPHA, background=background)
    tests = {
        "scaled": (test(camera.background), (x, y)),
        "pixel": (test(camera.background / n), (x, y)),
        "poisson": (test(0.0), np.clip((x, y), 0, None)),
    }
    rejection = {
        name: 1 - test(*xy, level=level).mean() for name, (test, xy) in tests.items()
    }
    return {"N": N, "read_noise": read_noise, "level": level, "pixels": n, **rejection}


def report_calibration(r: dict) -> str:
    return (
        f"  N={r['N']:4d} σ={r['read_noise']:.0f}e level={r['level']} "
        f"({r['pixels']:3d} px)  scaled={r['scaled']:.3f} "
        f"pixel={r['pixel']:.3f} poisson={r['poisson']:.3f}"
    )


# 2. RMSE
RMSE_PHOTONS = [20, 100, 1000]
RMSE_READ_NOISE = 2.0


def box(size: int):
    return lambda X, b: unmix(
        scipy.ndimage.uniform_filter(X, size=(size, size, 1), mode="nearest"),
        H,
        background=b,
    )


METHODS = {
    "raw": lambda X, b: unmix(X, H, background=b),
    "box3": box(3),
    "box7": box(7),
    "binlets 0.05": lambda X, b: binlets_unmix(X, H, alpha=0.05, background=b)[0],
    "binlets 0.01": lambda X, b: binlets_unmix(X, H, alpha=0.01, background=b)[0],
    "poisson binlets 0.01": lambda X, b: binlets_unmix(
        np.clip(X, 0, None), H, alpha=0.01
    )[0],
}


def run_rmse(N: int) -> list[dict]:
    camera = Camera(gain=GAIN, read_noise=RMSE_READ_NOISE * GAIN)
    X = camera.to_electrons(camera.digitize(rng, measure(rng, F, H, N)))

    def evaluate(name: str) -> dict:
        start = time.perf_counter()
        f1 = fractions(METHODS[name](X, camera.background))[0][..., 0]
        seconds = time.perf_counter() - start
        errors = {region: rmse(f1, F[..., 0], mask) for region, mask in REGIONS.items()}
        return {"N": N, "method": name, "seconds": seconds, "rmse": errors}

    return list(map(evaluate, METHODS))


def report_rmse(r: dict) -> str:
    errors = " ".join(f"{e:.4f}" for e in r["rmse"].values())
    return f"  {r['method']:21s}{errors}   ({r['seconds']:.1f}s)"


if __name__ == "__main__":
    calibration = list(
        itertools.starmap(
            run_calibration, itertools.product(PHOTONS, READ_NOISE, LEVELS)
        )
    )
    print(f"== calibration: rejection rate at alpha={ALPHA}, equal fractions")
    print("\n".join(map(report_calibration, calibration)))

    results = list(itertools.chain.from_iterable(map(run_rmse, RMSE_PHOTONS)))
    print(f"== RMSE of f1, read noise {RMSE_READ_NOISE} e⁻:", " | ".join(REGIONS))
    for N, rows in itertools.groupby(results, key=lambda r: r["N"]):
        print(f"N={N}")
        print("\n".join(map(report_rmse, rows)))

    save_results(__file__, {"calibration": calibration, "rmse": results})
