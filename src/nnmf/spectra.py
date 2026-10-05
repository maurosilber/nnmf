"""Spectra, fractions and simulated measurements.

Notation (same as manual.ipynb), wavelength last:
    X: (*pixels, wavelength) measured photons, X ~ Poisson(W @ H)
    W: (*pixels, components) photons per component
    H: (components, wavelength) spectra, each row sums to 1

Per pixel, N = W.sum() is the total photon count and f = W / N the
fraction of photons per component (a point in the simplex).
"""

import numpy as np
import scipy.stats
from numpy.typing import NDArray

WAVELENGTH = np.linspace(400, 700, 100)


def normalize(H: NDArray) -> NDArray:
    """Scale each row to sum 1."""
    return H / H.sum(-1, keepdims=True)


def gaussian_spectra(centers, width=50.0, wavelength=WAVELENGTH) -> NDArray:
    centers = np.asarray(centers, dtype=float)[:, None]
    return normalize(scipy.stats.norm(centers, width).pdf(wavelength))


def random_spectra(rng: np.random.Generator, shape) -> NDArray:
    """Uniform random spectra, used as blind initial guesses."""
    return normalize(rng.uniform(0, 1, shape))


def fractions(W: NDArray, pseudocount: float = 0.0) -> tuple[NDArray, NDArray]:
    """Fractions f and total photons N per pixel.

    The pseudocount is added to each component, pulling f away from the
    border of the simplex.
    """
    N = W.sum(-1)
    f = (W + pseudocount) / (N + W.shape[-1] * pseudocount)[..., None]
    return f, N


def measure(
    rng: np.random.Generator,
    F: NDArray,
    H: NDArray,
    N,
    pixels: int | None = None,
) -> NDArray:
    """Poisson photons of pixels with fractions F and N photons.

    If pixels is given, F is a single set of fractions repeated over them.
    """
    expected = N * F @ H
    if pixels is not None:
        expected = np.broadcast_to(expected, (pixels, expected.shape[-1]))
    return rng.poisson(expected).astype(float)
