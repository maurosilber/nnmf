"""Adaptive binning (binlets) of spectral images before unmixing.

Spectra H are assumed known (or estimated beforehand, e.g. by blind NMF on
the whole image). Binlets merges neighbouring bins while their fractions
f = W / N are statistically compatible, according to the error model.
For camera data, X are photoelectrons and background is the read noise
variance per pixel and channel (Camera.background).
"""

from collections.abc import Callable

import numpy as np
import scipy.stats
from binlets import binlets
from numpy.typing import NDArray

from .error_model import chi2_statistic, cov_fractions
from .spectra import fractions
from .updates import unmix


def pixels_per_bin(ndim: int, level: int) -> int:
    """Pixels summed in each of the two bins binlets compares.

    The unnormalized Haar approximation at a level sums 2**(ndim * (level+1))
    pixels, and the test compares its two halves.
    """
    return 2 ** (ndim * (level + 1) - 1)


def compatibility_test(
    H: NDArray,
    *,
    alpha: float = 1e-3,
    pseudocount: float = 0.5,
    background: float = 0.0,
    **unmix_kwargs,
) -> Callable[[NDArray, NDArray, int], NDArray]:
    """Binlets test: True where bins x and y have compatible fractions.

    Under the null hypothesis both bins share the fractions f, estimated
    from the pooled photons x + y. Then

        (f_x - f_y)^T [C(f, N_x) + C(f, N_y)]^+ (f_x - f_y) ~ chi2(K - 1)

    with C(f, N) the bound on the covariance of the fractions of a bin
    with N photons. Without read noise C(f, N) = C(f) / N, so the sum is
    C(f) (1/N_x + 1/N_y). With read noise (background, in photoelectrons²
    per pixel and channel), a bin summing n pixels has background n times
    larger, so C(f, N) no longer scales as 1/N.

    The pseudocount keeps C(f) regular when a fraction is ~0,
    where the linear error model underestimates the error.

    x and y are channel-first, (wavelength, *spatial), as binlets passes them.
    """
    threshold = scipy.stats.chi2.ppf(1 - alpha, df=len(H) - 1)

    def estimate(x: NDArray, background: float, pseudocount: float = 0.0):
        W = unmix(np.moveaxis(x, 0, -1), H, background=background, **unmix_kwargs)
        return fractions(W, pseudocount)

    def test(x: NDArray, y: NDArray, level: int) -> NDArray:
        b = background * pixels_per_bin(x.ndim - 1, level)
        (f_x, N_x), (f_y, N_y) = (estimate(x, b), estimate(y, b))
        f, _ = estimate(x + y, 2 * b, pseudocount)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = cov_fractions(H, f, N_x, background=b) + cov_fractions(
                H, f, N_y, background=b
            )
        chi2 = chi2_statistic(f_x - f_y, cov)

        # Empty bins carry no information: merge them.
        empty = (N_x <= 0) | (N_y <= 0) | ~np.isfinite(chi2)
        return empty | (chi2 < threshold)

    return test


def binlets_unmix(
    X: NDArray,
    H: NDArray,
    *,
    levels: int | None = None,
    alpha: float = 1e-3,
    pseudocount: float = 0.5,
    background: float = 0.0,
    **unmix_kwargs,
) -> tuple[NDArray, NDArray]:
    """Denoise X (*spatial, wavelength) with binlets and unmix it.

    The non-linear Haar reconstruction can produce small negative counts,
    which make the KL updates unstable (W < 0, fractions outside [0, 1]),
    so without background the denoised X is clipped at 0. With background,
    X are camera photoelectrons, negative values are expected, and clipping
    would bias the fractions; the Gaussian updates handle them. The final
    unmixing weights each pixel with the single-pixel background.

    Returns W (*spatial, components) and the denoised X.
    """
    test = compatibility_test(
        H, alpha=alpha, pseudocount=pseudocount, background=background, **unmix_kwargs
    )
    X_binned = binlets(np.moveaxis(X, -1, 0), levels=levels, test=test, linear=False)
    X_binned = np.moveaxis(X_binned, 0, -1)
    if not background:
        X_binned = np.clip(X_binned, 0, None)
    W = unmix(X_binned, H, background=background, **unmix_kwargs)
    return W, X_binned
