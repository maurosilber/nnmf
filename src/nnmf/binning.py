"""Adaptive binning (binlets) of spectral images before unmixing.

Spectra H are assumed known (or estimated beforehand, e.g. by blind NMF on
the whole image). Binlets merges neighbouring bins while their fractions
f = W / N are statistically compatible, according to the error model.
"""

from collections.abc import Callable

import numpy as np
import scipy.stats
from binlets import binlets
from numpy.typing import NDArray

from .error_model import chi2_statistic, cov_fractions
from .spectra import fractions
from .updates import unmix


def compatibility_test(
    H: NDArray,
    *,
    alpha: float = 1e-3,
    pseudocount: float = 0.5,
    **unmix_kwargs,
) -> Callable[[NDArray, NDArray], NDArray]:
    """Binlets test: True where bins x and y have compatible fractions.

    Under the null hypothesis both bins share the fractions f, estimated
    from the pooled photons x + y. Then

        (f_x - f_y)^T [C(f) (1/N_x + 1/N_y)]^+ (f_x - f_y) ~ chi2(K - 1)

    with C(f) / N the Cramér-Rao covariance of the fractions.
    The pseudocount keeps C(f) regular when a fraction is ~0,
    where the linear error model underestimates the error.

    x and y are channel-first, (wavelength, *spatial), as binlets passes them.
    """
    threshold = scipy.stats.chi2.ppf(1 - alpha, df=len(H) - 1)

    def estimate(x: NDArray, pseudocount: float = 0.0):
        return fractions(unmix(np.moveaxis(x, 0, -1), H, **unmix_kwargs), pseudocount)

    def test(x: NDArray, y: NDArray) -> NDArray:
        (f_x, N_x), (f_y, N_y) = map(estimate, (x, y))
        f, _ = estimate(x + y, pseudocount)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = cov_fractions(H, f, 1 / (1 / N_x + 1 / N_y))
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
    **unmix_kwargs,
) -> tuple[NDArray, NDArray]:
    """Denoise X (*spatial, wavelength) with binlets and unmix it.

    The non-linear Haar reconstruction can produce small negative counts,
    which make the KL updates unstable (W < 0, fractions outside [0, 1]),
    so the denoised X is clipped at 0.

    Returns W (*spatial, components) and the denoised X.
    """
    test = compatibility_test(H, alpha=alpha, pseudocount=pseudocount, **unmix_kwargs)
    X_binned = binlets(np.moveaxis(X, -1, 0), levels=levels, test=test, linear=False)
    X_binned = np.clip(np.moveaxis(X_binned, 0, -1), 0, None)
    return unmix(X_binned, H, **unmix_kwargs), X_binned
