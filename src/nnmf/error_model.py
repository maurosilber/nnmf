"""Error model of the fractions f = W / N when the spectra H are known.

With read noise (Gaussian approximation), channel l has mean
mu_l = N (f @ H)_l and variance mu_l + background, in photons. With
background = 0 the covariances have the Poisson form C(f, H) / N.
"""

import numpy as np
from numpy.typing import NDArray


def channel_variance(H: NDArray, f: NDArray, N=1.0, background=0.0) -> NDArray:
    N = np.asarray(N, dtype=float)[..., None]
    return N * (f @ H) + background


def fisher_information(H: NDArray, f: NDArray, N=1.0, background=0.0) -> NDArray:
    """Fisher information for W, H diag(1 / variance) H.T.

    With the defaults, the information per photon of the multinomial model,
    G = H diag(1/p) H.T with p = f @ H. With background, the information of
    the mean only (quasi-likelihood), which the reweighted ("gauss")
    estimator reaches.
    """
    v = channel_variance(H, f, N, background)
    return np.einsum("...kl,...jl->...kj", H / v[..., None, :], H)


def cov_w_ml(H: NDArray, f: NDArray, N, background) -> NDArray:
    """Cramér-Rao bound, reached by the Poisson ML (KL) estimator, or the
    Gaussian-approximation bound with background."""
    return np.linalg.inv(fisher_information(H, f, N, background))


def cov_w_ls(H: NDArray, f: NDArray, N, background) -> NDArray:
    """Sandwich covariance of the (unconstrained) least squares estimator."""
    A = np.linalg.solve(H @ H.T, H)
    v = channel_variance(H, f, N, background)
    return np.einsum("kl,...l,jl->...kj", A, v, A)


COV_W = {"kl": cov_w_ml, "gauss": cov_w_ml, "ls": cov_w_ls}


def fraction_jacobian(f: NDArray) -> NDArray:
    """Jacobian of W -> W / N, times N."""
    return np.eye(f.shape[-1]) - f[..., :, None]


def cov_fractions(
    H: NDArray, f: NDArray, N, loss: str = "kl", background=0.0
) -> NDArray:
    """Covariance of the estimated fractions, to first order."""
    J = fraction_jacobian(f)
    cov_w = COV_W[loss](H, f, N, background)
    N = np.asarray(N, dtype=float)[..., None, None]
    return J @ cov_w @ np.swapaxes(J, -1, -2) / N**2


def std_fraction_two_components(h1, h2, f, N):
    """Closed form for K=2: var(f) = 1 / (N * sum((h1 - h2)**2 / p))."""
    p = np.multiply.outer(f, h1) + np.multiply.outer(1 - f, h2)
    return 1 / np.sqrt(N * ((h1 - h2) ** 2 / p).sum(-1))


def chi2_statistic(d: NDArray, cov: NDArray) -> NDArray:
    """d^T cov^+ d, per pixel.

    The covariance of fractions is singular along the sum-to-one direction,
    so the pseudo-inverse is used, and the statistic has K - 1 degrees of
    freedom.
    """
    S = np.linalg.pinv(cov, hermitian=True)
    return np.einsum("...k,...kj,...j->...", d, S, d)


def chi2_difference(f_a, f_b, cov_a, cov_b):
    """Test statistic for 'bins a and b have the same fractions'."""
    return chi2_statistic(f_a - f_b, cov_a + cov_b)
