"""Error model of the fractions f = W / N when the spectra H are known.

All covariances are per pixel and have the form C(f, H) / N.
"""

import numpy as np
from numpy.typing import NDArray


def fisher_information(H: NDArray, f: NDArray) -> NDArray:
    """Fisher information per photon for the fractions' multinomial model.

    Spectrum shape p = f @ H. Returns G = H diag(1/p) H.T, so that the
    Fisher information for W is G / N.
    """
    p = f @ H
    return np.einsum("...kl,...jl->...kj", H / p[..., None, :], H)


def cov_w_kl(H: NDArray, f: NDArray) -> NDArray:
    """Cramér-Rao bound, reached by the Poisson ML (KL) estimator."""
    return np.linalg.inv(fisher_information(H, f))


def cov_w_ls(H: NDArray, f: NDArray) -> NDArray:
    """Sandwich covariance of the (unconstrained) least squares estimator."""
    A = np.linalg.solve(H @ H.T, H)
    return np.einsum("kl,...l,jl->...kj", A, f @ H, A)


COV_W = {"kl": cov_w_kl, "ls": cov_w_ls}


def fraction_jacobian(f: NDArray) -> NDArray:
    """Jacobian of W -> W / N, times N."""
    return np.eye(f.shape[-1]) - f[..., :, None]


def cov_fractions(H: NDArray, f: NDArray, N, loss: str = "kl") -> NDArray:
    """Covariance of the estimated fractions, to first order."""
    N = np.asarray(N, dtype=float)[..., None, None]
    J = fraction_jacobian(f)
    return J @ COV_W[loss](H, f) @ np.swapaxes(J, -1, -2) / N


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
