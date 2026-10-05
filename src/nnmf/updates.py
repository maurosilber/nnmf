"""Lee-Seung multiplicative updates for X ≈ W @ H.

loss="ls" is the Frobenius update of manual.ipynb; loss="kl" is the
generalized KL (Poisson maximum likelihood, Richardson-Lucy) update.

An iteration is a pure function `step: (W, H) -> (W, H)`. The sequence of
iterates is produced lazily by `iterate`, so intermediate results can be
picked without storing all of them.
"""

import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12

type State = tuple[NDArray, NDArray]


# Functional helpers


def iterate[T](f: Callable[[T], T], x: T) -> Iterator[T]:
    """Lazy sequence x, f(x), f(f(x)), ..."""
    return itertools.accumulate(itertools.repeat(f), lambda x, f: f(x), initial=x)


def nth[T](iterable: Iterable[T], n: int) -> T:
    return next(itertools.islice(iterable, n, None))


def first_or_last[T](
    predicate: Callable[[T], bool], iterable: Iterable[T], default: T
) -> T:
    """First item satisfying predicate, or the last one if none does."""
    item = default
    for item in iterable:
        if predicate(item):
            break
    return item


def checkpoints[T](f: Callable[[T], T], x: T, at: Sequence[int]) -> list[T]:
    """[f^n(x) for n in at], for an increasing sequence at."""
    steps = [b - a for a, b in itertools.pairwise([0, *at])]
    advance = lambda x, n: nth(iterate(f, x), n)
    return list(itertools.accumulate(steps, advance, initial=x))[1:]


# Updates


def update_w_ls(W, H, X, eps):
    return W * (X @ H.T) / (W @ (H @ H.T) + eps)


def update_w_kl(W, H, X, eps):
    return W * ((X / (W @ H + eps)) @ H.T) / (H.sum(1) + eps)


def update_h_ls(W, H, X, eps):
    return H * (W.T @ X) / ((W.T @ W) @ H + eps)


def update_h_kl(W, H, X, eps):
    return H * (W.T @ (X / (W @ H + eps))) / (W.sum(0)[:, None] + eps)


UPDATES = {"ls": (update_w_ls, update_h_ls), "kl": (update_w_kl, update_h_kl)}


def make_step(
    H: NDArray,
    X: NDArray,
    *,
    loss: str = "kl",
    fixed_h: NDArray | bool = True,
    eps: float = EPS,
) -> Callable[[State], State]:
    """One iteration (W, H) -> (W, H).

    fixed_h: True (all spectra known), False (blind), or a boolean mask
    of shape (components,) with the known spectra. Updated spectra are
    renormalized to sum 1, and the scale is moved into W.
    """
    update_w, update_h = UPDATES[loss]
    fixed_h = np.broadcast_to(fixed_h, H.shape[:1])

    if fixed_h.all() and loss == "ls":
        # X @ H.T and H @ H.T are constant: compute them once.
        xht, hht = X @ H.T, H @ H.T
        return lambda s: (s[0] * xht / (s[0] @ hht + eps), s[1])

    if fixed_h.all():
        return lambda s: (update_w(*s, X, eps), s[1])

    def step(state: State) -> State:
        W, H = state
        W = update_w(W, H, X, eps)
        H_new = update_h(W, H, X, eps)
        scale = H_new.sum(1)
        H = np.where(fixed_h[:, None], H, H_new / scale[:, None])
        W = W * np.where(fixed_h, 1, scale)
        return W, H

    return step


def fit_path(
    W: NDArray,
    H: NDArray,
    X: NDArray,
    *,
    at: Sequence[int],
    loss: str = "kl",
    fixed_h: NDArray | bool = True,
    eps: float = EPS,
) -> tuple[NDArray, NDArray]:
    """W and H after each number of iterations in at, stacked on axis 0."""
    step = make_step(H, X, loss=loss, fixed_h=fixed_h, eps=eps)
    Ws, Hs = zip(*checkpoints(step, (W, H), at))
    return np.stack(Ws), np.stack(Hs)


def fit(W: NDArray, H: NDArray, X: NDArray, *, iterations: int, **kwargs) -> State:
    """W and H after a number of iterations. See make_step for kwargs."""
    Ws, Hs = fit_path(W, H, X, at=[iterations], **kwargs)
    return Ws[0], Hs[0]


# Unmixing with known spectra


def least_squares_guess(X: NDArray, H: NDArray, eps: float = EPS) -> NDArray:
    """Unconstrained least squares, clipped to small positive values."""
    N = X.sum(-1, keepdims=True)
    W = np.linalg.solve(H @ H.T, H @ X.T).T
    return np.clip(W, 1e-3 * N / len(H) + eps, None)


def relative_change(W_old: NDArray, W_new: NDArray, eps: float = EPS) -> float:
    """Largest change of W, relative to each pixel's photons."""
    return (np.abs(W_new - W_old).max(-1) / (W_new.sum(-1) + eps)).max()


def unmix(
    X: NDArray,
    H: NDArray,
    *,
    max_iterations: int = 1000,
    rtol: float = 1e-4,
    eps: float = EPS,
) -> NDArray:
    """Poisson maximum-likelihood unmixing with known spectra.

    KL multiplicative updates, started from the clipped least-squares
    solution and stopped when the relative change of W is below rtol.
    X can have any number of leading (spatial) dimensions.
    """
    shape = X.shape[:-1]
    X = X.reshape(-1, X.shape[-1])
    step = make_step(H, X, loss="kl", eps=eps)
    W0 = least_squares_guess(X, H, eps)

    Ws = itertools.islice(iterate(lambda W: step((W, H))[0], W0), max_iterations + 1)
    converged = lambda pair: relative_change(*pair, eps) < rtol
    _, W = first_or_last(converged, itertools.pairwise(Ws), default=(W0, W0))
    return W.reshape(*shape, len(H))
