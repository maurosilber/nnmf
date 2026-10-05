"""Blind unmixing, K=2: how close to the CRB are the fractions when H is
also estimated, depending on the number of pixels and on whether there are
(nearly) pure pixels? Pixels come in pairs with equal fractions, to check
the pair difference test that binlets uses (std(z) should be 1)."""

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

H = gaussian_spectra([500, 600])
N, REPETITIONS = 1000, 8
AT = (30, 100, 300, 1000, 3000)
LOSSES = ["kl", "ls"]
LOWER = [0.0, 0.3]  # fractions in [lower, 1 - lower]
PIXELS = [20, 200, 2000]


def repetition(rng, f1, crb, loss):
    """One blind fit. Returns normalized errors, H error and pair z-scores."""
    F = np.stack([f1, 1 - f1], 1)
    X = measure(rng, F, H, N)
    W0, H0 = rng.uniform(0, 1, (len(f1), 2)), random_spectra(rng, H.shape)
    Ws, Hs = fit_path(W0, H0, X, at=AT, loss=loss, fixed_h=False)
    order = match_components(Hs[-1], H)
    Ws, H_est = Ws[..., order], Hs[-1][order]

    f1_est = Ws[..., 0] / Ws.sum(-1)
    # plug-in pair test on the last iteration, using estimated H and f
    F_est = np.stack([f1_est[-1], 1 - f1_est[-1]], 1).clip(1e-6)
    std = np.sqrt(cov_fractions(H_est, F_est, Ws[-1].sum(-1))[:, 0, 0])
    z = (f1_est[-1, ::2] - f1_est[-1, 1::2]) / np.hypot(std[::2], std[1::2])
    return (f1_est - f1) / crb, total_variation(H_est, H).max(), z


def run(loss: str, lower: float, pixels: int) -> dict:
    rng = np.random.default_rng(pixels)
    f1 = np.repeat(rng.uniform(lower, 1 - lower, pixels // 2), 2)
    crb = np.sqrt(cov_fractions(H, np.stack([f1, 1 - f1], 1), N)[:, 0, 0])
    errors, h_errors, z = map(
        np.array, zip(*(repetition(rng, f1, crb, loss) for _ in range(REPETITIONS)))
    )  # errors: (repetitions, iterations, pixels)
    return {
        "loss": loss,
        "lower": lower,
        "pixels": pixels,
        "iterations": AT,
        "H_TV": h_errors.mean(),
        "rmse/crb": np.sqrt((errors**2).mean((0, 2))),
        "systematic/crb": np.sqrt((errors.mean(0) ** 2).mean(-1)),
        "pair std(z)": z.std(),
    }


def report(r: dict) -> str:
    return (
        f"P={r['pixels']:5d} range=[{r['lower']},{1 - r['lower']}] {r['loss']} "
        f"Herr(TV)={r['H_TV']:.3f}  rmse/crb per it {r['rmse/crb'].round(2)}  "
        f"systematic/crb {r['systematic/crb'].round(2)}  pair std(z)={r['pair std(z)']:.2f}"
    )


if __name__ == "__main__":
    results = list(itertools.starmap(run, itertools.product(LOSSES, LOWER, PIXELS)))
    print("\n".join(map(report, results)))
    save_results(__file__, results)
