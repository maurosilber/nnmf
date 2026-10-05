"""Known spectra: does KL unmixing reach the CRB
(a) as the number of components K grows, and
(b) at low photon counts N and fractions near 0?
(b) also checks the plug-in error (estimated f and observed N) via std(z)."""

import itertools
import warnings

import numpy as np

from nnmf import (
    cov_fractions,
    fit,
    fit_path,
    gaussian_spectra,
    measure,
    save_results,
    std_fraction_two_components,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2)

PIXELS = 4000

# (a) K components evenly spaced in 450..650 nm, equal fractions.
COMPONENTS = [2, 3, 4, 5, 6]
AT = (100, 1000, 3000)


def run_components(K: int) -> list[dict]:
    H = gaussian_spectra(np.linspace(450, 650, K))
    F, N = np.full(K, 1 / K), 10000
    X = measure(rng, F, H, N, pixels=PIXELS)
    Ws, _ = fit_path(rng.uniform(0, 1, (PIXELS, K)), H, X, at=AT, loss="kl")
    crb = np.sqrt(np.diag(cov_fractions(H, F, N)))

    def at_iteration(iteration: int, W) -> dict:
        f = W / W.sum(1, keepdims=True)
        return {
            "K": K,
            "iterations": iteration,
            "crb": crb,
            "std/crb": f.std(0) / crb,
            "bias/crb": (f.mean(0) - F) / crb,
        }

    return list(map(at_iteration, AT, Ws))


def report_components(r: dict) -> str:
    return (
        f"{r['K']} {r['iterations']} crb {r['crb'].round(3)} "
        f"std/crb {r['std/crb'].round(2)} bias/crb {r['bias/crb'].round(2)}"
    )


# (b) Photons N and fraction f1, K=2, separation 100 nm.
H2 = gaussian_spectra([500, 600])
PHOTONS = [30, 100, 1000, 10000]
FRACTIONS = [0.0, 0.02, 0.1, 0.5]


def run_photons(N: int, f1: float) -> dict:
    X = measure(rng, np.array([f1, 1 - f1]), H2, N, pixels=PIXELS)
    W, _ = fit(rng.uniform(0, 1, (PIXELS, 2)) * N, H2, X, iterations=300, loss="kl")
    f1_est = W[:, 0] / W.sum(1)
    crb = std_fraction_two_components(H2[0], H2[1], max(f1, 1e-9), N)
    # plug-in: estimated f and observed N
    plugin = std_fraction_two_components(
        H2[0], H2[1], np.clip(f1_est, 1e-9, 1), X.sum(1)
    )
    return {
        "N": N,
        "f1": f1,
        "crb": crb,
        "std/crb": f1_est.std() / crb,
        "bias/crb": (f1_est.mean() - f1) / crb,
        "std(z_plugin)": ((f1_est - f1) / plugin).std(),
    }


def report_photons(r: dict) -> str:
    return (
        f"N={r['N']:6d} f1={r['f1']:.2f} crb={r['crb']:.4f} std/crb={r['std/crb']:5.2f} "
        f"bias/crb={r['bias/crb']:+.2f} std(z_plugin)={r['std(z_plugin)']:.2f}"
    )


if __name__ == "__main__":
    components = list(itertools.chain.from_iterable(map(run_components, COMPONENTS)))
    print("== K components evenly in 450..650, equal fractions, N=10000, KL 3000 it")
    print("\n".join(map(report_components, components)))

    photons = list(
        itertools.starmap(run_photons, itertools.product(PHOTONS, FRACTIONS))
    )
    print(
        "== photons N and fraction f1, sep=100, KL 300 it: std/crb, bias/crb, plug-in z"
    )
    print("\n".join(map(report_photons, photons)))

    save_results(__file__, {"components": components, "photons": photons})
