"""Known spectra, K=2: do LS and KL updates reach the Cramér-Rao bound (CRB),
and how many iterations do they need as the spectral overlap grows?"""

import itertools
import warnings

import numpy as np

from nnmf import (
    cov_fractions,
    fit_path,
    gaussian_spectra,
    measure,
    save_results,
    std_fraction_two_components,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)

PIXELS, N = 4000, 10000
F = np.array([0.5, 0.5])
SEPARATIONS = [25, 50, 100, 150]  # nm, for width 50 nm
AT = (3, 10, 30, 100, 300, 1000, 3000)
ITERATIONS = {"ls": 3000, "kl": 1000}


def run_separation(sep: float) -> list[dict]:
    H = gaussian_spectra([550 - sep / 2, 550 + sep / 2])
    X = measure(rng, F, H, N, pixels=PIXELS)
    W0 = rng.uniform(0, 1, (PIXELS, 2))
    crb = np.sqrt(cov_fractions(H, F, N)[0, 0])
    std_ls = np.sqrt(cov_fractions(H, F, N, "ls")[0, 0])

    def run_loss(loss: str) -> dict:
        at = [i for i in AT if i <= ITERATIONS[loss]]
        Ws, _ = fit_path(W0, H, X, at=at, loss=loss)
        f1 = Ws[..., 0] / Ws.sum(-1)
        return {
            "sep": sep,
            "loss": loss,
            "crb": crb,
            "ls/crb": std_ls / crb,
            "closed": std_fraction_two_components(H[0], H[1], 0.5, N),
            "iterations": at,
            "std/crb": f1.std(1) / crb,
            "bias/crb": (f1.mean(1) - 0.5) / crb,
        }

    return list(map(run_loss, ["ls", "kl"]))


def report(r: dict) -> str:
    return "\n".join(
        [
            f"sep={r['sep']} {r['loss']}: crb={r['crb']:.4f} ls/crb={r['ls/crb']:.2f} closed={r['closed']:.4f}",
            f"   std/crb  {r['std/crb'].round(2)}",
            f"   bias/crb {r['bias/crb'].round(2)}",
        ]
    )


if __name__ == "__main__":
    results = list(itertools.chain.from_iterable(map(run_separation, SEPARATIONS)))
    print(
        "== overlap (sep in nm, width 50): std/CRB and bias/CRB of f1 vs iterations", AT
    )
    print("\n".join(map(report, results)))
    save_results(__file__, results)
