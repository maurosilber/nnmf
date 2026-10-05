"""Camera data (gain, bias, read noise, ADC) in the Gaussian approximation.

(a) Noise model: are the ADU mean and variance gain * λ + offset' and
    gain² λ + read_noise² + 1/12? Without the 1/12 (quantization) term,
    how wrong is the variance?
(b) Unmixing, K=2 known spectra: the reweighted Gaussian estimator
    (variance W @ H + background) against KL on photoelectrons clipped at 0.
    Does it reach the Gaussian-approximation bound, and is the plug-in
    error calibrated (std(z) ≈ 1)?
"""

import itertools
import warnings

import numpy as np

from nnmf import Camera, cov_fractions, fractions, gaussian_spectra, measure, unmix
from nnmf import save_results

warnings.filterwarnings("ignore")
rng = np.random.default_rng(6)

GAIN = 1.7  # ADU / e⁻, not integer so that the ADC mixes levels

# (a)
READ_NOISE_ADU = [0.0, 0.2, 0.5, 1.0, 3.0]
EXPECTED = [0.0, 3.0, 30.0, 300.0]
SAMPLES = 400_000


def run_noise(rounding: str, read_noise: float, expected: float) -> dict:
    camera = Camera(gain=GAIN, read_noise=read_noise, rounding=rounding)
    Y = camera.digitize(rng, rng.poisson(expected, SAMPLES))
    return {
        "rounding": rounding,
        "read_noise": read_noise,
        "expected": expected,
        "mean-model": Y.mean() - camera.mean(expected),
        "var/model": Y.var() / camera.variance(expected),
        "var/no_adc": Y.var() / (GAIN**2 * expected + read_noise**2),
    }


def report_noise(r: dict) -> str:
    return (
        f"  {r['rounding']:5s} σ={r['read_noise']:.1f} λ={r['expected']:5.0f} "
        f"mean-model={r['mean-model']:+.3f} var/model={r['var/model']:.3f} "
        f"var/no_adc={r['var/no_adc']:7.3f}"
    )


# (b)
H = gaussian_spectra([500, 600])
F = np.array([0.3, 0.7])
PIXELS = 4000
PHOTONS = [30, 100, 1000, 10000]
READ_NOISE_E = [0.0, 1.0, 3.0]  # e⁻ per channel


def clipped_kl(X, camera):
    return unmix(np.clip(X, 0, None), H)


def gaussian(X, camera):
    return unmix(X, H, background=camera.background)


METHODS = {"gauss": gaussian, "kl clip": clipped_kl}


def run_unmix(N: int, read_noise: float) -> list[dict]:
    camera = Camera(gain=GAIN, read_noise=read_noise * GAIN)
    X = camera.to_electrons(camera.digitize(rng, measure(rng, F, H, N, PIXELS)))
    std = lambda background: np.sqrt(cov_fractions(H, F, N, background=background))
    bound, poisson = std(camera.background)[0, 0], std(0.0)[0, 0]

    def evaluate(method: str) -> dict:
        f, N_est = fractions(METHODS[method](X, camera))
        plugin = cov_fractions(
            H, np.clip(f, 1e-9, 1), N_est, background=camera.background
        )
        z = (f[:, 0] - F[0]) / np.sqrt(plugin[:, 0, 0])
        return {
            "N": N,
            "read_noise": read_noise,
            "method": method,
            "bound": bound,
            "bound/crb_poisson": bound / poisson,
            "std/bound": f[:, 0].std() / bound,
            "bias/bound": (f[:, 0].mean() - F[0]) / bound,
            "std(z_plugin)": z.std(),
        }

    return list(map(evaluate, METHODS))


def report_unmix(r: dict) -> str:
    return (
        f"  N={r['N']:6d} σ={r['read_noise']:.0f}e {r['method']:8s} "
        f"bound={r['bound']:.4f} (×{r['bound/crb_poisson']:5.2f} Poisson) "
        f"std/bound={r['std/bound']:5.2f} bias/bound={r['bias/bound']:+6.2f} "
        f"std(z_plugin)={r['std(z_plugin)']:.2f}"
    )


if __name__ == "__main__":
    noise = list(
        itertools.starmap(
            run_noise, itertools.product(["round", "floor"], READ_NOISE_ADU, EXPECTED)
        )
    )
    print(f"== (a) ADU moments vs model, gain={GAIN} ADU/e⁻")
    print("\n".join(map(report_noise, noise)))

    unmixing = list(
        itertools.chain.from_iterable(
            itertools.starmap(run_unmix, itertools.product(PHOTONS, READ_NOISE_E))
        )
    )
    print("== (b) unmixing f1=0.3, sep=100 nm, 100 channels")
    print("\n".join(map(report_unmix, unmixing)))

    save_results(__file__, {"noise": noise, "unmixing": unmixing})
