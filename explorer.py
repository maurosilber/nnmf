import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.ndimage
    import scipy.stats

    from nnmf import (
        WAVELENGTH,
        Camera,
        binlets_unmix,
        fractions,
        gaussian_spectra,
        measure,
        rmse,
        unmix,
    )

    return (
        Camera,
        WAVELENGTH,
        binlets_unmix,
        fractions,
        gaussian_spectra,
        measure,
        mo,
        np,
        plt,
        rmse,
        scipy,
        time,
        unmix,
    )


@app.cell
def _(mo):
    mo.md("""
    # Binlets explorer

    A phantom of K components with known spectra, unmixed four ways:
    **truth**, **raw** (each pixel alone), **box** (mean filter before unmixing)
    and **binlets**. With read noise, the data are camera photoelectrons
    (gain 1.7 ADU/e⁻, rounding ADC) and unmixing uses the Gaussian approximation.
    Change the settings and press **Run**.
    """)
    return


@app.cell
def _(mo):
    defaults = {
        "components": 2,
        "separation": 100,
        "width": 50,
        "photons": 100,
        "read_noise": 0.0,
        "size": 64,
        "box": 7,
        "alpha": 0.01,
        "repeats": 5,
        "seed": 0,
    }
    d = defaults
    controls = {
        "components": mo.ui.slider(2, 6, value=d["components"], label="components K", show_value=True),
        "separation": mo.ui.slider(
            10, 150, step=5, value=d["separation"], label="spectral separation (nm)", show_value=True
        ),
        "width": mo.ui.slider(
            20, 100, step=5, value=d["width"], label="spectral width σ (nm)", show_value=True
        ),
        "photons": mo.ui.slider(
            steps=[10, 20, 50, 100, 200, 500, 1000, 5000],
            value=d["photons"],
            label="photons per pixel N",
            show_value=True,
        ),
        "read_noise": mo.ui.slider(
            0, 5, step=0.5, value=d["read_noise"], label="read noise (e⁻ per channel)", show_value=True
        ),
        "size": mo.ui.slider(
            steps=[32, 64, 128], value=d["size"], label="image size (px)", show_value=True
        ),
        "box": mo.ui.slider(3, 15, step=2, value=d["box"], label="box size (px)", show_value=True),
        "alpha": mo.ui.slider(
            steps=[0.2, 0.1, 0.05, 0.01, 0.001],
            value=d["alpha"],
            label="binlets α",
            show_value=True,
        ),
        "repeats": mo.ui.slider(
            steps=[1, 2, 5, 10, 20],
            value=d["repeats"],
            label="noise realizations (for bias)",
            show_value=True,
        ),
        "seed": mo.ui.number(0, 10_000, value=d["seed"], label="seed"),
    }
    settings = (
        mo.md("\n\n".join(f"{{{key}}}" for key in controls))
        .batch(**controls)
        .form(submit_button_label="Run", bordered=True)
    )
    return defaults, settings


@app.cell
def _(np, scipy):
    def phantom(rng, K: int, n: int):
        """Fractions (n, n, K) and region labels (n, n): background (0),
        a disk (1), a square (2), a smooth gradient band (3) and a thin line
        without component 1 (4). Region fractions are random."""
        y, x = np.mgrid[:n, :n] / n
        labels = np.zeros((n, n), int)
        labels[(x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.15**2] = 1
        labels[(abs(x - 0.72) < 0.12) & (abs(y - 0.3) < 0.12)] = 2
        labels[y > 0.6] = 3
        labels[(y > 0.6) & (abs(x - 0.5) < 0.02)] = 4

        background, disk, square, left, right = rng.dirichlet(np.ones(K), size=5)
        line = left * (np.arange(K) > 0)
        F = np.empty((n, n, K))
        for label, f in enumerate([background, disk, square, None, line / line.sum()]):
            if f is not None:
                F[labels == label] = f
        t = x[labels == 3][:, None]
        F[labels == 3] = (1 - t) * left + t * right
        return F, labels

    def regions(labels):
        """Edges: pixels next to another region. Gradient: rest of the band."""
        edges = scipy.ndimage.morphological_gradient(labels, size=3) > 0
        gradient = ~edges & (labels == 3)
        return {"flat": ~edges & ~gradient, "edges": edges, "gradient": gradient}

    return phantom, regions


@app.cell
def _(
    Camera,
    defaults,
    gaussian_spectra,
    measure,
    np,
    phantom,
    regions,
    settings,
):
    s = settings.value or defaults
    K = s["components"]
    H = gaussian_spectra(
        550 + s["separation"] * (np.arange(K) - (K - 1) / 2), width=s["width"]
    )
    rng = np.random.default_rng(s["seed"])
    F, labels = phantom(rng, K, s["size"])
    masks = regions(labels)

    # Independent noise realizations of the same phantom; X is the first.
    Xs = np.stack([measure(rng, F, H, s["photons"]) for _ in range(s["repeats"])])
    if s["read_noise"] > 0:
        camera = Camera(gain=1.7, read_noise=s["read_noise"] * 1.7)
        Xs = camera.to_electrons(camera.digitize(rng, Xs))
        background = camera.background
    else:
        background = 0.0
    X = Xs[0]
    return F, H, K, X, Xs, background, masks, s


@app.cell
def _(F, H, Xs, background, binlets_unmix, fractions, np, s, scipy, time, unmix):
    def box(X):
        size = (s["box"], s["box"], 1)
        X = scipy.ndimage.uniform_filter(X, size=size, mode="nearest")
        return unmix(X, H, background=background)

    methods = {
        "raw": lambda X: unmix(X, H, background=background),
        f"box {s['box']}×{s['box']}": box,
        f"binlets α={s['alpha']}": lambda X: binlets_unmix(
            X, H, alpha=s["alpha"], background=background
        )[0],
    }

    # Fractions of every realization; maps and models use the first one.
    estimates, samples, seconds = {"truth": F}, {}, {}
    models = {"truth": s["photons"] * F @ H}
    for _name, _method in methods.items():
        _start = time.perf_counter()
        _Ws = np.stack([_method(_X) for _X in Xs])
        seconds[_name] = (time.perf_counter() - _start) / len(Xs)
        samples[_name] = fractions(_Ws)[0]
        estimates[_name] = samples[_name][0]
        models[_name] = _Ws[0] @ H
    return estimates, models, samples, seconds


@app.cell
def _(F, masks, mo, np, s, samples, seconds):
    def errors(f, mask):
        """RMS over pixels and components, with RMSE² = bias² + std²."""
        e = f - F
        return {
            "RMSE": np.sqrt((e**2).mean(0)[mask].mean()),
            "bias": np.sqrt((e.mean(0) ** 2)[mask].mean()),
            "std": np.sqrt(f.var(0)[mask].mean()),
        }

    table = mo.ui.table(
        [
            {
                "method": name,
                "region": region,
                **{k: round(float(v), 4) for k, v in errors(f, mask).items()},
                "seconds": round(seconds[name], 2),
            }
            for name, f in samples.items()
            for region, mask in masks.items()
        ],
        selection=None,
        page_size=12,
        label=(
            f"Fractions vs truth over {s['repeats']} realizations, by region "
            "(bias includes std/√realizations of noise)"
        ),
    )
    return (table,)


@app.cell
def _(F, K, np, plt, samples):
    _bias = {name: f.mean(0) - F for name, f in samples.items()}
    _vmax = max(np.percentile(np.abs(b), 99) for b in _bias.values())
    fig_bias, axes_bias = plt.subplots(
        K,
        len(_bias),
        figsize=(3 * len(_bias), 3 * K),
        squeeze=False,
        layout="constrained",
    )
    for _col, (_name, _b) in enumerate(_bias.items()):
        axes_bias[0, _col].set_title(_name)
        for _k in range(K):
            _ax = axes_bias[_k, _col]
            bias_image = _ax.imshow(
                _b[..., _k],
                cmap="RdBu_r",
                vmin=-_vmax,
                vmax=_vmax,
                interpolation="none",
            )
            _ax.set_xticks([])
            _ax.set_yticks([])
    for _k in range(K):
        axes_bias[_k, 0].set_ylabel(f"f{_k + 1}")
    fig_bias.colorbar(
        bias_image, ax=axes_bias, shrink=0.6, label="mean estimate − truth"
    )
    None
    return (fig_bias,)


@app.cell
def _(K, estimates, plt):
    fig, axes = plt.subplots(
        K,
        len(estimates),
        figsize=(3 * len(estimates), 3 * K),
        squeeze=False,
        layout="constrained",
    )
    for _col, (_name, _f) in enumerate(estimates.items()):
        axes[0, _col].set_title(_name)
        for _k in range(K):
            _ax = axes[_k, _col]
            image = _ax.imshow(
                _f[..., _k], cmap="Blues", vmin=0, vmax=1, interpolation="none"
            )
            _ax.set_xticks([])
            _ax.set_yticks([])
    for _k in range(K):
        axes[_k, 0].set_ylabel(f"f{_k + 1}")
    fig.colorbar(image, ax=axes, shrink=0.6, label="fraction")
    None
    return (fig,)


@app.cell
def _(H, WAVELENGTH, X, plt, s):
    fig_spectra, (left, right) = plt.subplots(
        1, 2, figsize=(11, 3.2), layout="constrained"
    )
    for _k, _h in enumerate(H):
        left.plot(WAVELENGTH, _h, lw=2, label=f"component {_k + 1}")
    left.set(xlabel="wavelength (nm)", ylabel="spectrum", title="Known spectra H")
    left.legend(frameon=False)
    right.plot(WAVELENGTH, X[0, 0], color="0.4", lw=1.5, label="corner pixel")
    right.plot(WAVELENGTH, X.mean((0, 1)), color="black", lw=2, label="image mean")
    right.set(
        xlabel="wavelength (nm)",
        ylabel="photoelectrons" if s["read_noise"] else "photons",
        title="Measured data",
    )
    right.legend(frameon=False)
    None
    return (fig_spectra,)


@app.cell
def _(K, background, masks, models, np, plt, scipy):
    from matplotlib.colors import LogNorm

    _truth = models["truth"]
    _sigma = np.sqrt(_truth + background)
    _methods = {name: m for name, m in models.items() if name != "truth"}
    fig_residuals, axes_residuals = plt.subplots(
        2,
        len(_methods),
        figsize=(3 * len(_methods), 6),
        squeeze=False,
        layout="constrained",
    )
    _bins = np.linspace(-3, 3, 61)
    _x = np.logspace(-3, 3, 301)
    for _col, (_name, _model) in enumerate(_methods.items()):
        # Model minus the true expected spectrum, in units of the noise σ.
        _chi2 = (((_model - _truth) / _sigma) ** 2).sum(-1)
        _top, _bottom = axes_residuals[:, _col]
        chi2_image = _top.imshow(
            _chi2,
            cmap="magma",
            norm=LogNorm(1e-2, 1e2, clip=True),
            interpolation="none",
        )
        _top.set_title(f"{_name}\nmean χ² = {_chi2.mean():.2f}")
        _top.set_xticks([])
        _top.set_yticks([])
        _log_chi2 = np.log10(np.clip(_chi2, 1e-3, 1e3))
        for _region, _mask in masks.items():
            _bottom.hist(
                _log_chi2[_mask],
                bins=_bins,
                density=True,
                histtype="step",
                label=_region,
            )
        # Unbiased estimator with the per-pixel noise: χ² with K degrees of freedom.
        _bottom.plot(
            np.log10(_x),
            scipy.stats.chi2(K).pdf(_x) * _x * np.log(10),
            color="black",
            lw=1,
            label=f"χ²({K})",
        )
        _bottom.set(xlabel="log₁₀ χ² per pixel")
    axes_residuals[0, 0].set_ylabel("χ² per pixel")
    axes_residuals[1, 0].set_ylabel("density per decade")
    axes_residuals[1, -1].legend(frameon=False, fontsize="small")
    fig_residuals.colorbar(
        chi2_image, ax=axes_residuals[0], shrink=0.8, label="χ² vs truth"
    )
    None
    return (fig_residuals,)


@app.cell
def _(fig, fig_bias, fig_residuals, fig_spectra, mo, settings, table):
    mo.hstack(
        [settings, mo.vstack([fig, table, fig_bias, fig_residuals, fig_spectra])]
    )
    return


if __name__ == "__main__":
    app.run()
