"""Figures for the report, from the JSON/NPZ files in results/."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
FIGURES = Path(__file__).parent / "figures"

INK, MUTED, GRID, SURFACE = "#1c1b19", "#5d5c57", "#e4e2dc", "#fbfaf7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 20,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 3,
        "lines.markersize": 9,
        "legend.frameon": False,
    }
)


def load(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


def reference_line(ax, y=1.0):
    ax.axhline(y, color=MUTED, lw=1.5, ls="--", zorder=0)


def convergence():
    """Exp 1: std/CRB vs iterations, per spectral separation."""
    results = load("1_known_overlap_iterations")
    separations = sorted({r["sep"] for r in results})
    fig, axes = plt.subplots(1, len(separations), figsize=(18, 5.2), sharey=True)
    for ax, sep in zip(axes, separations):
        for r, color in zip([r for r in results if r["sep"] == sep], [BLUE, ORANGE]):
            ax.plot(
                r["iterations"],
                r["std/crb"],
                "o-",
                color=color,
                label=r["loss"].upper(),
            )
        reference_line(ax)
        ax.set(xscale="log", yscale="log", title=f"Δλ = {sep} nm", xlabel="iterations")
        ax.title.set_color(INK)
    axes[0].set_ylabel("std(f₁) / CRB")
    axes[0].set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    axes[-1].legend(loc="upper right")
    return fig


def blind():
    """Exp 3: absolute error vs pair-difference calibration, KL."""
    results = [r for r in load("3_blind_pixels") if r["loss"] == "kl"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 5.6))
    for lower, color, label in [
        (0.0, BLUE, "with pure pixels"),
        (0.3, ORANGE, "mixed only"),
    ]:
        rows = [r for r in results if r["lower"] == lower]
        pixels = [r["pixels"] for r in rows]
        left.plot(
            pixels, [r["rmse/crb"][-1] for r in rows], "o-", color=color, label=label
        )
        right.plot(
            pixels, [r["pair std(z)"] for r in rows], "o-", color=color, label=label
        )
    for ax in (left, right):
        reference_line(ax)
        ax.set(xscale="log", xlabel="pixels")
    left.set(ylabel="RMSE(f₁) / CRB", ylim=(0, 7), title="Absolute fractions")
    right.set(
        ylabel="std(z), pair difference",
        ylim=(0, 1.6),
        title="Difference between pixels",
    )
    left.legend(loc="upper right")
    return fig


def binlets_maps(N: int = 100):
    """Exp 5: f1 maps of the phantom."""
    maps = np.load(RESULTS / "5_binlets_maps.npz")
    panels = [
        ("truth", "Ground truth"),
        (f"raw N={N}", "No denoising"),
        (f"box7 N={N}", "Box filter 7×7"),
        (f"binlets 0.01 N={N}", "Binlets, α = 0.01"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), layout="constrained")
    for ax, (key, title) in zip(axes, panels):
        im = ax.imshow(
            maps[key], cmap=SEQUENTIAL, vmin=0, vmax=0.9, interpolation="none"
        )
        ax.set_title(title, color=INK)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.85, label="f₁")
    return fig


def binlets_rmse(N: int = 100):
    """Exp 5: RMSE by region, grouped bars."""
    rows = [r for r in load("5_binlets")["rmse"] if r["N"] == N]
    methods = {
        "raw": "no denoising",
        "box3": "box 3×3",
        "box7": "box 7×7",
        "binlets 0.01": "binlets α=0.01",
    }
    regions = list(rows[0]["rmse"])
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(regions))
    for i, ((method, label), color) in enumerate(
        zip(methods.items(), [MUTED, YELLOW, ORANGE, BLUE])
    ):
        (r,) = [r for r in rows if r["method"] == method]
        ax.bar(
            x + (i - 1.5) * width,
            list(r["rmse"].values()),
            width * 0.9,
            color=color,
            label=label,
        )
    ax.set_xticks(x, regions)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="RMSE(f₁)", ylim=(0, 0.095))
    ax.legend(ncols=2, loc="upper left")
    return fig


def camera_unmixing(read_noise: float = 3.0):
    """Exp 6: std and bias of f1 relative to the bound, vs photons."""
    rows = [r for r in load("6_camera_gaussian")["unmixing"]]
    rows = [r for r in rows if r["read_noise"] == read_noise]
    methods = {"gauss": ("Gaussian approx.", BLUE), "kl clip": ("clip at 0 + KL", ORANGE)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 5.6))
    for method, (label, color) in methods.items():
        r = [r for r in rows if r["method"] == method]
        N = [r["N"] for r in r]
        left.plot(N, [r["std/bound"] for r in r], "o-", color=color, label=label)
        right.plot(N, [r["bias/bound"] for r in r], "o-", color=color, label=label)
    reference_line(left, 1.0)
    reference_line(right, 0.0)
    for ax in (left, right):
        ax.set(xscale="log", xlabel="photons per pixel")
    left.set(ylabel="std(f₁) / bound", ylim=(0, 1.3), title="Spread")
    right.set(ylabel="bias(f₁) / bound", ylim=(-0.2, 0.8), title="Bias")
    left.legend(loc="lower right")
    return fig


def binlets_camera_calibration(N: int = 100, read_noise: float = 3.0):
    """Exp 7: rejection rate per level for each background in the test."""
    rows = load("7_binlets_camera")["calibration"]
    rows = [r for r in rows if r["N"] == N and r["read_noise"] == read_noise]
    tests = {
        "scaled": ("background × pixels", BLUE),
        "pixel": ("single-pixel background", ORANGE),
        "poisson": ("Poisson, clipped at 0", MUTED),
    }
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(rows))
    for test, (label, color) in tests.items():
        ax.plot(x, [r[test] for r in rows], "o-", color=color, label=label)
    reference_line(ax, 0.05)
    ax.set_xticks(x, [f"{r['level']}\n{r['pixels']} px" for r in rows])
    ax.set(xlabel="level", ylabel="rejection rate", ylim=(0, 0.65))
    ax.legend(loc="upper left")
    return fig


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    figures = {
        "convergence": convergence,
        "blind": blind,
        "binlets_maps": binlets_maps,
        "binlets_rmse": binlets_rmse,
        "camera_unmixing": camera_unmixing,
        "binlets_camera_calibration": binlets_camera_calibration,
    }
    for name, make in figures.items():
        make().savefig(FIGURES / f"{name}.png", dpi=110, bbox_inches="tight")
        print(FIGURES / f"{name}.png")
