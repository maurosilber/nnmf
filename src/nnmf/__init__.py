"""Spectral unmixing (NNMF) of Poisson data, its error model, and binlets."""

from .camera import Camera
from .binning import binlets_unmix, compatibility_test
from .error_model import (
    chi2_difference,
    chi2_statistic,
    cov_fractions,
    fisher_information,
    std_fraction_two_components,
)
from .evaluation import match_components, rmse, save_results, total_variation
from .spectra import (
    WAVELENGTH,
    fractions,
    gaussian_spectra,
    measure,
    normalize,
    random_spectra,
)
from .updates import fit, fit_path, iterate, make_step, unmix

__all__ = [
    "WAVELENGTH",
    "Camera",
    "binlets_unmix",
    "chi2_difference",
    "chi2_statistic",
    "compatibility_test",
    "cov_fractions",
    "fisher_information",
    "fit",
    "fit_path",
    "fractions",
    "gaussian_spectra",
    "iterate",
    "make_step",
    "match_components",
    "measure",
    "normalize",
    "random_spectra",
    "rmse",
    "save_results",
    "std_fraction_two_components",
    "total_variation",
    "unmix",
]
