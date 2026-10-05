"""Camera noise: Poisson photoelectrons, gain, bias, read noise and ADC.

    Y = Q(gain * n + offset + read_noise * ε),   n ~ Poisson(λ), ε ~ N(0, 1)

Q rounds (or floors) to integer ADU and clips to the ADC range.
In the Gaussian approximation, Q adds uniform noise in (-1/2, 1/2)
(Sheppard), so that, in ADU,

    E[Y]   = gain * λ + effective_offset
    Var[Y] = gain**2 * λ + read_noise**2 + 1/12

Valid while read_noise ≳ 0.5 ADU. Floor shifts the offset by -1/2.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ROUNDING = {"round": np.round, "floor": np.floor}


@dataclass(frozen=True)
class Camera:
    gain: float = 1.0  # ADU per photoelectron
    offset: float = 100.0  # bias, ADU
    read_noise: float = 1.0  # std, ADU
    rounding: str = "round"
    bits: int = 16

    def digitize(self, rng: np.random.Generator, electrons: NDArray) -> NDArray:
        """ADU counts of the given photoelectrons."""
        x = self.gain * electrons + self.offset
        x = x + self.read_noise * rng.standard_normal(np.shape(x))
        return np.clip(ROUNDING[self.rounding](x), 0, 2**self.bits - 1)

    @property
    def effective_offset(self) -> float:
        return self.offset - 0.5 if self.rounding == "floor" else self.offset

    @property
    def dark_variance(self) -> float:
        """Read noise broadened by quantization, in ADU²."""
        return self.read_noise**2 + 1 / 12

    def mean(self, expected_electrons):
        return self.gain * expected_electrons + self.effective_offset

    def variance(self, expected_electrons):
        return self.gain**2 * expected_electrons + self.dark_variance

    def to_electrons(self, Y: NDArray) -> NDArray:
        """Unbiased photoelectrons; can be negative."""
        return (Y - self.effective_offset) / self.gain

    @property
    def background(self) -> float:
        """Variance added to each channel, in photoelectrons²."""
        return self.dark_variance / self.gain**2
