"""Forward physics used to generate the paper's online training spectra.

This module is a public-name transcription of the frozen training generator.
The numerical expressions are intentionally preserved, including the
implementation-defined Gaussian linewidth conversion used by the historical
training run.  Changing those expressions would generate a different training
distribution and would therefore not reproduce the released checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special


LIGHT_SPEED_M_S = 299_792_458
BOLTZMANN_J_K = 1.3806504e-23

_MASS_KG = {
    "Lithium": 1.1526219e-26,
    "Lithium-6": 9.9883414e-27,
    "Lithium-7": 1.1650482e-26,
}


def gaussian_linewidth_ghz(
    temperature_k: float,
    wavelength_nm: float,
    species: str,
) -> float:
    """Return the generator's Gaussian linewidth coordinate in GHz.

    This is the exact expression used by the frozen training source.  Its
    downstream conversion in :class:`_TransitionLine` is retained verbatim for
    checkpoint reproducibility; it must not be silently replaced by a revised
    Doppler-width convention.
    """

    mass = _MASS_KG[species]
    centre_frequency_ghz = LIGHT_SPEED_M_S / (wavelength_nm * 1e-9) * 1e-9
    return float(
        centre_frequency_ghz
        * (
            8
            * BOLTZMANN_J_K
            * float(temperature_k)
            * np.log(2)
            / mass
            / LIGHT_SPEED_M_S
            / LIGHT_SPEED_M_S
        )
        ** 0.5
    )


@dataclass(frozen=True)
class _TransitionLine:
    wavelength_nm: float
    strength: float
    species: str
    temperature_k: float
    lorentzian_linewidth_ghz: float
    pressure_linewidth_ghz: float = 0.0
    instrumental_broadening_nm: float = 0.0

    def profile(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Evaluate the historical Voigt expression on a wavelength grid."""

        wavelength_m = self.wavelength_nm / 1e9
        frequency_ghz = LIGHT_SPEED_M_S / (wavelength_nm * 1e-9) / 1e9
        centre_ghz = LIGHT_SPEED_M_S / wavelength_m / 1e9
        gaussian_linewidth = gaussian_linewidth_ghz(
            self.temperature_k,
            self.wavelength_nm,
            self.species,
        )
        # Historical conversion retained exactly for checkpoint reproducibility.
        # ``gaussian_linewidth`` is the nominal Doppler FWHM, whereas SciPy's
        # ``voigt_profile`` expects the Gaussian standard deviation sigma.  The
        # frozen generator divides only by ``sqrt(2 * log(2))``, so this argument
        # is 2*sigma and the rendered Gaussian FWHM is twice the nominal value.
        # This does not affect the paper's linewidth inference: training and
        # evaluation both use the rendered linewidth directly in GHz, and no
        # temperature is inferred from it.  It becomes a physical-interpretation
        # error only if the sampled 300--20,000 K variable is claimed as an actual
        # plasma-temperature range or a fitted linewidth is converted back to
        # temperature.  For that different use, replace the denominator below by
        # ``2 * sqrt(2 * log(2))``, then regenerate the training distribution,
        # retrain the model, and repeat the evaluations; do not change this line
        # when reproducing the released checkpoint.
        gaussian_argument = gaussian_linewidth / (2 * np.log(2)) ** 0.5
        instrumental = abs(
            LIGHT_SPEED_M_S
            / (self.wavelength_nm - self.instrumental_broadening_nm)
            - LIGHT_SPEED_M_S / self.wavelength_nm
        )
        lorentzian_hwhm = (
            self.lorentzian_linewidth_ghz + self.pressure_linewidth_ghz
        ) * 0.5
        scale = special.voigt_profile(
            0.0,
            (gaussian_argument + instrumental) * 0.5,
            lorentzian_hwhm,
        )
        return (
            self.strength
            * special.voigt_profile(
                frequency_ghz - centre_ghz,
                gaussian_argument,
                lorentzian_hwhm,
            )
            / scale
        )


class LithiumForwardModel:
    """Effective lithium emission/self-absorption model used for training.

    ``core_region_lorentzian_linewidth_ghz`` controls the emission lines and
    ``edge_region_lorentzian_linewidth_ghz`` controls the absorbing lines.
    The latter are displaced by ``emission_absorption_offset_nm``.
    """

    _CENTRES_NM = (670.7760, 670.7918, 670.7911, 670.8068)
    _SPECIES = ("Lithium-7", "Lithium-6", "Lithium-7", "Lithium-6")

    def __init__(
        self,
        *,
        intensity_scale: float = 1.0,
        lithium6_percentage: float = 7.5,
        peak_optical_depth: float = 2.5,
        temperature_k: float = 300.0,
        core_region_lorentzian_linewidth_ghz: float = 0.0,
        edge_region_lorentzian_linewidth_ghz: float = 0.0,
        pressure_linewidth_ghz: float = 0.0,
        emission_absorption_offset_nm: float = 0.0,
        instrumental_broadening_nm: float = 0.0,
    ) -> None:
        lithium6_rate = float(lithium6_percentage) / 7.5
        lithium7_rate = (100.0 - float(lithium6_percentage)) / 92.5
        strengths = (
            1.0 * lithium7_rate,
            0.081 * lithium6_rate,
            0.5 * lithium7_rate,
            0.041 * lithium6_rate,
        )

        self.intensity_scale = float(intensity_scale)
        self.peak_optical_depth = float(peak_optical_depth)
        self._emission_lines = tuple(
            _TransitionLine(
                wavelength_nm=centre,
                strength=strength,
                species=species,
                temperature_k=float(temperature_k),
                lorentzian_linewidth_ghz=float(
                    core_region_lorentzian_linewidth_ghz
                ),
                pressure_linewidth_ghz=float(pressure_linewidth_ghz),
                instrumental_broadening_nm=float(instrumental_broadening_nm),
            )
            for centre, strength, species in zip(
                self._CENTRES_NM,
                strengths,
                self._SPECIES,
                strict=True,
            )
        )
        self._absorption_lines = tuple(
            _TransitionLine(
                wavelength_nm=centre + float(emission_absorption_offset_nm),
                strength=strength,
                species=species,
                temperature_k=float(temperature_k),
                lorentzian_linewidth_ghz=float(
                    edge_region_lorentzian_linewidth_ghz
                ),
                pressure_linewidth_ghz=float(pressure_linewidth_ghz),
                instrumental_broadening_nm=float(instrumental_broadening_nm),
            )
            for centre, strength, species in zip(
                self._CENTRES_NM,
                strengths,
                self._SPECIES,
                strict=True,
            )
        )

    @staticmethod
    def _profiles(
        lines: tuple[_TransitionLine, ...],
        wavelength_nm: np.ndarray,
    ) -> np.ndarray:
        return np.stack([line.profile(wavelength_nm) for line in lines], axis=0)

    @staticmethod
    def _sum_four(profiles: np.ndarray) -> np.ndarray:
        # Preserve the left-associative addition order in the frozen source.
        return profiles[0] + profiles[1] + profiles[2] + profiles[3]

    def emission_components(self, wavelength_nm: np.ndarray) -> np.ndarray:
        profiles = self._profiles(self._emission_lines, wavelength_nm)
        return profiles / np.max(self._sum_four(profiles)) * self.intensity_scale

    def absorption_components(self, wavelength_nm: np.ndarray) -> np.ndarray:
        profiles = self._profiles(self._absorption_lines, wavelength_nm)
        return profiles / np.max(self._sum_four(profiles)) * self.peak_optical_depth

    def emission(self, wavelength_nm: np.ndarray) -> np.ndarray:
        profiles = self._profiles(self._emission_lines, wavelength_nm)
        emission = self._sum_four(profiles)
        return emission / np.max(emission) * self.intensity_scale

    def optical_depth(self, wavelength_nm: np.ndarray) -> np.ndarray:
        profiles = self._profiles(self._absorption_lines, wavelength_nm)
        optical_depth = self._sum_four(profiles)
        return optical_depth / np.max(optical_depth) * self.peak_optical_depth

    def intensity(self, wavelength_nm: np.ndarray) -> np.ndarray:
        emission_profiles = self._profiles(self._emission_lines, wavelength_nm)
        emission = self._sum_four(emission_profiles)
        emission = emission / np.max(emission) * self.intensity_scale
        absorption_profiles = self._profiles(self._absorption_lines, wavelength_nm)
        optical_depth = self._sum_four(absorption_profiles)
        transmittance = np.exp(
            -optical_depth / np.max(optical_depth) * self.peak_optical_depth
        )
        return emission * transmittance


def add_detector_noise(
    intensity: np.ndarray,
    *,
    gaussian_standard_deviation: float,
    poisson_scale: float,
) -> np.ndarray:
    """Apply the same Gaussian-plus-scaled-Poisson noise as the training run."""

    if gaussian_standard_deviation <= 0 and poisson_scale <= 0:
        return intensity.copy()

    noisy = intensity.copy()
    if gaussian_standard_deviation > 0:
        noisy = noisy + np.random.normal(
            0.0,
            gaussian_standard_deviation,
            size=intensity.shape,
        )
    if poisson_scale > 0:
        scaled = np.clip(intensity / poisson_scale, 0.0, None)
        poisson_noise = (
            np.random.poisson(scaled).astype(float) * poisson_scale - intensity
        )
        noisy = noisy + poisson_noise
    return noisy


__all__ = [
    "LithiumForwardModel",
    "add_detector_noise",
    "gaussian_linewidth_ghz",
]
