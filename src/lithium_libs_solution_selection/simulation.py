"""Online Li--Ne spectrum generation used by the published training run."""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from .physics import LithiumForwardModel, add_detector_noise


CALIBRATION_QUADRATIC = -0.000000028581
CALIBRATION_LINEAR = 0.00167
CALIBRATION_CONSTANT = 669.43069

LITHIUM_PIXEL_START = 577
LITHIUM_PIXEL_END = 1088
NEON_PIXEL_START = 1088
NEON_PIXEL_END = 1600

LIBS_INTENSITY_SCALE = 10_000.0
LITHIUM6_PERCENTAGE_RANGE = (0.0, 100.0)
PEAK_OPTICAL_DEPTH_RANGE = (0.0, 2.0)
# The original generator used this K-valued variable to drive the Doppler-width
# formula, but accidentally omitted a factor 2 when converting the resulting
# FWHM to SciPy's Gaussian sigma. The released checkpoint was trained on those
# profiles, so both the sampled range and conversion are retained for integrity.
# Do not interpret 300--20,000 as a validated plasma-temperature range. For this
# model and paper, interpret/report the actual rendered Gaussian FWHM in GHz:
# rendered FWHM = 2 * gaussian_linewidth_ghz(scale, wavelength, species).
# See the calculation and physical-temperature correction path in physics.py.
HISTORICAL_GAUSSIAN_WIDTH_SCALE_RANGE_K = (300.0, 20_000.0)
CORE_EDGE_LORENTZIAN_INCREMENT_RANGE_GHZ = (0.1, 120.0)
EDGE_REGION_LORENTZIAN_LINEWIDTH_RANGE_GHZ = (0.1, 20.0)
EMISSION_ABSORPTION_OFFSET_RANGE_NM = (-0.007, 0.007)

REFERENCE_LITHIUM6_PERCENTAGE = 7.5
REFERENCE_TO_LIBS_INTENSITY_RANGE = (0.3, 1.5)
REFERENCE_GAUSSIAN_WIDTH_SCALE_K = 600.0
REFERENCE_GAUSSIAN_WIDTH_SCALE_VARIATION = 0.10
NEON_CENTRE_NM = 671.70430
NEON_FWHM_NM = 0.008846
NEON_FWHM_VARIATION = 0.05
NEON_TO_REFERENCE_LITHIUM_RATIO = 0.647
NEON_TO_REFERENCE_LITHIUM_VARIATION = 0.05

FRAME_SHIFT_RANGE_PX = (-10.0, 10.0)
GAUSSIAN_NOISE_RANGE = (0.0, 180.0)
POISSON_NOISE_RANGE = (0.0, 15.0)


@dataclass(frozen=True)
class GeneratorParameters:
    """Ground-truth generator coordinates for one synthetic spectrum."""

    intensity_scale: float
    lithium6_percentage: float
    peak_optical_depth: float
    gaussian_width_scale_k: float
    core_edge_lorentzian_increment_ghz: float
    edge_region_lorentzian_linewidth_ghz: float
    emission_absorption_offset_nm: float
    applied_frame_shift_px: float
    gaussian_noise_standard_deviation: float
    poisson_noise_scale: float

    @property
    def core_region_lorentzian_linewidth_ghz(self) -> float:
        """Return edge linewidth plus the sampled core--edge increment.

        This quantity is not sampled independently. Its derived support is
        0.2--140 GHz and is nonuniform because it is the sum of two uniforms.
        """

        return (
            self.core_edge_lorentzian_increment_ghz
            + self.edge_region_lorentzian_linewidth_ghz
        )


@dataclass(frozen=True)
class TrainingSample:
    """One generated model input and all targets used by the loss."""

    input_spectra: np.ndarray
    applied_frame_shift_px: float
    isotope_distribution: np.ndarray
    lithium6_percentage: float
    targets: np.ndarray
    parameters: GeneratorParameters

    @property
    def component_targets(self) -> np.ndarray:
        return self.targets[:8]

    @property
    def observed_reconstruction_target(self) -> np.ndarray:
        return self.targets[8]

    @property
    def canonical_reconstruction_target(self) -> np.ndarray:
        return self.targets[9]


def pixel_to_wavelength(pixel_indices: np.ndarray) -> np.ndarray:
    """Apply the quadratic CCD wavelength calibration used in training."""

    return (
        CALIBRATION_QUADRATIC * pixel_indices**2
        + CALIBRATION_LINEAR * pixel_indices
        + CALIBRATION_CONSTANT
    )


def create_isotope_target_distribution(
    lithium6_percentage: float,
    *,
    standard_deviation: float = 1.0,
    support_half_width: float = 20.0,
    number_of_bins: int = 300,
) -> np.ndarray:
    """Create the mean-corrected truncated-Gaussian isotope target.

    The original unconstrained Gaussian mean is solved so that truncation to
    0--100 percent retains the requested expectation.  Values farther than
    ``support_half_width`` percentage points from the target are then set to
    zero before discrete normalization.
    """

    target = float(lithium6_percentage)

    def expectation_residual(candidate_mean: np.ndarray) -> np.ndarray:
        alpha = (0.0 - candidate_mean) / standard_deviation
        beta = (100.0 - candidate_mean) / standard_deviation
        probability = norm.cdf(beta) - norm.cdf(alpha)
        truncated_mean = np.where(
            probability > 1e-10,
            candidate_mean
            + standard_deviation
            * (norm.pdf(alpha) - norm.pdf(beta))
            / probability,
            candidate_mean,
        )
        return truncated_mean - target

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = float(fsolve(expectation_residual, target, full_output=False)[0])

    bin_centres = np.linspace(0.0, 100.0, number_of_bins)
    lower = max(0.0, target - support_half_width)
    upper = min(100.0, target + support_half_width)
    total_probability = norm.cdf(100.0, loc=mean, scale=standard_deviation) - norm.cdf(
        0.0,
        loc=mean,
        scale=standard_deviation,
    )
    target_distribution = np.zeros(number_of_bins)
    active = (bin_centres >= lower) & (bin_centres <= upper)
    target_distribution[active] = (
        norm.pdf(bin_centres[active], loc=mean, scale=standard_deviation)
        / total_probability
    )
    if target_distribution.sum() > 0:
        target_distribution /= target_distribution.sum()
    return target_distribution


def _random_generator_parameters() -> tuple[GeneratorParameters, float, float, float]:
    # Reproducibility contract: the order and number of Python-random draws in
    # this function are part of the frozen generator and must not be changed.
    lithium6_percentage = random.uniform(*LITHIUM6_PERCENTAGE_RANGE)
    peak_optical_depth = random.uniform(*PEAK_OPTICAL_DEPTH_RANGE)
    gaussian_width_scale_k = random.uniform(
        *HISTORICAL_GAUSSIAN_WIDTH_SCALE_RANGE_K
    )
    core_increment_ghz = random.uniform(
        *CORE_EDGE_LORENTZIAN_INCREMENT_RANGE_GHZ
    )
    edge_linewidth_ghz = random.uniform(
        *EDGE_REGION_LORENTZIAN_LINEWIDTH_RANGE_GHZ
    )
    emission_absorption_offset_nm = random.uniform(
        *EMISSION_ABSORPTION_OFFSET_RANGE_NM
    )

    reference_ratio = random.uniform(*REFERENCE_TO_LIBS_INTENSITY_RANGE)
    reference_intensity = LIBS_INTENSITY_SCALE * reference_ratio
    reference_width_scale = REFERENCE_GAUSSIAN_WIDTH_SCALE_K * random.uniform(
        1.0 - REFERENCE_GAUSSIAN_WIDTH_SCALE_VARIATION,
        1.0 + REFERENCE_GAUSSIAN_WIDTH_SCALE_VARIATION,
    )
    neon_ratio = NEON_TO_REFERENCE_LITHIUM_RATIO * random.uniform(
        1.0 - NEON_TO_REFERENCE_LITHIUM_VARIATION,
        1.0 + NEON_TO_REFERENCE_LITHIUM_VARIATION,
    )
    neon_intensity = reference_intensity * neon_ratio

    applied_frame_shift_px = random.uniform(*FRAME_SHIFT_RANGE_PX)
    gaussian_noise = random.uniform(*GAUSSIAN_NOISE_RANGE)
    poisson_noise = random.uniform(*POISSON_NOISE_RANGE)
    parameters = GeneratorParameters(
        intensity_scale=LIBS_INTENSITY_SCALE,
        lithium6_percentage=lithium6_percentage,
        peak_optical_depth=peak_optical_depth,
        gaussian_width_scale_k=gaussian_width_scale_k,
        core_edge_lorentzian_increment_ghz=core_increment_ghz,
        edge_region_lorentzian_linewidth_ghz=edge_linewidth_ghz,
        emission_absorption_offset_nm=emission_absorption_offset_nm,
        applied_frame_shift_px=applied_frame_shift_px,
        gaussian_noise_standard_deviation=gaussian_noise,
        poisson_noise_scale=poisson_noise,
    )
    return parameters, reference_intensity, reference_width_scale, neon_intensity


def _fixed_generator_parameters() -> tuple[GeneratorParameters, float, float, float]:
    reference_intensity = LIBS_INTENSITY_SCALE * 0.8
    parameters = GeneratorParameters(
        intensity_scale=LIBS_INTENSITY_SCALE,
        lithium6_percentage=50.0,
        peak_optical_depth=1.0,
        gaussian_width_scale_k=3000.0,
        core_edge_lorentzian_increment_ghz=40.0,
        edge_region_lorentzian_linewidth_ghz=10.0,
        emission_absorption_offset_nm=0.002,
        applied_frame_shift_px=0.0,
        gaussian_noise_standard_deviation=50.0,
        poisson_noise_scale=5.0,
    )
    return (
        parameters,
        reference_intensity,
        REFERENCE_GAUSSIAN_WIDTH_SCALE_K,
        reference_intensity * NEON_TO_REFERENCE_LITHIUM_RATIO,
    )


def _neon_peak(wavelength_nm: np.ndarray, intensity: float) -> np.ndarray:
    fwhm = NEON_FWHM_NM * random.uniform(
        1.0 - NEON_FWHM_VARIATION,
        1.0 + NEON_FWHM_VARIATION,
    )
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return intensity * np.exp(
        -(wavelength_nm - NEON_CENTRE_NM) ** 2 / (2.0 * sigma**2)
    )


def generate_training_sample(
    *,
    output_length: int = 512,
    number_of_isotope_bins: int = 300,
    normalization_range: tuple[float, float] = (0.0, 1.0),
    random_parameters: bool = True,
) -> TrainingSample:
    """Generate one two-channel input and the final paper training targets.

    Component targets and the canonical reconstruction use the unshifted
    lithium coordinate.  The second reconstruction target uses the observed
    input frame.  This fixed target contract replaces the development-time
    frame/version switches from the private code.
    """

    if random_parameters:
        parameters, reference_intensity, reference_width_scale, neon_intensity = (
            _random_generator_parameters()
        )
    else:
        parameters, reference_intensity, reference_width_scale, neon_intensity = (
            _fixed_generator_parameters()
        )

    lithium_pixels = np.linspace(
        LITHIUM_PIXEL_START,
        LITHIUM_PIXEL_END,
        output_length,
    ) + parameters.applied_frame_shift_px
    neon_pixels = np.linspace(
        NEON_PIXEL_START,
        NEON_PIXEL_END,
        output_length,
    ) + parameters.applied_frame_shift_px
    lithium_wavelength = pixel_to_wavelength(lithium_pixels)
    neon_wavelength = pixel_to_wavelength(neon_pixels)

    spectrum_model = LithiumForwardModel(
        intensity_scale=parameters.intensity_scale,
        lithium6_percentage=parameters.lithium6_percentage,
        peak_optical_depth=parameters.peak_optical_depth,
        temperature_k=parameters.gaussian_width_scale_k,
        core_region_lorentzian_linewidth_ghz=(
            parameters.core_region_lorentzian_linewidth_ghz
        ),
        edge_region_lorentzian_linewidth_ghz=(
            parameters.edge_region_lorentzian_linewidth_ghz
        ),
        emission_absorption_offset_nm=parameters.emission_absorption_offset_nm,
    )
    lithium_clean = spectrum_model.intensity(lithium_wavelength)
    lithium_noisy = add_detector_noise(
        lithium_clean,
        gaussian_standard_deviation=(
            parameters.gaussian_noise_standard_deviation
        ),
        poisson_scale=parameters.poisson_noise_scale,
    )

    # This unused reference-lithium calculation is deliberately retained.  It
    # consumed NumPy random numbers in the frozen generator before the neon
    # noise was sampled, so omitting it changes seeded training inputs.
    reference_lithium_model = LithiumForwardModel(
        intensity_scale=reference_intensity,
        lithium6_percentage=REFERENCE_LITHIUM6_PERCENTAGE,
        peak_optical_depth=0.0,
        temperature_k=reference_width_scale,
        core_region_lorentzian_linewidth_ghz=0.0,
        edge_region_lorentzian_linewidth_ghz=0.0,
        emission_absorption_offset_nm=0.0,
    )
    reference_lithium_clean = reference_lithium_model.intensity(lithium_wavelength)
    _ = add_detector_noise(
        reference_lithium_clean,
        gaussian_standard_deviation=(
            parameters.gaussian_noise_standard_deviation
        ),
        poisson_scale=parameters.poisson_noise_scale,
    )

    neon_clean = _neon_peak(neon_wavelength, neon_intensity)
    neon_noisy = add_detector_noise(
        neon_clean,
        gaussian_standard_deviation=(
            parameters.gaussian_noise_standard_deviation
        ),
        poisson_scale=parameters.poisson_noise_scale,
    )
    input_spectra = np.stack(
        (
            lithium_noisy - np.min(lithium_noisy),
            neon_noisy - np.min(neon_noisy),
        ),
        axis=0,
    )

    canonical_pixels = np.linspace(
        LITHIUM_PIXEL_START,
        LITHIUM_PIXEL_END,
        output_length,
    )
    canonical_wavelength = pixel_to_wavelength(canonical_pixels)
    canonical_clean = spectrum_model.intensity(canonical_wavelength)
    canonical_min = np.min(canonical_clean)
    canonical_max = np.max(canonical_clean)
    canonical_span = canonical_max - canonical_min + 1e-8

    emission_components = (
        spectrum_model.emission_components(canonical_wavelength) / canonical_span
    )
    emission_components = (
        emission_components
        * (normalization_range[1] - normalization_range[0])
        + normalization_range[0]
    )
    absorption_components = spectrum_model.absorption_components(canonical_wavelength)

    observed_clean = spectrum_model.intensity(lithium_wavelength)
    observed_reconstruction = (
        observed_clean.reshape(1, -1) - np.min(observed_clean)
    ) / (np.max(observed_clean) - np.min(observed_clean) + 1e-8)
    observed_reconstruction = (
        observed_reconstruction
        * (normalization_range[1] - normalization_range[0])
        + normalization_range[0]
    )
    canonical_reconstruction = (
        canonical_clean.reshape(1, -1) - canonical_min
    ) / canonical_span
    canonical_reconstruction = (
        canonical_reconstruction
        * (normalization_range[1] - normalization_range[0])
        + normalization_range[0]
    )
    targets = np.concatenate(
        (
            emission_components,
            absorption_components,
            observed_reconstruction,
            canonical_reconstruction,
        ),
        axis=0,
    )
    isotope_distribution = create_isotope_target_distribution(
        parameters.lithium6_percentage,
        number_of_bins=number_of_isotope_bins,
    )
    return TrainingSample(
        input_spectra=input_spectra,
        applied_frame_shift_px=parameters.applied_frame_shift_px,
        isotope_distribution=isotope_distribution,
        lithium6_percentage=parameters.lithium6_percentage,
        targets=targets,
        parameters=parameters,
    )


__all__ = [
    "CORE_EDGE_LORENTZIAN_INCREMENT_RANGE_GHZ",
    "EDGE_REGION_LORENTZIAN_LINEWIDTH_RANGE_GHZ",
    "EMISSION_ABSORPTION_OFFSET_RANGE_NM",
    "FRAME_SHIFT_RANGE_PX",
    "GeneratorParameters",
    "HISTORICAL_GAUSSIAN_WIDTH_SCALE_RANGE_K",
    "LITHIUM6_PERCENTAGE_RANGE",
    "PEAK_OPTICAL_DEPTH_RANGE",
    "TrainingSample",
    "create_isotope_target_distribution",
    "generate_training_sample",
    "pixel_to_wavelength",
]
