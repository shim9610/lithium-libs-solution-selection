"""Training objective used for the released solution-selection checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainingLossWeights:
    """Weights for the final paper objective and its component ablation."""

    component_profile: float = 1.0
    component_peak_height: float = 1.0
    component_peak_position: float = 1.0
    canonical_reconstruction: float = 1.0
    observed_reconstruction: float = 1.0
    direct_shift_supervision: float = 0.0
    isotope_kl: float = 1.0
    isotope_entropy: float = 0.0

    @classmethod
    def full_objective(cls) -> "TrainingLossWeights":
        return cls()

    @classmethod
    def component_supervision_ablation(cls) -> "TrainingLossWeights":
        """Retain both reconstruction terms and KL, remove component targets."""

        return cls(
            component_profile=0.0,
            component_peak_height=0.0,
            component_peak_position=0.0,
        )


@dataclass(frozen=True)
class TrainingLossBreakdown:
    component_mse: torch.Tensor
    component_relative_l1: torch.Tensor
    component_peak_height_relative_l1: torch.Tensor
    component_peak_position: torch.Tensor
    canonical_reconstruction: torch.Tensor
    observed_reconstruction: torch.Tensor
    direct_shift_supervision: torch.Tensor
    isotope_kl: torch.Tensor
    isotope_entropy_reciprocal: torch.Tensor

    def detached_floats(self) -> dict[str, float]:
        return {
            field.name: float(getattr(self, field.name).detach().cpu())
            for field in fields(self)
        }


def soft_argmax(
    values: torch.Tensor,
    *,
    dimension: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    weights = torch.softmax(values / temperature, dim=dimension)
    indices = torch.arange(
        values.shape[dimension],
        device=values.device,
        dtype=values.dtype,
    )
    return torch.sum(weights * indices, dim=dimension)


def reconstruct_canonical_spectrum(
    components: torch.Tensor,
    *,
    crop_size: int = 0,
) -> torch.Tensor:
    """Combine eight component profiles and min--max normalize each sample."""

    emission = components[:, 0:4, :].sum(dim=1)
    optical_depth = components[:, 4:8, :].sum(dim=1)
    spectrum = emission * torch.exp(-optical_depth)
    if crop_size > 0:
        spectrum = spectrum[..., crop_size:-crop_size]
    return _minmax(spectrum)


def _minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min(dim=-1, keepdim=True).values
    maximum = values.max(dim=-1, keepdim=True).values
    return (values - minimum) / (maximum - minimum + 1e-8)


def _component_relative_l1(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Historical component relative error, divided by the prediction."""

    return torch.abs((target - prediction) / (prediction + 1e-8)).mean()


def _masked_reconstruction_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_target: float = 0.1,
) -> torch.Tensor:
    """MSE plus target-relative L1 where the ground truth is at least 0.1."""

    mask = target >= minimum_target
    difference = prediction[mask] - target[mask]
    return (difference**2).mean() + (difference.abs() / target[mask]).mean()


def compute_training_loss(
    canonical_components: torch.Tensor,
    isotope_logits: torch.Tensor,
    predicted_shift_px: torch.Tensor,
    observed_reconstruction: torch.Tensor,
    applied_frame_shift_px: torch.Tensor,
    isotope_target: torch.Tensor,
    targets: torch.Tensor,
    isotope_bin_centres: torch.Tensor,
    *,
    crop_size: int = 50,
    weights: TrainingLossWeights | None = None,
) -> tuple[torch.Tensor, TrainingLossBreakdown]:
    """Compute the exact final-paper training loss.

    ``targets`` has shape ``[B, 10, L]``: four canonical emission
    components, four canonical absorption components, the observed-frame
    reconstruction, and the canonical reconstruction.  The dominance switch
    selects the lithium-7 D1 component below 50% lithium-6 and the lithium-6
    D1 component at or above 50%.
    """

    if weights is None:
        weights = TrainingLossWeights.full_objective()
    central = slice(crop_size, -crop_size) if crop_size > 0 else slice(None)
    batch_size = canonical_components.shape[0]
    batch_indices = torch.arange(batch_size, device=canonical_components.device)

    target_percentage = (isotope_target * isotope_bin_centres).sum(dim=1)
    lithium6_dominant = target_percentage >= 50.0
    emission_indices = torch.where(
        lithium6_dominant,
        torch.ones(batch_size, dtype=torch.long, device=canonical_components.device),
        torch.zeros(batch_size, dtype=torch.long, device=canonical_components.device),
    )
    absorption_indices = torch.where(
        lithium6_dominant,
        torch.full(
            (batch_size,),
            5,
            dtype=torch.long,
            device=canonical_components.device,
        ),
        torch.full(
            (batch_size,),
            4,
            dtype=torch.long,
            device=canonical_components.device,
        ),
    )
    emission_target = targets[batch_indices, emission_indices]
    emission_prediction = canonical_components[batch_indices, emission_indices]
    absorption_target = targets[batch_indices, absorption_indices]
    absorption_prediction = canonical_components[batch_indices, absorption_indices]

    component_mse = F.mse_loss(
        emission_target[:, central],
        emission_prediction[:, central],
    ) + F.mse_loss(
        absorption_target[:, central],
        absorption_prediction[:, central],
    )
    component_relative_l1 = _component_relative_l1(
        emission_target[:, central],
        emission_prediction[:, central],
    ) + _component_relative_l1(
        absorption_target[:, central],
        absorption_prediction[:, central],
    )

    maximum_emission_target = emission_target.max(dim=1).values
    maximum_emission_prediction = emission_prediction.max(dim=1).values
    maximum_absorption_target = absorption_target.max(dim=1).values
    maximum_absorption_prediction = absorption_prediction.max(dim=1).values
    peak_height = (
        torch.abs(maximum_emission_target - maximum_emission_prediction)
        / torch.clamp(maximum_emission_target.abs(), min=1e-4)
    ).mean() + (
        torch.abs(maximum_absorption_target - maximum_absorption_prediction)
        / torch.clamp(maximum_absorption_target.abs(), min=1e-4)
    ).mean()

    predicted_positions = soft_argmax(
        canonical_components[:, 0:7, :],
        dimension=-1,
    )
    target_positions = soft_argmax(targets[:, 0:7, :], dimension=-1)
    peak_position = (
        torch.abs(predicted_positions - target_positions) / 512.0
    ).mean()

    canonical_prediction = reconstruct_canonical_spectrum(
        canonical_components,
        crop_size=crop_size,
    )
    canonical_reconstruction = _masked_reconstruction_error(
        canonical_prediction,
        targets[:, 9, central],
    )
    observed_reconstruction_loss = _masked_reconstruction_error(
        _minmax(observed_reconstruction[..., central]),
        targets[:, 8, central],
    )
    direct_shift = F.smooth_l1_loss(
        predicted_shift_px,
        -applied_frame_shift_px.to(predicted_shift_px.dtype),
    )

    log_probabilities = F.log_softmax(isotope_logits, dim=1)
    probabilities = torch.softmax(isotope_logits, dim=1)
    isotope_kl = (
        isotope_target
        * (torch.log(isotope_target + 1e-8) - log_probabilities)
    ).sum(dim=1).mean()
    expected_percentage = torch.sum(
        probabilities * isotope_bin_centres,
        dim=1,
    )
    histogram = torch.histc(
        expected_percentage,
        bins=isotope_target.shape[1],
        min=0.0,
        max=100.0,
    )
    histogram_probability = histogram / histogram.sum()
    isotope_entropy_reciprocal = 1.0 / (
        -torch.sum(
            histogram_probability * torch.log(histogram_probability + 1e-8)
        )
        + 1e-8
    )

    total = (
        weights.component_profile * (component_mse + component_relative_l1)
        + weights.component_peak_height * peak_height
        + weights.component_peak_position * peak_position
        + weights.canonical_reconstruction * canonical_reconstruction
        + weights.observed_reconstruction * observed_reconstruction_loss
        + weights.direct_shift_supervision * direct_shift
        + weights.isotope_kl * isotope_kl
        + weights.isotope_entropy * isotope_entropy_reciprocal
    )
    breakdown = TrainingLossBreakdown(
        component_mse=component_mse,
        component_relative_l1=component_relative_l1,
        component_peak_height_relative_l1=peak_height,
        component_peak_position=peak_position,
        canonical_reconstruction=canonical_reconstruction,
        observed_reconstruction=observed_reconstruction_loss,
        direct_shift_supervision=direct_shift,
        isotope_kl=isotope_kl,
        isotope_entropy_reciprocal=isotope_entropy_reciprocal,
    )
    return total, breakdown


__all__ = [
    "TrainingLossBreakdown",
    "TrainingLossWeights",
    "compute_training_loss",
    "reconstruct_canonical_spectrum",
    "soft_argmax",
]
