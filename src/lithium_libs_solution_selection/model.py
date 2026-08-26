"""Reference-conditioned spectrum model used for the published evaluation.

The network receives a lithium LIBS spectrum and a neon reference spectrum as
two aligned 512-sample channels.  A transformer encodes the lithium channel,
cross-attends to spatial features from the neon channel, and decodes a
physics-constrained lithium line decomposition plus an output-frame shift.

The module names intentionally match the released checkpoints.  This makes a
strict ``state_dict`` load possible while giving the public API stable,
descriptive class names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "in_channels": 2,
    "base_dim": 288,
    "norm_range": (0.0, 1.0),
    "num_classes": 300,
    "data_length": 512,
    "internal_length": 512,
    "num_heads": 16,
    "num_layers": 6,
    "patch_size": 8,
    "shift_max_px": 10.0,
}


def set_dropout(module: nn.Module, probability: float) -> None:
    """Set every dropout probability in ``module`` to the same value.

    ``nn.MultiheadAttention`` stores its probability as a plain attribute, so
    it is handled in addition to the regular dropout modules.
    """

    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("dropout probability must be between 0 and 1")

    for child in module.modules():
        if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            child.p = probability
        if isinstance(child, nn.MultiheadAttention):
            child.dropout = probability


def coordinate_scale(
    wavelength_nm: Sequence[float] | torch.Tensor,
    *,
    reference_wavelength_nm: float = 670.7760,
    coordinate_range: tuple[float, float] = (-1.0, 1.0),
    light_speed_nm_ghz: float = 299_792_458.0,
) -> tuple[float, float]:
    """Return decoder-coordinate units per nanometre and per gigahertz.

    The decoder uses an evenly spaced coordinate grid while the spectrometer
    wavelength calibration is generally nonlinear.  The returned local
    derivatives are linearly interpolated at ``reference_wavelength_nm``.
    This helper keeps physical-unit conversion tied to the supplied
    calibration rather than to the learned transition-spacing parameter.
    """

    wavelength = torch.as_tensor(wavelength_nm, dtype=torch.float64)
    if wavelength.ndim != 1 or wavelength.numel() < 3:
        raise ValueError("wavelength_nm must be a one-dimensional grid with at least 3 values")
    if not bool(torch.all(torch.diff(wavelength) > 0)):
        raise ValueError("wavelength_nm must be strictly increasing")

    reference = float(reference_wavelength_nm)
    if not float(wavelength[0]) <= reference <= float(wavelength[-1]):
        raise ValueError("reference_wavelength_nm must lie inside the wavelength grid")

    coordinate = torch.linspace(
        float(coordinate_range[0]),
        float(coordinate_range[1]),
        wavelength.numel(),
        dtype=torch.float64,
    )
    frequency_ghz = float(light_speed_nm_ghz) / wavelength
    coordinate_per_nm = torch.gradient(coordinate, spacing=(wavelength,))[0]
    coordinate_per_ghz = torch.abs(
        torch.gradient(coordinate, spacing=(frequency_ghz,))[0]
    )

    def interpolate(values: torch.Tensor) -> float:
        upper = int(torch.searchsorted(wavelength, reference, right=False))
        upper = min(max(upper, 1), wavelength.numel() - 1)
        lower = upper - 1
        span = wavelength[upper] - wavelength[lower]
        weight = (reference - wavelength[lower]) / span
        return float(values[lower] + weight * (values[upper] - values[lower]))

    return interpolate(coordinate_per_nm), interpolate(coordinate_per_ghz)


class _PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 5000):
        super().__init__()
        encoding = torch.zeros(max_length, dimension)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2).float()
            * (-math.log(10000.0) / dimension)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))


class _CrossAttentionLayer(nn.Module):
    def __init__(self, dimension: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dimension)
        self.norm_kv = nn.LayerNorm(dimension)
        self.cross_attn = nn.MultiheadAttention(
            dimension,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dimension)
        self.ff = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(tokens)
        keys_and_values = self.norm_kv(reference)
        attended, _ = self.cross_attn(
            query,
            keys_and_values,
            keys_and_values,
        )
        tokens = tokens + attended
        return tokens + self.ff(self.norm_ff(tokens))


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ReferenceEncoder(nn.Module):
    """Encode the neon channel as spatial tokens without batch statistics."""

    def __init__(self, base_dim: int = 288):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            _group_norm(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            _group_norm(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            _group_norm(256),
            nn.ReLU(),
            nn.Conv1d(256, base_dim, kernel_size=3, stride=2, padding=1),
            _group_norm(base_dim),
            nn.ReLU(),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.encoder(spectrum).transpose(1, 2)


class _ParameterHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2048)
        self.ln1 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.relu(self.ln1(self.fc1(features))))
        features = self.dropout(self.relu(self.ln2(self.fc2(features))))
        return self.fc3(features)


class ContinuousDecoder(nn.Module):
    """Map binned parameter logits to continuous lithium line parameters.

    Seven groups are decoded: normalized emission-component scale, peak
    optical depth, shared Gaussian standard deviation, a nonnegative
    core--edge Lorentzian HWHM increment, edge-region absorption Lorentzian
    HWHM, emission--absorption offset, and lithium-6 percentage.  The reported
    core-region HWHM is the increment plus the edge-region HWHM.  The four
    transition positions and relative strengths are constrained by the
    lithium doublet physics used during training.
    """

    def __init__(self, num_classes: int = 300, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer(
            "peak_spacing",
            torch.tensor(
                [0.0, 0.5129870129874444, 0.49025974026218505, 1.0],
                dtype=dtype,
            ),
        )
        self.center = nn.Parameter(torch.tensor(-0.02476484701037406921))

        parameter_ranges = (
            (1.0, 5.0),
            (0.0, 2.0),
            (1e-8, 0.05),
            (1e-8, 0.5001),
            (1e-8, 0.1001),
            (-0.02, 0.02),
            (0.0, 100.0),
        )
        self.register_buffer(
            "bin_centers",
            torch.stack(
                [torch.linspace(low, high, num_classes, dtype=dtype) for low, high in parameter_ranges]
            ),
        )
        self.register_buffer("D1_relative", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("D2_relative", torch.tensor(0.5, dtype=dtype))
        self.register_buffer("li6_strength_factor", torch.tensor(0.081, dtype=dtype))

    def forward(
        self,
        parameter_logits: torch.Tensor,
        spacing_parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(parameter_logits, dim=-1)
        parameters = torch.sum(
            probabilities * self.bin_centers.unsqueeze(0),
            dim=-1,
        )

        batch_size = parameters.size(0)
        emission_component_scale = parameters[:, 0]
        peak_optical_depth = parameters[:, 1]
        gaussian_standard_deviation = parameters[:, 2]
        core_region_lorentzian_hwhm = parameters[:, 3] + parameters[:, 4]
        edge_region_lorentzian_hwhm = parameters[:, 4]
        absorption_offset = parameters[:, 5]
        isotope_percentage = parameters[:, 6].detach()

        center = torch.sigmoid(self.center * 10) - 0.5
        spacing = torch.sigmoid(spacing_parameter) * 10
        emission_positions = center + self.peak_spacing * (spacing / 100)
        emission_positions = emission_positions.unsqueeze(0).expand(batch_size, -1)
        absorption_positions = emission_positions.detach() + absorption_offset.unsqueeze(-1)

        lithium_6_rate = isotope_percentage / 7.5
        lithium_7_rate = (100 - isotope_percentage) / 92.5
        amplitudes = torch.stack(
            (
                self.D1_relative * lithium_7_rate,
                self.D1_relative * self.li6_strength_factor * lithium_6_rate,
                self.D2_relative * lithium_7_rate,
                self.D2_relative * self.li6_strength_factor * lithium_6_rate,
            ),
            dim=1,
        )

        emission_parameters = torch.stack(
            (
                emission_positions,
                amplitudes,
                gaussian_standard_deviation.unsqueeze(-1).expand(batch_size, 4),
                core_region_lorentzian_hwhm.unsqueeze(-1).expand(batch_size, 4),
            ),
            dim=1,
        )
        absorption_parameters = torch.stack(
            (
                absorption_positions,
                amplitudes,
                gaussian_standard_deviation.unsqueeze(-1).expand(batch_size, 4),
                edge_region_lorentzian_hwhm.unsqueeze(-1).expand(batch_size, 4),
            ),
            dim=1,
        )
        return (
            emission_parameters,
            absorption_parameters,
            emission_component_scale.unsqueeze(-1),
            peak_optical_depth.unsqueeze(-1),
        )


class _LorentzianProfiles(nn.Module):
    def __init__(
        self,
        data_length: int,
        axis: tuple[int, int],
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.register_buffer("x_grid", torch.linspace(-1, 1, data_length, dtype=dtype))
        self.axis = axis
        self.pi = math.pi

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        position = parameters[:, self.axis[0], :]
        width = parameters[:, self.axis[1], :]
        difference = position.unsqueeze(-1) - self.x_grid.unsqueeze(0).unsqueeze(0)
        expanded_width = width.unsqueeze(-1)
        return (1.0 / self.pi) * expanded_width / (
            difference**2 + expanded_width**2
        )


class _VoigtProfiles(nn.Module):
    def __init__(
        self,
        data_length: int,
        internal_length: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.data_length = data_length
        self.internal_length = internal_length
        self.register_buffer("D1_relative", torch.tensor(1.0, dtype=dtype))
        self.register_buffer("D2_relative", torch.tensor(0.5, dtype=dtype))
        self.register_buffer("li6_strength_factor", torch.tensor(0.081, dtype=dtype))
        self.register_buffer("x_prime", torch.linspace(-1, 1, internal_length, dtype=dtype))
        self.lorentz_layer = _LorentzianProfiles(
            data_length,
            axis=(0, 3),
            dtype=dtype,
        )
        self.pi = 3.14159265359

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        lorentzian = self.lorentz_layer(parameters)
        gaussian_standard_deviation = parameters[:, 2, :].unsqueeze(-1) + 1e-10
        gaussian_kernel = torch.exp(
            -(self.x_prime.unsqueeze(0).unsqueeze(0) ** 2)
            / (2 * gaussian_standard_deviation**2)
        )
        gaussian_kernel = gaussian_kernel / (
            gaussian_standard_deviation * (2 * self.pi) ** 0.5
        )

        kernel_size = gaussian_kernel.size(-1)
        batch_size, line_count, _ = lorentzian.shape
        padded = F.pad(
            lorentzian,
            (kernel_size // 2, kernel_size // 2),
            value=0,
        )
        flattened = padded.reshape(1, batch_size * line_count, -1)
        kernels = gaussian_kernel.reshape(
            batch_size * line_count,
            1,
            kernel_size,
        ).flip(-1)
        convolved = F.conv1d(
            flattened,
            kernels,
            groups=batch_size * line_count,
        )
        result = convolved.view(batch_size, line_count, -1)
        result = result[:, :, : self.data_length]
        maximum = result.max(dim=2, keepdim=True)[0]
        result = (result + 1e-10) / (maximum + 1e-10)
        return result * parameters[:, 1, :].unsqueeze(-1)


class ReferenceConditionedSpectrumModel(nn.Module):
    """Two-channel, shift-aware lithium spectrum solution selector.

    Parameters default to the architecture used by both released checkpoints.

    ``forward`` returns ``(components, isotope_logits, shift_px,
    shifted_reconstruction)``.  ``components`` contains four emission followed
    by four absorption profiles in canonical coordinates.  Only the summed
    reconstruction is shifted into the observed input frame.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_dim: int = 288,
        norm_range: tuple[float, float] = (0.0, 1.0),
        num_classes: int = 300,
        data_length: int = 512,
        internal_length: int = 512,
        num_heads: int = 16,
        num_layers: int = 6,
        patch_size: int = 8,
        shift_max_px: float = 10.0,
    ):
        super().__init__()
        if in_channels != 2:
            raise ValueError("the released model requires exactly two input channels")
        if data_length % patch_size:
            raise ValueError("data_length must be divisible by patch_size")

        self.range = list(norm_range)
        self.data_length = data_length
        self.base_dim = base_dim
        self.patch_size = patch_size
        self.num_patches = data_length // patch_size
        self.num_classes = num_classes

        self.patch_embed = nn.Conv1d(
            1,
            base_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.ref_encoder = _ReferenceEncoder(base_dim=base_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, base_dim))
        self.pos_encoding = _PositionalEncoding(
            base_dim,
            max_length=self.num_patches + 3,
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": nn.TransformerEncoderLayer(
                            d_model=base_dim,
                            nhead=num_heads,
                            dim_feedforward=base_dim * 4,
                            dropout=0.1,
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        ),
                        "cross_attn": _CrossAttentionLayer(
                            base_dim,
                            num_heads=num_heads,
                            dropout=0.1,
                        ),
                    }
                )
            )

        self.final_norm = nn.LayerNorm(base_dim)
        self.space_parm = nn.Parameter(torch.tensor(1.01538944244384765625))
        self.classifier = _ParameterHead(base_dim, 8 * num_classes)
        self.get_predictvalue = ContinuousDecoder(num_classes=num_classes)
        self.voigt = _VoigtProfiles(data_length, internal_length, dtype=torch.float32)
        self.shift_max_px = float(shift_max_px)
        self.register_buffer(
            "shift_bins",
            torch.linspace(-self.shift_max_px, self.shift_max_px, num_classes),
        )

    def _decode_shift(self, shift_logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(shift_logits, dim=-1)
        return (probabilities * self.shift_bins.unsqueeze(0)).sum(dim=-1)

    def _frac_shift(self, spectrum: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = spectrum.shape
        output_positions = torch.arange(
            length,
            device=spectrum.device,
            dtype=spectrum.dtype,
        ).view(1, 1, length)
        source_positions = output_positions - shift_px.view(batch_size, 1, 1)
        lower = torch.floor(source_positions)
        fraction = source_positions - lower
        lower_indices = torch.remainder(lower, length).long().expand(
            batch_size,
            channels,
            length,
        )
        upper_indices = torch.remainder(lower + 1, length).long().expand(
            batch_size,
            channels,
            length,
        )
        return (1 - fraction) * torch.gather(
            spectrum,
            2,
            lower_indices,
        ) + fraction * torch.gather(spectrum, 2, upper_indices)

    def forward(
        self,
        inputs: torch.Tensor,
        mode: str = "train",
        norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del mode  # Retained for compatibility with the training-time call signature.
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape [batch, 2, length]")

        if norm:
            batch_size = inputs.shape[0]
            flattened = inputs.reshape(batch_size, -1)
            minimum = flattened.min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            maximum = flattened.max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            inputs = (inputs - minimum) / (maximum - minimum + 1e-8)
            inputs = inputs * (self.range[1] - self.range[0]) + self.range[0]

        lithium = inputs[:, 0:1, :]
        reference = inputs[:, 1:2, :]
        patch_tokens = self.patch_embed(lithium).transpose(1, 2)
        reference_features = self.ref_encoder(reference)
        class_tokens = self.cls_token.expand(inputs.size(0), -1, -1)
        tokens = torch.cat((class_tokens, patch_tokens), dim=1)
        tokens = tokens + self.pos_encoding.pe[:, : tokens.size(1), :]

        for layer in self.layers:
            tokens = layer["self_attn"](tokens)
            tokens = layer["cross_attn"](tokens, reference_features)
        tokens = self.final_norm(tokens)

        parameter_logits = self.classifier(tokens[:, 0, :]).view(
            -1,
            8,
            self.num_classes,
        )
        shift_px = self._decode_shift(parameter_logits[:, 7, :])
        (
            emission_parameters,
            absorption_parameters,
            emission_component_scale,
            peak_optical_depth,
        ) = (
            self.get_predictvalue(parameter_logits[:, :7, :], self.space_parm)
        )

        emission = self.voigt(emission_parameters)
        absorption = self.voigt(absorption_parameters)
        emission = (
            emission
            / emission.sum(dim=1).max(dim=1).values.unsqueeze(1).unsqueeze(2)
        ) * emission_component_scale.unsqueeze(2)
        absorption = (
            absorption
            / absorption.sum(dim=1).max(dim=1).values.unsqueeze(1).unsqueeze(2)
        ) * peak_optical_depth.unsqueeze(2)

        canonical_components = torch.cat((emission, absorption), dim=1)
        canonical_reconstruction = emission.sum(dim=1) * torch.exp(
            -absorption.sum(dim=1)
        )
        shifted_reconstruction = self._frac_shift(
            canonical_reconstruction.detach().unsqueeze(1),
            shift_px,
        ).squeeze(1)
        return (
            canonical_components,
            parameter_logits[:, 6, :],
            shift_px,
            shifted_reconstruction,
        )


def _state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a state_dict or a mapping containing model_state_dict")
    candidate = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(candidate, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    ):
        raise TypeError("checkpoint does not contain a valid model state_dict")
    return candidate


def load_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    device: str | torch.device = "cpu",
    model_config: Mapping[str, Any] | None = None,
    strict: bool = True,
    dropout: float | None = 0.0,
    eval_mode: bool = True,
) -> ReferenceConditionedSpectrumModel:
    """Instantiate the published architecture and load a checkpoint.

    The released files include training metadata and therefore require trusted
    pickle loading.  Only load checkpoint files obtained with this package or
    from another source you trust.
    """

    config = dict(DEFAULT_MODEL_CONFIG)
    if model_config is not None:
        config.update(model_config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = _state_dict_from_checkpoint(checkpoint)
    model = ReferenceConditionedSpectrumModel(**config)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict, checkpoint

    if dropout is not None:
        set_dropout(model, dropout)
    model.to(device)
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "ContinuousDecoder",
    "DEFAULT_MODEL_CONFIG",
    "ReferenceConditionedSpectrumModel",
    "coordinate_scale",
    "load_checkpoint",
    "set_dropout",
]
