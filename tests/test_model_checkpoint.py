"""Checkpoint compatibility and inference-contract tests."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from lithium_libs_solution_selection.model import (
    ContinuousDecoder,
    ReferenceConditionedSpectrumModel,
    coordinate_scale,
    load_checkpoint,
    set_dropout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "solution_selector.pth",
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "component_supervision_ablation.pth",
)


@pytest.mark.parametrize("checkpoint_path", CHECKPOINTS, ids=("solution-selector", "ablation"))
def test_released_checkpoint_loads_strictly(checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        pytest.skip(f"optional checkpoint is not present: {checkpoint_path.name}")

    model = load_checkpoint(checkpoint_path, device="cpu", strict=True)
    state = model.state_dict()

    assert len(state) == 201
    assert state["classifier.fc3.weight"].shape == (2400, 1024)
    assert state["classifier.ln1.weight"].shape == (2048,)
    assert state["ref_encoder.encoder.10.weight"].shape == (288,)
    assert state["shift_bins"].shape == (300,)
    assert not model.training
    assert all(
        module.p == 0.0
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    )

    del state, model
    gc.collect()


def test_checkpoint_forward_is_deterministic_and_preserves_output_semantics() -> None:
    checkpoint_path = CHECKPOINTS[0]
    if not checkpoint_path.is_file():
        pytest.skip(f"optional checkpoint is not present: {checkpoint_path.name}")

    model = load_checkpoint(checkpoint_path, device="cpu", strict=True)
    generator = torch.Generator(device="cpu").manual_seed(7)
    inputs = torch.rand((1, 2, 512), generator=generator)

    with torch.inference_mode():
        first = model(inputs)
        second = model(inputs.clone())

    assert len(first) == 4
    assert first[0].shape == (1, 8, 512)
    assert first[1].shape == (1, 300)
    assert first[2].shape == (1,)
    assert first[3].shape == (1, 512)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert all(bool(torch.isfinite(value).all()) for value in first)
    assert bool(torch.all(first[2].abs() <= model.shift_max_px))

    emission = first[0][:, :4, :]
    absorption = first[0][:, 4:, :]
    canonical = emission.sum(dim=1) * torch.exp(-absorption.sum(dim=1))
    expected = model._frac_shift(canonical.unsqueeze(1), first[2]).squeeze(1)
    torch.testing.assert_close(first[3], expected, rtol=0.0, atol=0.0)


def test_continuous_decoder_and_coordinate_scale_public_helpers() -> None:
    decoder = ContinuousDecoder(num_classes=5)
    logits = torch.zeros((2, 7, 5))
    spacing_parameter = torch.tensor(0.0)
    emission, absorption, emission_scale, peak_optical_depth = decoder(
        logits,
        spacing_parameter,
    )

    assert emission.shape == (2, 4, 4)
    assert absorption.shape == (2, 4, 4)
    assert emission_scale.shape == (2, 1)
    assert peak_optical_depth.shape == (2, 1)

    wavelength = torch.linspace(670.7, 670.85, 512, dtype=torch.float64)
    per_nm, per_ghz = coordinate_scale(wavelength)
    assert per_nm > 0.0
    assert per_ghz > 0.0


def test_published_architecture_parameter_count() -> None:
    model = ReferenceConditionedSpectrumModel()
    assert sum(parameter.numel() for parameter in model.parameters()) == 17_596_994

    model = ReferenceConditionedSpectrumModel(
        base_dim=32,
        num_classes=5,
        data_length=32,
        internal_length=32,
        num_heads=4,
        num_layers=1,
        patch_size=8,
    )
    set_dropout(model, 0.25)
    assert all(
        module.p == 0.25
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    )
