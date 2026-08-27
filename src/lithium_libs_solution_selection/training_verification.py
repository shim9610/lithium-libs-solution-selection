"""Golden checks for the public simulator and training objective."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .losses import TrainingLossWeights, compute_training_loss
from .datasets import sample_to_tensors
from .model import ReferenceConditionedSpectrumModel, set_dropout
from .simulation import generate_training_sample


_GENERATOR_HASHES = {
    "input_spectra": "1d658e6dcff1ea666d965bbfe796ef4945acc7f03d9104841d7c68a40b99f60b",
    "applied_frame_shift_px": "1479c5ab7ea7a7064b4614a66d72d99f6963609563d97b78633fbde37853fb63",
    "isotope_distribution": "54ef91b697a9549cf9e934e977d89acfbe9009e0875bf8e1c74b38964a53f109",
    "lithium6_percentage": "d25dee69b6a3d9e47891bea25e843ef761179d65c193e08a5345c049d82739bc",
    "targets": "d3ae2d9ef71b4f2a61020156296e88e4b5d49b212c8ac630b085812340702295",
}
_LOSS_TOTAL = 5.334783554077148
_LOSS_TERMS = {
    "component_mse": 0.3308842182159424,
    "component_relative_l1": 2.1731696128845215,
    "component_peak_height_relative_l1": 0.056314073503017426,
    "component_peak_position": 0.0031900566536933184,
    "canonical_reconstruction": 0.9247961640357971,
    "observed_reconstruction": 0.9408042430877686,
    "isotope_kl": 0.905625581741333,
}
_ONE_STEP_LOSS = 6.795426368713379
_ONE_STEP_STATE_HASH = "bbbbbd9d07ab64718a76d63eeefee5c5e1abadf266e762e3b4844ba7b87634f2"


@dataclass(frozen=True)
class TrainingImplementationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class TrainingVerificationReport:
    status: str
    checks: tuple[TrainingImplementationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "checks": [asdict(check) for check in self.checks],
        }


def _array_hash(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _generator_checks() -> list[TrainingImplementationCheck]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(12345)
        np.random.seed(67890)
        sample = generate_training_sample()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    values = {
        "input_spectra": sample.input_spectra,
        "applied_frame_shift_px": sample.applied_frame_shift_px,
        "isotope_distribution": sample.isotope_distribution,
        "lithium6_percentage": sample.lithium6_percentage,
        "targets": sample.targets,
    }
    checks = []
    for name, expected in _GENERATOR_HASHES.items():
        actual = _array_hash(values[name])
        checks.append(
            TrainingImplementationCheck(
                name=f"generator.{name}",
                passed=actual == expected,
                detail=f"sha256={actual}",
            )
        )
    return checks


def _loss_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    batch_size, length, classes = 4, 512, 300
    components = torch.rand(batch_size, 8, length, requires_grad=True) + 0.05
    logits = torch.randn(batch_size, classes, requires_grad=True)
    # Preserve the frozen verification fixture's RNG positions after narrowing
    # the public objective signature to tensors used by the final loss.
    _ = torch.randn(batch_size)
    reconstruction = torch.rand(batch_size, length, requires_grad=True)
    _ = torch.randn(batch_size)
    isotope_target = torch.softmax(torch.randn(batch_size, classes), dim=1)
    targets = torch.rand(batch_size, 10, length) + 0.02
    bin_centres = torch.linspace(0.0, 100.0, classes)
    return (
        components,
        logits,
        reconstruction,
        isotope_target,
        targets,
        bin_centres,
    )


def _loss_checks() -> list[TrainingImplementationCheck]:
    torch_state = torch.get_rng_state()
    try:
        inputs = _loss_inputs()
        total, breakdown = compute_training_loss(*inputs)
        ablated, _ = compute_training_loss(
            *inputs,
            weights=TrainingLossWeights.component_supervision_ablation(),
        )
    finally:
        torch.set_rng_state(torch_state)

    total_value = float(total.detach())
    checks = [
        TrainingImplementationCheck(
            name="loss.total",
            passed=abs(total_value - _LOSS_TOTAL) <= 1e-7,
            detail=f"value={total_value:.12g}",
        )
    ]
    actual_terms = breakdown.detached_floats()
    for name, expected in _LOSS_TERMS.items():
        actual = actual_terms[name]
        checks.append(
            TrainingImplementationCheck(
                name=f"loss.{name}",
                passed=abs(actual - expected) <= 1e-7,
                detail=f"value={actual:.12g}",
            )
        )
    removed_auxiliaries = (
        breakdown.component_mse
        + breakdown.component_relative_l1
        + breakdown.component_peak_height_relative_l1
        + breakdown.component_peak_position
    )
    identity_error = abs(
        float(((total - ablated) - removed_auxiliaries).detach())
    )
    checks.append(
        TrainingImplementationCheck(
            name="loss.ablation_identity",
            passed=identity_error <= 1e-6,
            detail=f"absolute_error={identity_error:.12g}",
        )
    )
    return checks


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _optimizer_step_checks() -> list[TrainingImplementationCheck]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        torch.manual_seed(7)
        model = ReferenceConditionedSpectrumModel(
            in_channels=2,
            base_dim=32,
            norm_range=(0.0, 1.0),
            num_classes=300,
            data_length=512,
            internal_length=512,
            num_heads=4,
            num_layers=1,
            patch_size=8,
            shift_max_px=10.0,
        )
        set_dropout(model, 0.0)
        model.train()
        random.seed(11)
        np.random.seed(12)
        tensors = sample_to_tensors(generate_training_sample())
        batch = tuple(
            value.unsqueeze(0) if value.ndim > 0 else value.unsqueeze(0)
            for value in tensors
        )
        inputs, isotope_target, _, targets = batch
        bin_centres = torch.linspace(0.0, 100.0, 300)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
        outputs = model(inputs)
        loss, _ = compute_training_loss(
            outputs[0],
            outputs[1],
            outputs[3],
            isotope_target,
            targets,
            bin_centres,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        state_hash = _model_state_hash(model)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

    return [
        TrainingImplementationCheck(
            name="training.one_adam_step_loss",
            passed=abs(loss_value - _ONE_STEP_LOSS) <= 1e-7,
            detail=f"value={loss_value:.12g}",
        ),
        TrainingImplementationCheck(
            name="training.one_adam_step_state",
            passed=state_hash == _ONE_STEP_STATE_HASH,
            detail=f"sha256={state_hash}",
        ),
    ]


def verify_training_implementation() -> TrainingVerificationReport:
    """Compare public calculations with frozen-source golden values."""

    checks = tuple(_generator_checks() + _loss_checks() + _optimizer_step_checks())
    return TrainingVerificationReport(
        status="PASS" if all(check.passed for check in checks) else "FAIL",
        checks=checks,
    )


__all__ = [
    "TrainingImplementationCheck",
    "TrainingVerificationReport",
    "verify_training_implementation",
]
