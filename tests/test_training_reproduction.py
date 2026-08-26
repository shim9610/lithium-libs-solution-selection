from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path

import pytest
import torch

from lithium_libs_solution_selection.losses import (
    TrainingLossWeights,
    compute_training_loss,
)
from lithium_libs_solution_selection.training import TrainingConfig, run_training
from lithium_libs_solution_selection.training_verification import (
    verify_training_implementation,
)


def _loss_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(101)
    batch, length, bins = 3, 512, 300
    components = torch.rand(batch, 8, length, requires_grad=True) + 0.1
    logits = torch.randn(batch, bins, requires_grad=True)
    shift = torch.randn(batch, requires_grad=True)
    shifted_reconstruction = torch.rand(batch, length, requires_grad=True)
    frame_shift = torch.randn(batch)
    isotope_target = torch.softmax(torch.randn(batch, bins), dim=1)
    targets = torch.rand(batch, 10, length) + 0.1
    bin_centres = torch.linspace(0.0, 100.0, bins)
    return (
        components,
        logits,
        shift,
        shifted_reconstruction,
        frame_shift,
        isotope_target,
        targets,
        bin_centres,
    )


def test_frozen_source_generator_and_loss_goldens() -> None:
    report = verify_training_implementation()
    assert report.passed, [check for check in report.checks if not check.passed]


def test_component_ablation_removes_only_four_auxiliary_terms() -> None:
    inputs = _loss_inputs()
    full, breakdown = compute_training_loss(*inputs)
    ablated, _ = compute_training_loss(
        *inputs,
        weights=TrainingLossWeights.component_supervision_ablation(),
    )
    expected_difference = (
        breakdown.component_mse
        + breakdown.component_relative_l1
        + breakdown.component_peak_height_relative_l1
        + breakdown.component_peak_position
    )
    torch.testing.assert_close(full - ablated, expected_difference)


def test_inactive_direct_shift_term_has_no_gradient() -> None:
    inputs = _loss_inputs()
    total, _ = compute_training_loss(*inputs)
    total.backward()
    predicted_shift = inputs[2]
    assert predicted_shift.grad is None or torch.count_nonzero(predicted_shift.grad) == 0


def test_paper_training_update_and_optimizer_counts() -> None:
    config = TrainingConfig()
    assert config.adam_steps_per_online_update == 16
    assert config.configured_adam_steps == 320_000
    assert config.validation_samples == 4096
    assert config.random_seed is None


def test_public_code_and_docs_have_no_development_version_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    version_prefix = "v"
    legacy_frame_switch = "gt" + "_frame"
    pattern = re.compile(
        r"\b"
        + version_prefix
        + r"[34]\b|"
        + "|".join(
            role + "_" + version_prefix + r"\d" for role in ("main", "model", "loss")
        )
        + "|"
        + legacy_frame_switch,
        flags=re.IGNORECASE,
    )
    candidates = [root / "README.md", *(root / "docs").glob("*.md")]
    candidates.extend((root / "src").rglob("*.py"))
    matches = {
        str(path.relative_to(root)): pattern.findall(path.read_text(encoding="utf-8"))
        for path in candidates
        if pattern.search(path.read_text(encoding="utf-8"))
    }
    assert matches == {}


@pytest.mark.slow
def test_training_smoke_writes_checkpoint_and_history(tmp_path) -> None:
    config = replace(
        TrainingConfig().smoke_test(),
        base_dimension=32,
        number_of_attention_heads=4,
        number_of_attention_blocks=1,
    )
    result = run_training(
        config=config,
        output_directory=tmp_path,
        device="cpu",
    )
    assert result.best_checkpoint.is_file()
    assert (tmp_path / "training_history.csv").is_file()
    assert (tmp_path / "training_config.json").is_file()
    assert result.completed_online_updates == 1
    assert result.completed_adam_steps == 1
