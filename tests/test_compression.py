"""Small CPU unit checks; no released-model training or dataset generation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lithium_libs_solution_selection import distillation
from lithium_libs_solution_selection.checkpoints import export_checkpoint
from lithium_libs_solution_selection.cli import main
from lithium_libs_solution_selection.distillation import (
    DistillationConfig,
    compute_distillation_loss,
    run_distillation,
)
from lithium_libs_solution_selection.model import (
    ReferenceConditionedSpectrumModel,
    load_checkpoint,
    set_dropout,
)
from lithium_libs_solution_selection.training import build_training_model


ROOT = Path(__file__).resolve().parents[1]


def _tiny_model() -> ReferenceConditionedSpectrumModel:
    model = ReferenceConditionedSpectrumModel(
        base_dim=8,
        num_layers=1,
        num_heads=2,
        classifier_hidden_dims=(16, 8),
        data_length=32,
        internal_length=32,
    )
    set_dropout(model, 0)
    return model.eval()


def test_export_preserves_tensors_architecture_and_source(tmp_path: Path) -> None:
    model = _tiny_model()
    source = tmp_path / "training.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config,
            "optimizer_state_dict": {"sentinel": torch.ones(100)},
            "scheduler_state_dict": {"last_epoch": 1},
        },
        source,
    )
    original = source.read_bytes()
    destination = tmp_path / "inference.pth"
    report = export_checkpoint(source, destination)
    payload = torch.load(destination, weights_only=True)
    assert set(payload) == {
        "format",
        "schema_version",
        "model_config",
        "model_state_dict",
    }
    restored = load_checkpoint(destination)
    assert restored.model_config == model.model_config
    assert all(
        torch.equal(tensor, restored.state_dict()[name])
        for name, tensor in model.state_dict().items()
    )
    assert report.tensors_identical
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        export_checkpoint(source, destination)
    with pytest.raises(ValueError, match="source checkpoint"):
        export_checkpoint(source, source)


def test_hash_failure_prevents_deserialization(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "untrusted.pth"
    source.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("deserialized before checking hash")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_checkpoint(source, tmp_path / "output.pth", expected_sha256="0" * 64)
    assert not (tmp_path / "output.pth").exists()


def test_export_preserves_half_precision_bare_state(tmp_path: Path) -> None:
    model = _tiny_model().half()
    source = tmp_path / "half_state.pth"
    torch.save(model.state_dict(), source)
    destination = tmp_path / "half_inference.pth"
    export_checkpoint(source, destination, model_config=model.model_config)
    restored = torch.load(destination, weights_only=True)["model_state_dict"]
    for name, tensor in model.state_dict().items():
        assert restored[name].dtype == tensor.dtype
        assert torch.equal(restored[name], tensor)


def test_parameter_logits_path_preserves_public_forward() -> None:
    model = _tiny_model()
    inputs = torch.linspace(0, 1, 128).reshape(2, 2, 32)
    with torch.no_grad():
        normal = model(inputs)
        logits = model.predict_parameter_logits(inputs)
        decoded = model.decode_parameter_logits(logits)
    assert logits.shape == (2, 8, 300)
    assert all(torch.equal(left, right) for left, right in zip(normal, decoded))
    assert [value.shape for value in normal] == [(2, 8, 32), (2, 300), (2,), (2, 32)]


def test_student_configuration_and_parameter_count() -> None:
    data = json.loads((ROOT / "configs/distillation_small.json").read_text())
    config = DistillationConfig.from_dict(data)
    model = build_training_model(config.student)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_664_722
    assert model.classifier.fc1.out_features == 512
    assert model.classifier.fc2.out_features == 256
    assert model.classifier.fc3.out_features == 2400
    assert model.model_config["classifier_hidden_dims"] == (512, 256)


@pytest.mark.parametrize(
    "data",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"logits_weight": -1},
        {"supervised_weight": 0},
        {"student": {"batch_size": 0}},
        {"student": {"number_of_attention_heads": 7}},
        {"student": {"batch_size": 1.5}},
        {"student": {"random_seed": "42"}},
        {"student": {"normalization_range": [1, 0]}},
        {"student": {"classifier_hidden_dimensions": [0, 256]}},
        {"student": {"checkpoint_interval": 0}},
        {"unknown_option": True},
        {"student": {"objective": "component-supervision-ablation"}},
    ],
)
def test_invalid_distillation_settings_are_rejected(data) -> None:
    with pytest.raises(ValueError):
        DistillationConfig.from_dict(data)


def _outputs(value: float, requires_grad: bool = False):
    return tuple(
        torch.full(shape, value, requires_grad=requires_grad)
        for shape in (
            (2, 8, 512),
            (2, 300),
            (2,),
            (2, 512),
        )
    )


def test_distillation_detaches_teacher_and_keeps_supervised_gradients() -> None:
    config = replace(DistillationConfig(), components_weight=0.5, supervised_weight=2)
    student = torch.zeros(2, 8, 300, requires_grad=True)
    teacher = torch.linspace(-1, 1, 4800).reshape(2, 8, 300).requires_grad_()
    student_outputs = _outputs(0.5, True)
    teacher_outputs = _outputs(1.0, True)
    supervised = torch.tensor(3.0, requires_grad=True)
    total, terms = compute_distillation_loss(
        student, teacher, student_outputs, teacher_outputs, supervised, config
    )
    # Shift is normalized by +/-10 px; a 0.5-px difference gives 0.0025.
    assert terms["shift"].item() == pytest.approx(0.0025)
    assert terms["reconstruction"].item() == pytest.approx(0.25)
    total.backward()
    assert supervised.grad.item() == 2
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert student_outputs[2].grad is not None
    assert teacher.grad is None
    assert all(value.grad is None for value in teacher_outputs)


def test_identical_teacher_and_student_have_zero_kd_loss() -> None:
    logits = torch.zeros(2, 8, 300)
    outputs = _outputs(1.0)
    total, terms = compute_distillation_loss(
        logits,
        logits,
        outputs,
        outputs,
        torch.tensor(0.0),
        DistillationConfig(),
    )
    assert total.item() == pytest.approx(0.0, abs=1e-6)
    assert all(value.item() == pytest.approx(0.0, abs=1e-6) for value in terms.values())


def test_cli_dry_run_performs_no_training_or_writes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("dry-run invoked model loading or training")

    monkeypatch.setattr(distillation, "run_distillation", unexpected)
    monkeypatch.setattr(distillation, "load_checkpoint", unexpected)
    output = tmp_path / "experiment"
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "distill",
                "--config",
                str(ROOT / "configs/distillation_small.json"),
                "--output-dir",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["configured_adam_steps"] == 320000
    assert payload["configuration"]["student"]["base_dimension"] == 144
    assert not output.exists()


def test_cli_rejects_export_into_frozen_artifacts(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "export-checkpoint",
                "--output",
                str(ROOT / "artifacts/checkpoints/forbidden.pth"),
            ]
        )
        == 2
    )
    assert not (ROOT / "artifacts/checkpoints/forbidden.pth").exists()


def test_runner_orchestration_without_real_training(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise saving/selection with stub models, loaders and optimizer steps."""
    config = DistillationConfig.from_dict(
        {
            "student": {
                "batch_size": 1,
                "samples_per_online_update": 1,
                "online_updates": 2,
                "validation_samples": 1,
                "number_of_workers": 0,
                "checkpoint_interval": 1,
            }
        }
    )

    class StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.model_config = {
                "in_channels": 2,
                "num_classes": 300,
                "data_length": 512,
                "internal_length": 512,
                "norm_range": (0.0, 1.0),
                "shift_max_px": 10.0,
            }

    teacher, student = StubModel(), StubModel()
    calls = {"steps": 0, "validation": 0}

    class StubOptimizer:
        def __init__(self, parameters, lr):
            self.param_groups = [{"lr": lr}]

        def zero_grad(self, **kwargs):
            student.weight.grad = None

        def step(self):
            calls["steps"] += 1  # No parameter update or real optimizer execution.

        def state_dict(self):
            return {"stub_steps": calls["steps"]}

    class StubScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            pass

        def state_dict(self):
            return {}

    batch = (
        torch.zeros(1, 2, 512),
        torch.zeros(1, 300),
        torch.zeros(1),
        torch.zeros(1, 10, 512),
    )

    def fake_loss(student_model, teacher_model, batch, config, bins):
        assert not teacher_model.training
        assert not teacher_model.weight.requires_grad
        if student_model.training:
            total = student_model.weight.square()
            supervised = total
        else:
            calls["validation"] += 1
            supervised = torch.tensor(float(calls["validation"]))
            total = torch.tensor(3.0 - calls["validation"])
        terms = {
            name: supervised if name == "supervised" else torch.tensor(0.0)
            for name in (
                "supervised",
                "logits",
                "shift",
                "reconstruction",
                "components",
            )
        }
        return total, terms, torch.zeros(1)

    source = tmp_path / "teacher.pth"
    source.write_bytes(b"stub checkpoint")
    monkeypatch.setattr(distillation, "load_checkpoint", lambda *a, **k: teacher)
    monkeypatch.setattr(distillation, "build_training_model", lambda *a: student)
    monkeypatch.setattr(distillation, "_loader", lambda *a, **k: [batch])
    monkeypatch.setattr(distillation, "_batch_loss", fake_loss)
    monkeypatch.setattr(torch.optim, "Adam", StubOptimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", StubScheduler)
    output = tmp_path / "run"
    result = run_distillation(
        config=config, teacher_checkpoint=source, output_directory=output, device="cpu"
    )
    assert calls["steps"] == result.completed_adam_steps == 2
    assert result.best_validation_loss == 1.0
    checkpoint = torch.load(result.best_checkpoint, weights_only=True)
    # Physical validation worsens while total KD validation improves: keep update 1.
    assert checkpoint["completed_online_updates"] == 1
    inference = torch.load(output / "best_model_inference.pth", weights_only=True)
    assert "optimizer_state_dict" not in inference
    assert (output / "checkpoint_latest.pth").is_file()
    assert len((output / "distillation_history.csv").read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="new or empty"):
        run_distillation(
            config=config,
            teacher_checkpoint=source,
            output_directory=output,
            device="cpu",
        )
