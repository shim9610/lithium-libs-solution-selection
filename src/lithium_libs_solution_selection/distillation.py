"""Online physics-supervised distillation into a smaller Li--Ne fitter."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .checkpoints import check_checkpoint_hash, inference_payload
from .datasets import TensorSample, make_synthetic_dataloader
from .losses import compute_training_loss
from .model import ReferenceConditionedSpectrumModel, load_checkpoint
from .training import (
    TrainingConfig,
    TrainingRunResult,
    build_training_model,
    seed_random_generators,
)


def _student_defaults() -> TrainingConfig:
    return TrainingConfig(
        base_dimension=144,
        number_of_attention_heads=8,
        number_of_attention_blocks=3,
        classifier_hidden_dimensions=(512, 256),
        batch_size=32,
        samples_per_online_update=512,
        validation_samples=1024,
        number_of_workers=4,
        random_seed=42,
    )


@dataclass(frozen=True)
class DistillationConfig:
    """Experiment defaults, not claims of a validated compression recipe."""

    student: TrainingConfig = field(default_factory=_student_defaults)
    temperature: float = 2.0
    supervised_weight: float = 1.0
    logits_weight: float = 1.0
    shift_weight: float = 1.0
    reconstruction_weight: float = 1.0
    components_weight: float = 0.0

    def validate(self) -> None:
        self.student.validate()
        if self.student.objective != "full":
            raise ValueError(
                "Distillation retains the full seven-term physics objective"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        weights = (
            self.supervised_weight,
            self.logits_weight,
            self.shift_weight,
            self.reconstruction_weight,
            self.components_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Distillation weights must be finite and nonnegative")
        if self.supervised_weight <= 0:
            raise ValueError(
                "supervised_weight must be positive to retain physical supervision"
            )
        if not any(value > 0 for value in weights[1:]):
            raise ValueError("At least one teacher loss must have a positive weight")
        student = self.student
        if not math.isfinite(student.maximum_shift_px) or student.maximum_shift_px <= 0:
            raise ValueError("maximum_shift_px must be finite and positive")
        if not 0 <= student.crop_size < student.output_length // 2:
            raise ValueError("crop_size must leave a nonempty reconstruction")
        if not math.isfinite(student.learning_rate) or student.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= student.minimum_learning_rate <= student.learning_rate:
            raise ValueError(
                "minimum_learning_rate must be between zero and learning_rate"
            )
        if (
            not math.isfinite(student.gradient_clip_norm)
            or student.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if student.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if not 0 <= student.dropout_probability <= 1:
            raise ValueError("dropout_probability must be between zero and one")
        if (
            len(student.normalization_range) != 2
            or not all(math.isfinite(value) for value in student.normalization_range)
            or student.normalization_range[0] >= student.normalization_range[1]
        ):
            raise ValueError(
                "normalization_range must contain two finite increasing values"
            )
        if student.random_seed is not None and (
            type(student.random_seed) is not int or not 0 <= student.random_seed < 2**63
        ):
            raise ValueError("random_seed must be null or an integer in [0, 2**63)")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DistillationConfig":
        """Merge a partial JSON configuration with student experiment defaults."""

        if not isinstance(values, Mapping):
            raise ValueError("Distillation configuration must be a JSON object")
        data = dict(values)
        student_data = data.pop("student", {})
        if not isinstance(student_data, Mapping):
            raise ValueError("student configuration must be an object")
        merged = {**asdict(_student_defaults()), **student_data}
        try:
            config = cls(student=TrainingConfig(**merged), **data)
            config.validate()
        except (TypeError, AttributeError) as error:
            raise ValueError(f"Invalid distillation configuration: {error}") from error
        return config


ModelOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_outputs: ModelOutputs,
    teacher_outputs: ModelOutputs,
    supervised_loss: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean KL over eight rows, normalized shift MSE, and reconstruction MSE.

    Teacher tensors are detached even for callers outside the training runner.
    The decoder's original detach points remain unchanged in student outputs.
    """

    config.validate()
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError(
            "Teacher and student logits must have matching [batch, rows, bins] shapes"
        )
    if student_logits.shape[1:] != (8, config.student.number_of_isotope_bins):
        raise ValueError("Distillation expects eight rows of 300 parameter bins")
    temperature = config.temperature
    terms = {
        "supervised": supervised_loss,
        "logits": F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits.detach() / temperature, dim=-1),
            reduction="none",
        )
        .sum(dim=-1)
        .mean()
        * temperature**2,
        "shift": F.mse_loss(
            student_outputs[2] / config.student.maximum_shift_px,
            teacher_outputs[2].detach() / config.student.maximum_shift_px,
        ),
        "reconstruction": F.mse_loss(
            student_outputs[3],
            teacher_outputs[3].detach(),
        ),
        "components": F.mse_loss(
            student_outputs[0],
            teacher_outputs[0].detach(),
        ),
    }
    total = sum(
        getattr(config, f"{name}_weight") * value for name, value in terms.items()
    )
    return total, terms


def _batch_loss(
    student: ReferenceConditionedSpectrumModel,
    teacher: ReferenceConditionedSpectrumModel,
    batch: TensorSample,
    config: DistillationConfig,
    bins: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    inputs, isotope_target, _, targets = batch
    student_logits = student.predict_parameter_logits(inputs)
    student_outputs = student.decode_parameter_logits(student_logits)
    with torch.no_grad():
        teacher_logits = teacher.predict_parameter_logits(inputs)
        teacher_outputs = teacher.decode_parameter_logits(teacher_logits)
    supervised, _ = compute_training_loss(
        student_outputs[0],
        student_outputs[1],
        student_outputs[3],
        isotope_target,
        targets,
        bins,
        crop_size=config.student.crop_size,
        weights=config.student.loss_weights,
    )
    total, terms = compute_distillation_loss(
        student_logits,
        teacher_logits,
        student_outputs,
        teacher_outputs,
        supervised,
        config,
    )
    prediction = (torch.softmax(student_outputs[1], dim=-1) * bins).sum(dim=-1)
    return total, terms, prediction


def _loader(config: TrainingConfig, *, validation: bool, update: int = 0):
    generator = None
    if config.random_seed is not None:
        generator = torch.Generator().manual_seed(config.random_seed + 1 + update)
    return make_synthetic_dataloader(
        number_of_samples=(
            config.validation_samples
            if validation
            else config.samples_per_online_update
        ),
        batch_size=config.batch_size,
        output_length=config.output_length,
        number_of_isotope_bins=config.number_of_isotope_bins,
        normalization_range=config.normalization_range,
        shuffle=not validation,
        number_of_workers=config.number_of_workers,
        prefetch_factor=config.prefetch_factor,
        pregenerate=validation,
        generator=generator,
    )


def run_distillation(
    *,
    config: DistillationConfig,
    teacher_checkpoint: str | Path,
    output_directory: str | Path,
    device: str | torch.device | None = None,
    expected_teacher_sha256: str | None = None,
) -> TrainingRunResult:
    """Train a fresh student; select its best model using supervised validation.

    Uses online synthetic training and a fixed synthetic validation set. No
    measured evaluation artifacts are used to train or select the student.
    """

    config.validate()
    settings = config.student
    output = Path(output_directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Distillation requires a new or empty output directory")
    source = Path(teacher_checkpoint).resolve()
    teacher_hash = check_checkpoint_hash(source, expected_teacher_sha256)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    teacher = (
        load_checkpoint(source, device=resolved_device).requires_grad_(False).eval()
    )
    # Width, depth, head count and patch size may differ; parameter semantics may not.
    expected_contract = {
        "in_channels": settings.input_channels,
        "num_classes": settings.number_of_isotope_bins,
        "data_length": settings.output_length,
        "internal_length": settings.internal_length,
        "norm_range": tuple(settings.normalization_range),
        "shift_max_px": settings.maximum_shift_px,
    }
    for name, expected in expected_contract.items():
        if teacher.model_config[name] != expected:
            raise ValueError(f"Teacher/student parameter contract differs: {name}")
    if settings.random_seed is not None:
        seed_random_generators(settings.random_seed)
    student = build_training_model(settings).to(resolved_device)
    optimizer = torch.optim.Adam(student.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.online_updates,
        eta_min=settings.minimum_learning_rate,
    )
    bins = torch.linspace(
        0, 100, settings.number_of_isotope_bins, device=resolved_device
    )
    validation_loader = _loader(settings, validation=True)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "path": str(source),
        "sha256": teacher_hash,
        "model_config": teacher.model_config,
    }
    (output / "distillation_config.json").write_text(
        json.dumps({"configuration": asdict(config), "teacher": provenance}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    best_validation = float("inf")
    best_checkpoint = output / "best_model.pth"
    completed_steps = 0
    for update in range(settings.online_updates):
        started = time.perf_counter()
        student.train()
        train_sum = 0.0
        train_samples = 0
        for batch in _loader(settings, validation=False, update=update):
            moved = tuple(
                t.to(resolved_device, non_blocking=resolved_device.type == "cuda")
                for t in batch
            )
            optimizer.zero_grad(set_to_none=True)
            total, _, _ = _batch_loss(student, teacher, moved, config, bins)
            if not bool(torch.isfinite(total)):
                raise RuntimeError("Nonfinite distillation training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                settings.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            count = int(batch[0].shape[0])
            train_sum += float(total.detach()) * count
            train_samples += count
            completed_steps += 1
        student.eval()
        validation_sum = {
            name: 0.0
            for name in (
                "total",
                "supervised",
                "logits",
                "shift",
                "reconstruction",
                "components",
                "isotope_mae_pp",
            )
        }
        validation_samples = 0
        with torch.no_grad():
            for batch in validation_loader:
                moved = tuple(t.to(resolved_device) for t in batch)
                total, terms, prediction = _batch_loss(
                    student, teacher, moved, config, bins
                )
                if not bool(torch.isfinite(total)):
                    raise RuntimeError("Nonfinite distillation validation loss")
                count = int(batch[0].shape[0])
                validation_sum["total"] += float(total) * count
                for name, value in terms.items():
                    validation_sum[name] += float(value) * count
                validation_sum["isotope_mae_pp"] += float(
                    (prediction - moved[2]).abs().sum()
                )
                validation_samples += count
        validation = {
            name: value / validation_samples for name, value in validation_sum.items()
        }
        scheduler.step()
        row = {
            "online_update": update + 1,
            "completed_adam_steps": completed_steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_total": train_sum / train_samples,
            **{f"validation_{name}": value for name, value in validation.items()},
            "elapsed_seconds": time.perf_counter() - started,
        }
        with (output / "distillation_history.csv").open(
            "a", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if update == 0:
                writer.writeheader()
            writer.writerow(row)
        payload = {
            "model_state_dict": student.state_dict(),
            "model_config": student.model_config,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "distillation_config": asdict(config),
            "teacher": provenance,
            "completed_online_updates": update + 1,
            "completed_adam_steps": completed_steps,
            "metrics": row,
        }
        # Select using the physical objective, not agreement with teacher alone.
        if validation["supervised"] < best_validation:
            best_validation = validation["supervised"]
            torch.save(payload, best_checkpoint)
            torch.save(inference_payload(student), output / "best_model_inference.pth")
        if (update + 1) % settings.checkpoint_interval == 0:
            name = (
                f"checkpoint_update_{update + 1}.pth"
                if (update + 1) % settings.permanent_checkpoint_interval == 0
                else "checkpoint_latest.pth"
            )
            torch.save(payload, output / name)
        print(
            f"Update {update + 1}/{settings.online_updates}: physical validation={validation['supervised']:.8g}, isotope MAE={validation['isotope_mae_pp']:.8g} pp",
            flush=True,
        )
    return TrainingRunResult(
        output_directory=output,
        best_checkpoint=best_checkpoint,
        best_validation_loss=best_validation,
        completed_online_updates=settings.online_updates,
        completed_adam_steps=completed_steps,
    )
