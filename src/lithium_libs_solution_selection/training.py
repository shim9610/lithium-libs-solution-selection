"""End-to-end online training for the published model and objective."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .datasets import TensorSample, make_synthetic_dataloader
from .losses import TrainingLossWeights, compute_training_loss
from .model import ReferenceConditionedSpectrumModel, set_dropout


ObjectiveName = Literal["full", "component-supervision-ablation"]


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration of the final paper training protocol.

    One ``online_update`` creates 16,384 new spectra and performs 16 Adam
    steps at the paper batch size of 1,024.  The configured 20,000 online
    updates therefore contain 320,000 optimizer steps when run to completion.
    """

    base_dimension: int = 288
    number_of_attention_heads: int = 16
    number_of_attention_blocks: int = 6
    input_channels: int = 2
    output_length: int = 512
    internal_length: int = 512
    number_of_isotope_bins: int = 300
    normalization_range: tuple[float, float] = (0.0, 1.0)
    patch_size: int = 8
    maximum_shift_px: float = 10.0
    dropout_probability: float = 0.0

    batch_size: int = 1024
    online_updates: int = 20_000
    samples_per_online_update: int = 16_384
    validation_samples: int = 4_096
    learning_rate: float = 0.0006
    minimum_learning_rate: float = 1e-7
    number_of_workers: int = 8
    prefetch_factor: int = 4
    gradient_clip_norm: float = 1000.0
    crop_size: int = 50

    objective: ObjectiveName = "full"
    random_seed: int | None = None
    checkpoint_interval: int = 10
    permanent_checkpoint_interval: int = 1000

    def validate(self) -> None:
        if self.input_channels != 2:
            raise ValueError("the paper model requires two input channels")
        if self.output_length != 512 or self.internal_length != 512:
            raise ValueError("the paper objective and checkpoint require length 512")
        if self.number_of_isotope_bins != 300:
            raise ValueError("the paper isotope target requires 300 bins")
        if self.samples_per_online_update < self.batch_size:
            raise ValueError("samples_per_online_update must be at least batch_size")
        if self.samples_per_online_update % self.batch_size:
            raise ValueError("samples_per_online_update must be divisible by batch_size")
        if self.online_updates <= 0 or self.validation_samples <= 0:
            raise ValueError("online_updates and validation_samples must be positive")
        if self.number_of_workers < 0:
            raise ValueError("number_of_workers cannot be negative")
        if self.objective not in ("full", "component-supervision-ablation"):
            raise ValueError(f"unsupported objective: {self.objective}")

    @property
    def adam_steps_per_online_update(self) -> int:
        return self.samples_per_online_update // self.batch_size

    @property
    def configured_adam_steps(self) -> int:
        return self.online_updates * self.adam_steps_per_online_update

    @property
    def loss_weights(self) -> TrainingLossWeights:
        if self.objective == "full":
            return TrainingLossWeights.full_objective()
        return TrainingLossWeights.component_supervision_ablation()

    def smoke_test(self) -> "TrainingConfig":
        """Return a one-sample, one-update configuration for installation QA."""

        return replace(
            self,
            batch_size=1,
            online_updates=1,
            samples_per_online_update=1,
            validation_samples=1,
            number_of_workers=0,
            checkpoint_interval=1,
            permanent_checkpoint_interval=1,
            random_seed=0 if self.random_seed is None else self.random_seed,
        )


@dataclass(frozen=True)
class UpdateMetrics:
    online_update: int
    completed_adam_steps: int
    learning_rate: float
    train_total: float
    validation_total: float
    validation_lithium6_mae_pp: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingRunResult:
    output_directory: Path
    best_checkpoint: Path
    best_validation_loss: float
    completed_online_updates: int
    completed_adam_steps: int


def seed_random_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_model(config: TrainingConfig) -> ReferenceConditionedSpectrumModel:
    model = ReferenceConditionedSpectrumModel(
        in_channels=config.input_channels,
        base_dim=config.base_dimension,
        norm_range=config.normalization_range,
        num_classes=config.number_of_isotope_bins,
        data_length=config.output_length,
        internal_length=config.internal_length,
        num_heads=config.number_of_attention_heads,
        num_layers=config.number_of_attention_blocks,
        patch_size=config.patch_size,
        shift_max_px=config.maximum_shift_px,
    )
    set_dropout(model, config.dropout_probability)
    return model


def _move_batch(batch: TensorSample, device: torch.device) -> TensorSample:
    return tuple(
        tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch
    )  # type: ignore[return-value]


def _loss_for_batch(
    model: ReferenceConditionedSpectrumModel,
    batch: TensorSample,
    config: TrainingConfig,
    isotope_bin_centres: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, frame_shift, isotope_target, lithium6_percentage, targets = batch
    components, isotope_logits, predicted_shift, observed_reconstruction = model(inputs)
    loss, _ = compute_training_loss(
        components,
        isotope_logits,
        predicted_shift,
        observed_reconstruction,
        frame_shift,
        isotope_target,
        targets,
        isotope_bin_centres,
        crop_size=config.crop_size,
        weights=config.loss_weights,
    )
    predicted_percentage = torch.sum(
        torch.softmax(isotope_logits, dim=1) * isotope_bin_centres,
        dim=1,
    )
    return loss, predicted_percentage, lithium6_percentage


def _evaluate(
    model: ReferenceConditionedSpectrumModel,
    validation_loader: torch.utils.data.DataLoader[TensorSample],
    config: TrainingConfig,
    isotope_bin_centres: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in validation_loader:
            moved = _move_batch(batch, device)
            loss, prediction, target = _loss_for_batch(
                model,
                moved,
                config,
                isotope_bin_centres,
            )
            batch_size = int(target.numel())
            total_loss += float(loss.detach())
            total_absolute_error += float(torch.abs(prediction - target).sum())
            total_samples += batch_size
    return total_loss / len(validation_loader), total_absolute_error / total_samples


def _write_history(path: Path, history: list[UpdateMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(history[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in history)


def _checkpoint_payload(
    *,
    model: ReferenceConditionedSpectrumModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    metrics: UpdateMetrics,
) -> dict[str, object]:
    return {
        "online_update": metrics.online_update,
        "completed_online_updates": metrics.online_update + 1,
        "completed_adam_steps": metrics.completed_adam_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_config": asdict(config),
        "metrics": asdict(metrics),
    }


def run_training(
    *,
    config: TrainingConfig,
    output_directory: str | Path,
    device: str | torch.device | None = None,
) -> TrainingRunResult:
    """Run the paper training algorithm from a fresh initialization.

    The historical run did not record Python, NumPy, or PyTorch RNG states.
    Passing ``random_seed`` makes a new run repeatable, but does not claim to
    regenerate the released checkpoint bit for bit.
    """

    config.validate()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if config.random_seed is not None:
        seed_random_generators(config.random_seed)

    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    validation_loader = make_synthetic_dataloader(
        number_of_samples=config.validation_samples,
        batch_size=config.batch_size,
        output_length=config.output_length,
        number_of_isotope_bins=config.number_of_isotope_bins,
        normalization_range=config.normalization_range,
        shuffle=False,
        number_of_workers=config.number_of_workers,
        prefetch_factor=config.prefetch_factor,
        pregenerate=True,
    )
    model = build_training_model(config).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.online_updates,
        eta_min=config.minimum_learning_rate,
    )
    isotope_bin_centres = torch.linspace(
        0.0,
        100.0,
        config.number_of_isotope_bins,
        device=resolved_device,
    )

    best_validation_loss = float("inf")
    best_checkpoint = output / "best_model.pth"
    history: list[UpdateMetrics] = []
    completed_adam_steps = 0

    for online_update in range(config.online_updates):
        started = time.perf_counter()
        loader_generator = None
        if config.random_seed is not None:
            loader_generator = torch.Generator().manual_seed(
                config.random_seed + 1 + online_update
            )
        training_loader = make_synthetic_dataloader(
            number_of_samples=config.samples_per_online_update,
            batch_size=config.batch_size,
            output_length=config.output_length,
            number_of_isotope_bins=config.number_of_isotope_bins,
            normalization_range=config.normalization_range,
            shuffle=True,
            number_of_workers=config.number_of_workers,
            prefetch_factor=config.prefetch_factor,
            pregenerate=False,
            generator=loader_generator,
        )

        model.train()
        training_loss = 0.0
        for batch in training_loader:
            moved = _move_batch(batch, resolved_device)
            optimizer.zero_grad()
            loss, _, _ = _loss_for_batch(
                model,
                moved,
                config,
                isotope_bin_centres,
            )
            loss.backward()
            if online_update > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.gradient_clip_norm,
                )
            optimizer.step()
            training_loss += float(loss.detach())
            completed_adam_steps += 1
        training_loss /= len(training_loader)

        validation_loss, validation_mae = _evaluate(
            model,
            validation_loader,
            config,
            isotope_bin_centres,
            resolved_device,
        )
        scheduler.step()
        metrics = UpdateMetrics(
            online_update=online_update,
            completed_adam_steps=completed_adam_steps,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            train_total=training_loss,
            validation_total=validation_loss,
            validation_lithium6_mae_pp=validation_mae,
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(metrics)
        _write_history(output / "training_history.csv", history)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            metrics=metrics,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(payload, best_checkpoint)

        completed_updates = online_update + 1
        if completed_updates % config.checkpoint_interval == 0:
            if completed_updates % config.permanent_checkpoint_interval == 0:
                checkpoint_path = output / f"checkpoint_update_{completed_updates}.pth"
            else:
                checkpoint_path = output / "checkpoint_latest.pth"
            torch.save(payload, checkpoint_path)

    return TrainingRunResult(
        output_directory=output,
        best_checkpoint=best_checkpoint,
        best_validation_loss=best_validation_loss,
        completed_online_updates=config.online_updates,
        completed_adam_steps=completed_adam_steps,
    )


__all__ = [
    "ObjectiveName",
    "TrainingConfig",
    "TrainingRunResult",
    "UpdateMetrics",
    "build_training_model",
    "run_training",
    "seed_random_generators",
]
