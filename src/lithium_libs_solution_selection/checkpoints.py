"""Export trusted training checkpoints as self-describing inference weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .artifacts import sha256_file
from .model import (
    ReferenceConditionedSpectrumModel,
    _model_config_from_checkpoint,
    _state_dict_from_checkpoint,
)


def check_checkpoint_hash(path: str | Path, expected_sha256: str | None) -> str:
    actual = sha256_file(Path(path))
    if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch; refusing to deserialize it")
    return actual


def inference_payload(
    model: ReferenceConditionedSpectrumModel,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Keep architecture and CPU tensors, excluding all optimizer/training state."""

    return {
        "format": "lithium-libs-inference",
        "schema_version": 1,
        "model_config": dict(model.model_config),
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in (
                model.state_dict() if state_dict is None else state_dict
            ).items()
        },
    }


@dataclass(frozen=True)
class CheckpointExportReport:
    source: str
    destination: str
    source_bytes: int
    exported_bytes: int
    source_sha256: str
    exported_sha256: str
    parameter_count: int
    tensors_identical: bool


def export_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> CheckpointExportReport:
    """Strip optimizer state without altering weights or overwriting any file.

    As with ``load_checkpoint``, the source must be trusted: historical training
    files use pickle objects that require ``weights_only=False``. The exported
    payload itself supports ``torch.load(..., weights_only=True)``.
    """

    original = Path(source).resolve()
    target = Path(destination).resolve()
    if original == target:
        raise ValueError("The source checkpoint cannot be its export destination")
    if target.exists():
        raise FileExistsError(f"Export destination already exists: {target}")
    source_hash = check_checkpoint_hash(original, expected_sha256)
    checkpoint = torch.load(original, map_location="cpu", weights_only=False)
    state = _state_dict_from_checkpoint(checkpoint)
    model = ReferenceConditionedSpectrumModel(
        **_model_config_from_checkpoint(checkpoint, model_config)
    )
    model.load_state_dict(state, strict=True)
    # Preserve original dtypes too; strict model loading may cast to FP32.
    payload = inference_payload(model, state)
    del state, checkpoint
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents an accidental overwrite after the check.
    with target.open("xb") as stream:
        torch.save(payload, stream)
    restored = torch.load(target, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    restored_state = restored["model_state_dict"]
    identical = state.keys() == restored_state.keys() and all(
        tensor.dtype == restored_state[name].dtype
        and torch.equal(tensor, restored_state[name])
        for name, tensor in state.items()
    )
    if not identical:
        raise RuntimeError("Exported tensors differ from the source model")
    return CheckpointExportReport(
        source=str(original),
        destination=str(target),
        source_bytes=original.stat().st_size,
        exported_bytes=target.stat().st_size,
        source_sha256=source_hash,
        exported_sha256=sha256_file(target),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tensors_identical=identical,
    )
