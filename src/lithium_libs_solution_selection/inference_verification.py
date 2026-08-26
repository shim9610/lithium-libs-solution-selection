"""Re-run representative measured-spectrum inference on the CPU.

The verifier checks that the released checkpoint still interprets the frozen
preprocessed inputs as recorded in the evaluation artifact.  It deliberately
uses fixed, evenly spaced row indices so every run evaluates the same spectra
from the beginning, middle, and end of the dataset.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .artifacts import sha256_file
from .model import load_checkpoint
from .paths import ProjectPaths


REPRESENTATIVE_INDICES: tuple[int, ...] = (
    0,
    525,
    1050,
    1575,
    2100,
    2625,
    3150,
    3675,
    4200,
    4724,
    5249,
    5774,
    6299,
    6824,
    7349,
    7874,
    8399,
)

# The saved values were produced on a different numerical backend.  Across
# CPU runs with 1, 2, 4, 8, and 12 intra-op threads, the largest observed
# differences were 0.0019340516 percentage point and 0.0001444817 pixel.
# These limits retain factors of about five and seven respectively while
# remaining negligible relative to the 0--100 % and -10--10 px output ranges.
ISOTOPE_ABSOLUTE_TOLERANCE_PP = 0.01
SHIFT_ABSOLUTE_TOLERANCE_PX = 0.001
BASELINE_CPU_MAX_ISOTOPE_ERROR_PP = 0.0019340516
BASELINE_CPU_MAX_SHIFT_ERROR_PX = 0.0001444817

RELEASE_CHECKPOINT_SHA256 = (
    "13db17a026ab5826b98eb60b88eab0583430f4a97926244ac37940e4fe910ed4"
)
RELEASE_EVALUATION_SHA256 = (
    "9be87294a6ff4b379306a66b52c4a3bf94b92d147860f82eb85aaf236b7c05f8"
)


@dataclass(frozen=True)
class ArtifactIdentity:
    """Identity and integrity result for one verification input."""

    name: str
    size_bytes: int
    sha256: str
    expected_sha256: str | None
    hash_matches: bool | None


@dataclass(frozen=True)
class ErrorMetric:
    """Absolute-error summary for one model output."""

    name: str
    unit: str
    tolerance: float
    baseline_cpu_max_error: float
    max_absolute_error: float
    mean_absolute_error: float
    max_error_index: int
    expected_at_max_error: float
    observed_at_max_error: float
    passed: bool


@dataclass(frozen=True)
class SampleComparison:
    """Saved and recomputed values for one representative spectrum."""

    index: int
    expected_isotope_percentage: float
    observed_isotope_percentage: float
    isotope_absolute_error_pp: float
    expected_shift_pixel: float
    observed_shift_pixel: float
    shift_absolute_error_px: float


@dataclass(frozen=True)
class InferenceVerificationReport:
    """JSON-serializable result of representative CPU inference."""

    status: str
    passed: bool
    representative_indices: tuple[int, ...]
    dataset_size: int
    checkpoint: ArtifactIdentity
    evaluation_data: ArtifactIdentity
    isotope: ErrorMetric
    shift: ErrorMetric
    samples: tuple[SampleComparison, ...]
    environment: dict[str, Any]
    method: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a representation accepted by ``json.dumps``."""

        return {
            "status": self.status,
            "passed": self.passed,
            "representative_indices": list(self.representative_indices),
            "dataset_size": self.dataset_size,
            "checkpoint": asdict(self.checkpoint),
            "evaluation_data": asdict(self.evaluation_data),
            "isotope": asdict(self.isotope),
            "shift": asdict(self.shift),
            "samples": [asdict(sample) for sample in self.samples],
            "environment": dict(self.environment),
            "method": dict(self.method),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report without non-standard JSON values."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    def write_json(self, output_path: str | Path, *, indent: int = 2) -> Path:
        """Write the report as UTF-8 JSON and return the resolved path."""

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return destination


def default_artifact_paths() -> tuple[Path, Path]:
    """Return the release checkpoint and measured-evaluation artifact paths."""

    paths = ProjectPaths.discover()
    checkpoint = paths.checkpoints / "solution_selector.pth"
    evaluation = (
        paths.evaluation
        / "measured_standards"
        / "measurement_outputs.npz"
    )
    return checkpoint, evaluation


def _artifact_identity(path: Path, expected_sha256: str | None) -> ArtifactIdentity:
    actual_sha256 = sha256_file(path)
    return ArtifactIdentity(
        name=path.name,
        size_bytes=path.stat().st_size,
        sha256=actual_sha256,
        expected_sha256=expected_sha256,
        hash_matches=(
            None
            if expected_sha256 is None
            else actual_sha256.lower() == expected_sha256.lower()
        ),
    )


def _error_metric(
    *,
    name: str,
    unit: str,
    indices: np.ndarray,
    expected: np.ndarray,
    observed: np.ndarray,
    tolerance: float,
    baseline_cpu_max_error: float,
) -> ErrorMetric:
    errors = np.abs(
        observed.astype(np.float64, copy=False)
        - expected.astype(np.float64, copy=False)
    )
    position = int(np.argmax(errors))
    maximum = float(errors[position])
    return ErrorMetric(
        name=name,
        unit=unit,
        tolerance=float(tolerance),
        baseline_cpu_max_error=float(baseline_cpu_max_error),
        max_absolute_error=maximum,
        mean_absolute_error=float(np.mean(errors)),
        max_error_index=int(indices[position]),
        expected_at_max_error=float(expected[position]),
        observed_at_max_error=float(observed[position]),
        passed=maximum <= float(tolerance),
    )


def _environment() -> dict[str, Any]:
    return {
        "device": "cpu",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "byte_order": sys.byteorder,
    }


def verify_saved_inference(
    checkpoint_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    *,
    representative_indices: Sequence[int] = REPRESENTATIVE_INDICES,
    isotope_tolerance_pp: float = ISOTOPE_ABSOLUTE_TOLERANCE_PP,
    shift_tolerance_px: float = SHIFT_ABSOLUTE_TOLERANCE_PX,
    expected_checkpoint_sha256: str | None = RELEASE_CHECKPOINT_SHA256,
    expected_evaluation_sha256: str | None = RELEASE_EVALUATION_SHA256,
    output_json: str | Path | None = None,
) -> InferenceVerificationReport:
    """Recompute representative predictions and compare with saved values.

    Lithium-6 percentage is decoded exactly as in the evaluation pipeline: the
    softmax probabilities over 300 logits are multiplied by an evenly spaced
    0--100 percentage grid and summed.  Shift is taken directly from the
    model's differentiable shift expectation.

    Artifact hashes are part of the overall result by default.  Pass ``None``
    for an expected hash only when intentionally checking compatible external
    artifacts.
    """

    if isotope_tolerance_pp < 0 or shift_tolerance_px < 0:
        raise ValueError("verification tolerances must be non-negative")

    default_checkpoint, default_evaluation = default_artifact_paths()
    checkpoint = Path(checkpoint_path or default_checkpoint).resolve()
    evaluation = Path(evaluation_path or default_evaluation).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not evaluation.is_file():
        raise FileNotFoundError(f"evaluation data not found: {evaluation}")

    indices = np.asarray(tuple(representative_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("representative_indices must be a non-empty sequence")
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("representative_indices must be strictly increasing and non-negative")

    required_keys = {
        "calibrated_input",
        "calibrated_pred_isotope",
        "calibrated_pred_shift_pixel",
    }
    with np.load(evaluation, allow_pickle=False) as stored:
        missing = sorted(required_keys.difference(stored.files))
        if missing:
            raise KeyError(f"evaluation data is missing arrays: {', '.join(missing)}")

        inputs_array = stored["calibrated_input"]
        expected_isotope_array = stored["calibrated_pred_isotope"]
        expected_shift_array = stored["calibrated_pred_shift_pixel"]
        dataset_size = int(inputs_array.shape[0])

        if inputs_array.ndim != 3 or inputs_array.shape[1:] != (2, 512):
            raise ValueError("calibrated_input must have shape [samples, 2, 512]")
        if expected_isotope_array.shape != (dataset_size,):
            raise ValueError("calibrated_pred_isotope has an incompatible shape")
        if expected_shift_array.shape != (dataset_size,):
            raise ValueError("calibrated_pred_shift_pixel has an incompatible shape")
        if int(indices[-1]) >= dataset_size:
            raise IndexError("a representative index lies outside the evaluation data")

        inputs = np.ascontiguousarray(inputs_array[indices], dtype=np.float32)
        expected_isotope = np.asarray(
            expected_isotope_array[indices],
            dtype=np.float32,
        )
        expected_shift = np.asarray(
            expected_shift_array[indices],
            dtype=np.float32,
        )

    if not (
        np.isfinite(inputs).all()
        and np.isfinite(expected_isotope).all()
        and np.isfinite(expected_shift).all()
    ):
        raise ValueError("verification inputs and saved outputs must all be finite")

    checkpoint_identity = _artifact_identity(
        checkpoint,
        expected_checkpoint_sha256,
    )
    evaluation_identity = _artifact_identity(
        evaluation,
        expected_evaluation_sha256,
    )
    mismatched = [
        identity.name
        for identity in (checkpoint_identity, evaluation_identity)
        if identity.hash_matches is False
    ]
    if mismatched:
        joined = ", ".join(mismatched)
        raise RuntimeError(
            "Refusing inference because a frozen artifact hash changed: "
            f"{joined}"
        )

    model = load_checkpoint(checkpoint, device="cpu", strict=True)
    input_tensor = torch.from_numpy(inputs)
    with torch.inference_mode():
        _, isotope_logits, observed_shift_tensor, _ = model(input_tensor)
        isotope_bins = torch.linspace(
            0.0,
            100.0,
            300,
            dtype=isotope_logits.dtype,
            device=isotope_logits.device,
        )
        observed_isotope_tensor = (
            torch.softmax(isotope_logits, dim=-1)
            * isotope_bins.unsqueeze(0)
        ).sum(dim=-1)

    observed_isotope = observed_isotope_tensor.cpu().numpy()
    observed_shift = observed_shift_tensor.cpu().numpy()
    isotope_metric = _error_metric(
        name="lithium_6_percentage",
        unit="percentage_point",
        indices=indices,
        expected=expected_isotope,
        observed=observed_isotope,
        tolerance=isotope_tolerance_pp,
        baseline_cpu_max_error=BASELINE_CPU_MAX_ISOTOPE_ERROR_PP,
    )
    shift_metric = _error_metric(
        name="output_frame_shift",
        unit="pixel",
        indices=indices,
        expected=expected_shift,
        observed=observed_shift,
        tolerance=shift_tolerance_px,
        baseline_cpu_max_error=BASELINE_CPU_MAX_SHIFT_ERROR_PX,
    )

    isotope_errors = np.abs(
        observed_isotope.astype(np.float64)
        - expected_isotope.astype(np.float64)
    )
    shift_errors = np.abs(
        observed_shift.astype(np.float64)
        - expected_shift.astype(np.float64)
    )
    samples = tuple(
        SampleComparison(
            index=int(index),
            expected_isotope_percentage=float(expected_isotope[position]),
            observed_isotope_percentage=float(observed_isotope[position]),
            isotope_absolute_error_pp=float(isotope_errors[position]),
            expected_shift_pixel=float(expected_shift[position]),
            observed_shift_pixel=float(observed_shift[position]),
            shift_absolute_error_px=float(shift_errors[position]),
        )
        for position, index in enumerate(indices)
    )

    hashes_pass = all(
        identity.hash_matches is not False
        for identity in (checkpoint_identity, evaluation_identity)
    )
    passed = bool(isotope_metric.passed and shift_metric.passed and hashes_pass)
    report = InferenceVerificationReport(
        status="PASS" if passed else "FAIL",
        passed=passed,
        representative_indices=tuple(int(index) for index in indices),
        dataset_size=dataset_size,
        checkpoint=checkpoint_identity,
        evaluation_data=evaluation_identity,
        isotope=isotope_metric,
        shift=shift_metric,
        samples=samples,
        environment=_environment(),
        method={
            "isotope_decode": (
                "sum(softmax(logits) * linspace(0, 100, 300))"
            ),
            "shift_decode": "model shift-bin expectation",
            "inference_device": "cpu",
            "dropout_probability": 0.0,
            "checkpoint_load": "strict",
        },
    )
    if output_json is not None:
        report.write_json(output_json)
    return report


__all__ = [
    "ArtifactIdentity",
    "BASELINE_CPU_MAX_ISOTOPE_ERROR_PP",
    "BASELINE_CPU_MAX_SHIFT_ERROR_PX",
    "ErrorMetric",
    "ISOTOPE_ABSOLUTE_TOLERANCE_PP",
    "InferenceVerificationReport",
    "REPRESENTATIVE_INDICES",
    "SHIFT_ABSOLUTE_TOLERANCE_PX",
    "SampleComparison",
    "default_artifact_paths",
    "verify_saved_inference",
]
