"""Tests for representative CPU inference verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lithium_libs_solution_selection.inference_verification import (
    ISOTOPE_ABSOLUTE_TOLERANCE_PP,
    REPRESENTATIVE_INDICES,
    SHIFT_ABSOLUTE_TOLERANCE_PX,
    default_artifact_paths,
    verify_saved_inference,
)


def test_representative_indices_evenly_cover_the_frozen_dataset() -> None:
    indices = np.asarray(REPRESENTATIVE_INDICES)
    gaps = np.diff(indices)

    assert len(indices) == 17
    assert indices[0] == 0
    assert indices[-1] == 8399
    assert np.all(gaps > 0)
    assert int(gaps.max() - gaps.min()) <= 1


def test_released_predictions_are_reproduced_on_cpu(tmp_path: Path) -> None:
    checkpoint, evaluation = default_artifact_paths()
    if not checkpoint.is_file() or not evaluation.is_file():
        pytest.skip("optional checkpoint or measured evaluation artifact is absent")

    output_path = tmp_path / "inference-verification.json"
    report = verify_saved_inference(
        checkpoint,
        evaluation,
        output_json=output_path,
    )

    assert report.status == "PASS"
    assert report.passed
    assert report.checkpoint.hash_matches is True
    assert report.evaluation_data.hash_matches is True
    assert report.dataset_size == 8400
    assert report.representative_indices == REPRESENTATIVE_INDICES
    assert len(report.samples) == len(REPRESENTATIVE_INDICES)

    assert report.isotope.passed
    assert report.isotope.tolerance == ISOTOPE_ABSOLUTE_TOLERANCE_PP
    assert report.isotope.max_absolute_error <= ISOTOPE_ABSOLUTE_TOLERANCE_PP
    assert report.isotope.max_error_index in REPRESENTATIVE_INDICES

    assert report.shift.passed
    assert report.shift.tolerance == SHIFT_ABSOLUTE_TOLERANCE_PX
    assert report.shift.max_absolute_error <= SHIFT_ABSOLUTE_TOLERANCE_PX
    assert report.shift.max_error_index in REPRESENTATIVE_INDICES

    serialized = json.loads(output_path.read_text(encoding="utf-8"))
    assert serialized == report.to_dict()
    assert serialized["environment"]["device"] == "cpu"
    assert serialized["method"]["checkpoint_load"] == "strict"


def test_zero_tolerance_produces_a_structured_failure() -> None:
    checkpoint, evaluation = default_artifact_paths()
    if not checkpoint.is_file() or not evaluation.is_file():
        pytest.skip("optional checkpoint or measured evaluation artifact is absent")

    report = verify_saved_inference(
        checkpoint,
        evaluation,
        isotope_tolerance_pp=0.0,
        shift_tolerance_px=0.0,
    )

    assert report.status == "FAIL"
    assert not report.passed
    assert not report.isotope.passed
    assert not report.shift.passed
    assert report.isotope.max_absolute_error > 0.0
    assert report.shift.max_absolute_error > 0.0


def test_hash_mismatch_stops_before_checkpoint_deserialization() -> None:
    checkpoint, evaluation = default_artifact_paths()
    if not checkpoint.is_file() or not evaluation.is_file():
        pytest.skip("optional checkpoint or measured evaluation artifact is absent")

    with pytest.raises(RuntimeError, match="Refusing inference"):
        verify_saved_inference(
            checkpoint,
            evaluation,
            expected_checkpoint_sha256="0" * 64,
        )
