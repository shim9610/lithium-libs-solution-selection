"""Public artifact integrity and end-to-end figure reproduction tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from lithium_libs_solution_selection.artifacts import verify_artifacts
from lithium_libs_solution_selection.cli import main
from lithium_libs_solution_selection.figures import FIGURE_SPECS, render_figure
from lithium_libs_solution_selection.paths import ProjectPaths
from lithium_libs_solution_selection.verification import verify_figure


def test_every_frozen_file_matches_the_manifest() -> None:
    paths = ProjectPaths.discover()
    checks = verify_artifacts(paths.root)

    assert len(checks) == 39
    assert all(check.passed for check in checks)


def test_registry_matches_the_public_figure_manifest() -> None:
    paths = ProjectPaths.discover()
    manifest = json.loads(
        (paths.artifacts / "figure_manifest.json").read_text(encoding="utf-8")
    )
    entries = {int(entry["number"]): entry for entry in manifest["figures"]}

    assert set(entries) == set(FIGURE_SPECS) == set(range(1, 10))
    for number, spec in FIGURE_SPECS.items():
        entry = entries[number]
        assert entry["filename"] == spec.filename
        assert tuple(entry["required_files"]) == spec.required_files
        assert bool(entry.get("static_raster_source", False)) == (
            spec.static_raster_source
        )


def test_exact_first_verifier_distinguishes_bytes_from_pixels(
    tmp_path: Path,
) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[2:6, 2:6] = (32, 96, 192)
    image = Image.fromarray(pixels, mode="RGB")
    reference = tmp_path / "reference.png"
    generated = tmp_path / "generated.png"
    image.save(reference)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("container-note", "different bytes, same pixels")
    image.save(generated, pnginfo=metadata)

    check = verify_figure(reference, generated, semantic_ok=True)

    assert check.status == "PASS_PIXEL_EXACT"
    assert check.reason_code == "PNG_ANCILLARY_OR_COMPRESSION_ONLY"
    assert check.metrics.pixel_equal


def test_tolerance_fallback_rejects_changed_alpha(tmp_path: Path) -> None:
    rgb = np.full((8, 8, 3), (32, 96, 192), dtype=np.uint8)
    opaque = np.concatenate(
        (rgb, np.full((8, 8, 1), 255, dtype=np.uint8)),
        axis=-1,
    )
    transparent = opaque.copy()
    transparent[..., 3] = 0
    reference = tmp_path / "opaque.png"
    generated = tmp_path / "transparent.png"
    Image.fromarray(opaque).save(reference)
    Image.fromarray(transparent).save(generated)

    check = verify_figure(reference, generated, semantic_ok=True)

    assert check.status == "FAIL_VISUAL_UNKNOWN"
    assert check.metrics.changed_fraction_gt_8 == 1.0


def test_public_entry_points_refuse_to_overwrite_references(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover()
    protected = paths.references / FIGURE_SPECS[3].filename

    try:
        render_figure(3, paths=paths, output_path=protected)
    except ValueError as error:
        assert "reference_figures" in str(error)
    else:
        raise AssertionError("render_figure accepted a protected reference path")

    report = tmp_path / "should-not-exist.json"
    exit_code = main(
        [
            "--root",
            str(paths.root),
            "verify",
            "--figure",
            "3",
            "--generated-dir",
            str(paths.references),
            "--existing",
            "--report",
            str(report),
        ]
    )
    assert exit_code == 2
    assert not report.exists()


def test_all_nine_figures_reproduce_the_approved_images(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.discover()
    statuses: dict[int, str] = {}
    for number, spec in FIGURE_SPECS.items():
        generated = render_figure(
            number,
            paths=paths,
            output_path=tmp_path / spec.filename,
        )
        check = verify_figure(
            paths.references / spec.filename,
            generated,
            semantic_ok=True,
            static_raster_source=spec.static_raster_source,
        )
        statuses[number] = check.status

    assert statuses == {
        1: "PASS_PIXEL_EXACT",
        2: "PASS_PIXEL_EXACT",
        3: "PASS_BYTE_EXACT",
        4: "PASS_BYTE_EXACT",
        5: "PASS_BYTE_EXACT",
        6: "PASS_BYTE_EXACT",
        7: "PASS_BYTE_EXACT",
        8: "PASS_BYTE_EXACT",
        9: "PASS_BYTE_EXACT",
    }


def test_cli_artifact_entry_point(tmp_path: Path) -> None:
    paths = ProjectPaths.discover()
    report = tmp_path / "artifact-report.json"
    exit_code = main(
        [
            "--root",
            str(paths.root),
            "artifacts",
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert len(payload["files"]) == 39
