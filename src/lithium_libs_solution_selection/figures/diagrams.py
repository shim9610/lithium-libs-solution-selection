"""Render the paper's diagrams from their published source assets."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


VECTOR_SOURCES = {
    1: "figure_01_reference_conditioned_fitting.svg",
    2: "figure_02_training_gradient_routing.svg",
}
APPROVED_STATIC_RASTER = "figure_03_measurement_setup.png"


class DiagramRenderError(RuntimeError):
    """Raised when a diagram cannot be rendered from the published assets."""


def _imagemagick_executable() -> str:
    executable = shutil.which("magick")
    if executable is None:
        raise DiagramRenderError(
            "ImageMagick's 'magick' executable is required to render "
            "Figures 1 and 2."
        )
    return executable


def render_vector_diagram(
    figure_number: int,
    output_path: str | Path,
    source_root: str | Path,
) -> Path:
    """Rasterize Figure 1 or 2 from its SVG using ImageMagick."""

    if figure_number not in VECTOR_SOURCES:
        raise ValueError("Only Figures 1 and 2 have reproducible vector sources")

    source_path = Path(source_root) / VECTOR_SOURCES[figure_number]
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [_imagemagick_executable(), str(source_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DiagramRenderError(
            f"ImageMagick failed to render Figure {figure_number}: {detail}"
        )
    if not output_path.is_file():
        raise DiagramRenderError(
            f"ImageMagick did not create the expected output: {output_path}"
        )
    return output_path


def copy_approved_measurement_setup(
    output_path: str | Path,
    reference_root: str | Path,
) -> Path:
    """Copy Figure 3's approved raster without claiming to re-rasterize it.

    The editable SVG is supplied for inspection, but the exact rasterizer and
    font environment used for the approved paper PNG are not recoverable. The
    public reproduction path therefore copies the approved PNG byte-for-byte.
    """

    source_path = Path(reference_root) / APPROVED_STATIC_RASTER
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != output_path.resolve():
        shutil.copyfile(source_path, output_path)
    return output_path


def render(
    figure_number: int,
    output_path: str | Path,
    source_root: str | Path,
    reference_root: str | Path | None = None,
) -> Path:
    """Produce one diagram using its honest public reproduction path.

    Figures 1 and 2 are rasterized from SVG with ImageMagick. Figure 3 is the
    approved static raster and requires ``reference_root``.
    """

    if figure_number in VECTOR_SOURCES:
        return render_vector_diagram(figure_number, output_path, source_root)
    if figure_number == 3:
        if reference_root is None:
            raise ValueError("reference_root is required for approved Figure 3")
        return copy_approved_measurement_setup(output_path, reference_root)
    raise ValueError(f"Unsupported diagram figure number: {figure_number}")
