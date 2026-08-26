"""Uniform entry points for reproducing the nine paper figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..paths import ProjectPaths


@dataclass(frozen=True)
class FigureSpec:
    """Public metadata needed to render and verify one paper figure."""

    number: int
    filename: str
    title: str
    required_files: tuple[str, ...]
    static_raster_source: bool = False


FIGURE_SPECS: dict[int, FigureSpec] = {
    1: FigureSpec(
        1,
        "figure_01_reference_conditioned_fitting.png",
        "Reference-conditioned spectral fitting",
        (
            "figure_sources/figure_01_reference_conditioned_fitting.svg",
            "reference_figures/figure_01_reference_conditioned_fitting.png",
        ),
    ),
    2: FigureSpec(
        2,
        "figure_02_training_gradient_routing.png",
        "Training objective and gradient routing",
        (
            "figure_sources/figure_02_training_gradient_routing.svg",
            "reference_figures/figure_02_training_gradient_routing.png",
        ),
    ),
    3: FigureSpec(
        3,
        "figure_03_measurement_setup.png",
        "Measurement setup",
        (
            "figure_sources/figure_03_measurement_setup.svg",
            "reference_figures/figure_03_measurement_setup.png",
        ),
        static_raster_source=True,
    ),
    4: FigureSpec(
        4,
        "figure_04_synthetic_benchmark.png",
        "Synthetic parameter-recovery benchmark",
        (
            "artifacts/evaluation/synthetic_benchmark/random_spectra.csv",
            "artifacts/evaluation/synthetic_benchmark/generation_seeds.npy",
            "artifacts/evaluation/synthetic_benchmark/sampling_metadata.json",
            "reference_figures/figure_04_synthetic_benchmark.png",
        ),
    ),
    5: FigureSpec(
        5,
        "figure_05_joint_li_ne_multistart.png",
        "Synthetic joint lithium-neon multistart comparison",
        (
            "artifacts/evaluation/synthetic_multistart/arrays.npz",
            "artifacts/evaluation/synthetic_multistart/baseline_runs.csv",
            "artifacts/evaluation/synthetic_multistart/joint_reconstruction.npy",
            "artifacts/evaluation/synthetic_multistart/joint_runs.csv",
            "artifacts/evaluation/synthetic_multistart/spectra.csv",
            "reference_figures/figure_05_joint_li_ne_multistart.png",
        ),
    ),
    6: FigureSpec(
        6,
        "figure_06_checkpoint_ablation.png",
        "Component-supervision checkpoint ablation",
        (
            "artifacts/evaluation/component_supervision/full_objective.npz",
            "artifacts/evaluation/component_supervision/without_component_supervision.npz",
            "reference_figures/figure_06_checkpoint_ablation.png",
        ),
    ),
    7: FigureSpec(
        7,
        "figure_07_model_conditioned_decomposition.png",
        "Model-conditioned measured decomposition",
        (
            "artifacts/evaluation/measured_decomposition/component_fit_overlay_A.csv",
            "artifacts/evaluation/measured_decomposition/component_fit_overlay_B.csv",
            "artifacts/evaluation/measured_decomposition/isotope_labels.csv",
            "artifacts/evaluation/measured_decomposition/obs_recon_overlap_A.csv",
            "artifacts/evaluation/measured_decomposition/obs_recon_overlap_B.csv",
            "reference_figures/figure_07_model_conditioned_decomposition.png",
        ),
    ),
    8: FigureSpec(
        8,
        "figure_08_measured_isotope_calibration.png",
        "Measured-isotope calibration",
        (
            "artifacts/evaluation/measured_standards/calibration.json",
            "artifacts/evaluation/measured_standards/isotope_labels.csv",
            "artifacts/evaluation/measured_standards/measurement_outputs.npz",
            "artifacts/evaluation/measured_standards/per_file.csv",
            "artifacts/evaluation/measured_standards/summary.csv",
            "reference_figures/figure_08_measured_isotope_calibration.png",
        ),
    ),
    9: FigureSpec(
        9,
        "figure_09_joint_li_ne_refinement.png",
        "Measured joint lithium-neon refinement",
        (
            "artifacts/evaluation/joint_refinement/per_spectrum.csv",
            "artifacts/evaluation/joint_refinement/summary.json",
            "reference_figures/figure_09_joint_li_ne_refinement.png",
        ),
    ),
}


def get_figure_spec(number: int) -> FigureSpec:
    """Return the immutable public specification for a figure number."""

    try:
        return FIGURE_SPECS[int(number)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Figure number must be between 1 and 9: {number}") from error


def render_figure(
    number: int,
    *,
    paths: ProjectPaths | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Render one figure through the package-wide stable interface."""

    project = paths or ProjectPaths.discover()
    spec = get_figure_spec(number)
    output = Path(output_path) if output_path else project.output_figures / spec.filename
    resolved_output = output.resolve()
    resolved_references = project.references.resolve()
    if resolved_output == resolved_references or resolved_references in resolved_output.parents:
        raise ValueError(
            "Generated figures cannot be written inside reference_figures"
        )

    if number <= 3:
        from . import diagrams

        return diagrams.render(
            number,
            resolved_output,
            project.figure_sources,
            reference_root=project.references,
        )
    if number == 4:
        from . import figure04_synthetic_benchmark

        return figure04_synthetic_benchmark.render(
            output,
            project.evaluation,
        )
    if number == 5:
        from . import figure05_joint_multistart

        return figure05_joint_multistart.render(
            output,
            project.evaluation,
        )
    if number == 6:
        from . import figure06_checkpoint_ablation

        return figure06_checkpoint_ablation.render(
            output,
            project.evaluation,
        )
    if number == 7:
        from . import figure07_decomposition

        return figure07_decomposition.render(output, project.evaluation)
    if number == 8:
        from . import figure08_isotope_calibration

        return figure08_isotope_calibration.render(output, project.evaluation)
    if number == 9:
        from . import figure09_joint_refinement

        return figure09_joint_refinement.render(output, project.evaluation)
    raise AssertionError("unreachable figure dispatch")


def render_figures(
    numbers: Iterable[int],
    *,
    paths: ProjectPaths | None = None,
) -> list[Path]:
    """Render figures in caller-supplied order and return their output paths."""

    project = paths or ProjectPaths.discover()
    return [render_figure(number, paths=project) for number in numbers]


__all__ = [
    "FIGURE_SPECS",
    "FigureSpec",
    "get_figure_spec",
    "render_figure",
    "render_figures",
]
