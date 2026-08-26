"""Render the model-conditioned decomposition of measured lithium spectra."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DATA_DIRECTORY = Path("measured_decomposition")

CALIBRATION_A = -0.000000028581
CALIBRATION_B = 0.00167
CALIBRATION_C = 669.43069
PIXEL_START = 577.0
PIXEL_END = 1088.0
OUTPUT_LENGTH = 512

OBSERVED_COLOR = "0.18"
RECONSTRUCTION_COLOR = "#D55E00"
TRANSITION_COLORS = {
    "Li7-D1": "#0072B2",
    "Li6-D1": "#D55E00",
    "Li7-D2": "#009E73",
    "Li6-D2": "#CC79A7",
}
TRANSITION_LABELS = {
    "Li7-D1": r"$^{7}$Li D1",
    "Li6-D1": r"$^{6}$Li D1",
    "Li7-D2": r"$^{7}$Li D2",
    "Li6-D2": r"$^{6}$Li D2",
}


def _wavelength_from_grid(grid: np.ndarray) -> np.ndarray:
    detector_pixel = PIXEL_START + (PIXEL_END - PIXEL_START) * grid / (
        OUTPUT_LENGTH - 1
    )
    return (
        CALIBRATION_A * detector_pixel**2
        + CALIBRATION_B * detector_pixel
        + CALIBRATION_C
    )


def _load_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or reader.fieldnames is None:
        raise ValueError(f"No tabular data found in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in reader.fieldnames
    }


def _load_standard_compositions(path: Path) -> dict[str, dict[int, float]]:
    compositions: dict[str, dict[int, float]] = {"A": {}, "B": {}}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            level = int(float(row["nominal_li6_rich_powder_mass_percent"]))
            session = row["session"]
            if session in compositions and level in (0, 50, 100):
                compositions[session][level] = float(
                    row["lioh_mole_corrected_li6_atom_percent"]
                )

    missing = [
        f"Measure {session}, level {level}"
        for session in ("A", "B")
        for level in (0, 50, 100)
        if level not in compositions[session]
    ]
    if missing:
        raise ValueError(
            "The isotope-label table is missing required compositions: "
            + ", ".join(missing)
        )
    return compositions


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "facecolor": "white",
        "bbox_inches": "tight",
    }
    if output_path.suffix.lower() == ".png":
        options["dpi"] = 360
    figure.savefig(output_path, **options)


def render(output_path: str | Path, data_root: str | Path) -> Path:
    """Render Figure 7 from the published warm-start measurement artifacts.

    ``data_root`` is the package's ``artifacts/evaluation`` directory. The
    function reads ``measured_decomposition`` beneath it, including the
    LiOH-mole-fraction-corrected isotope label table.
    """

    output_path = Path(output_path)
    data_directory = Path(data_root) / DATA_DIRECTORY
    compositions = _load_standard_compositions(
        data_directory / "isotope_labels.csv"
    )

    with plt.rc_context():
        plt.rcdefaults()
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
        try:
            level = 50
            for panel, (axis, session) in enumerate(zip(axes, ("A", "B"))):
                values = _load_numeric_csv(
                    data_directory / f"component_fit_overlay_{session}.csv"
                )
                wavelength = _wavelength_from_grid(values["grid_index"])
                transmittance_axis = axis.twinx()

                axis.plot(
                    wavelength,
                    values[f"N{level}_observed"],
                    color=OBSERVED_COLOR,
                    lw=1.45,
                    label="Measured",
                    zorder=5,
                )
                axis.plot(
                    wavelength,
                    values[f"N{level}_reconstruction"],
                    color=RECONSTRUCTION_COLOR,
                    lw=1.25,
                    ls="--",
                    label="Reconstruction",
                    zorder=6,
                )
                axis.plot(
                    wavelength,
                    values[f"N{level}_emission_sum"],
                    color="0.35",
                    lw=1.05,
                    ls="-.",
                    label="Emission sum",
                    zorder=3,
                )

                for transition, color in TRANSITION_COLORS.items():
                    axis.plot(
                        wavelength,
                        values[f"N{level}_emission_{transition}"],
                        color=color,
                        lw=0.95,
                        alpha=0.95,
                    )
                    transmittance_axis.plot(
                        wavelength,
                        values[f"N{level}_transmittance_{transition}"],
                        color=color,
                        lw=0.9,
                        ls=":",
                        alpha=0.95,
                    )

                transmittance_axis.plot(
                    wavelength,
                    values[f"N{level}_transmittance_total"],
                    color="0.30",
                    lw=1.15,
                    ls=(0, (5, 2)),
                    zorder=4,
                )
                axis.set_title(
                    f"{chr(ord('a') + panel)}  Measure {session}, "
                    rf"$^{{6}}$Li = {compositions[session][level]:.2f}%",
                    loc="left",
                    fontsize=9,
                    fontweight="bold",
                )
                axis.set_xticks([670.70, 670.75, 670.80, 670.85])
                axis.set_xlabel("Wavelength (nm)", fontsize=8)
                axis.grid(alpha=0.17, lw=0.45)
                axis.tick_params(labelsize=7)
                transmittance_axis.set_ylim(0.0, 1.03)
                transmittance_axis.tick_params(
                    axis="y", labelsize=7, labelright=(panel == 1)
                )
                if panel == 0:
                    axis.set_ylabel("Normalized intensity", fontsize=8)
                    transmittance_axis.tick_params(labelright=False)
                else:
                    transmittance_axis.set_ylabel("Transmittance", fontsize=8)

            transition_handles = [
                Line2D(
                    [],
                    [],
                    color=color,
                    lw=1.2,
                    label=TRANSITION_LABELS[transition],
                )
                for transition, color in TRANSITION_COLORS.items()
            ]
            meaning_handles = [
                Line2D([], [], color=OBSERVED_COLOR, lw=1.5, label="Measured"),
                Line2D(
                    [],
                    [],
                    color=RECONSTRUCTION_COLOR,
                    lw=1.3,
                    ls="--",
                    label=r"Reconstruction $E\exp(-\sum_j A_j)$",
                ),
                Line2D(
                    [], [], color="0.35", lw=1.1, ls="-.", label=r"$\sum_j E_j$"
                ),
                Line2D([], [], color="0.25", lw=1.0, label=r"$E_j$ (solid)"),
                Line2D(
                    [],
                    [],
                    color="0.25",
                    lw=1.0,
                    ls=":",
                    label=r"$T_j$ (dotted)",
                ),
                Line2D(
                    [],
                    [],
                    color="0.30",
                    lw=1.2,
                    ls=(0, (5, 2)),
                    label=r"$\exp(-\sum_j A_j)$",
                ),
            ]
            figure.legend(
                handles=meaning_handles + transition_handles,
                loc="lower center",
                bbox_to_anchor=(0.52, -0.025),
                ncol=5,
                frameon=False,
                fontsize=7.2,
                columnspacing=1.25,
                handlelength=2.5,
            )
            figure.subplots_adjust(
                left=0.08,
                right=0.92,
                bottom=0.24,
                top=0.91,
                wspace=0.19,
            )
            _save_figure(figure, output_path)
        finally:
            plt.close(figure)

    return output_path
