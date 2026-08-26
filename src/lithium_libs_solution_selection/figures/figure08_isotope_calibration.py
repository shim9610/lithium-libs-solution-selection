"""Render held-out measured-isotope calibration for the two measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DATA_FILE = Path("measured_standards") / "measurement_outputs.npz"
CALIBRATION_LEVELS = np.array(
    [0, 15, 30, 45, 60, 75, 90, 100], dtype=float
)
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
}


def _aggregate_by_standard(
    data: np.lib.npyio.NpzFile, session: str
) -> np.ndarray:
    held_out = (data["session"] == session) & (~data["calibration_mask"])
    levels = np.unique(data["mix_n"][held_out])
    rows = []
    for level in levels:
        mask = held_out & (data["mix_n"] == level)
        rows.append(
            (
                level,
                float(data["true_li6"][mask][0]),
                float(np.mean(data["raw_pred_isotope"][mask])),
                float(np.mean(data["calibrated_pred_isotope"][mask])),
                float(
                    np.std(data["calibrated_pred_isotope"][mask], ddof=1)
                ),
                bool(np.isin(level, CALIBRATION_LEVELS)),
            )
        )
    dtype = [
        ("mix_n", float),
        ("true", float),
        ("raw_mean", float),
        ("cal_mean", float),
        ("cal_std", float),
        ("calibration_level", bool),
    ]
    return np.array(rows, dtype=dtype)


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if output_path.suffix.lower() == ".png":
        options["dpi"] = 300
    figure.savefig(output_path, **options)


def render(output_path: str | Path, data_root: str | Path) -> Path:
    """Render Figure 8 from the corrected warm-start evaluation set."""

    output_path = Path(output_path)
    input_path = Path(data_root) / DATA_FILE

    with np.load(input_path) as data, plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(PLOT_STYLE)

        figure = plt.figure(figsize=(7.2, 5.25), constrained_layout=True)
        try:
            grid = figure.add_gridspec(2, 2, height_ratios=(2.25, 1.0))
            axes_top = [figure.add_subplot(grid[0, i]) for i in range(2)]
            axes_bottom = [
                figure.add_subplot(grid[1, i], sharex=axes_top[i])
                for i in range(2)
            ]

            fit_color = "#2166AC"
            identity_color = "#222222"

            for column, session in enumerate(("A", "B")):
                rows = _aggregate_by_standard(data, session)
                unseen = ~rows["calibration_level"]
                seen = rows["calibration_level"]
                top = axes_top[column]
                bottom = axes_bottom[column]

                top.errorbar(
                    rows["true"][unseen],
                    rows["cal_mean"][unseen],
                    yerr=rows["cal_std"][unseen],
                    color=fit_color,
                    linestyle="none",
                    marker="o",
                    markersize=4.4,
                    capsize=2.0,
                    elinewidth=1.0,
                    label="Unseen standard",
                    zorder=4,
                )
                top.errorbar(
                    rows["true"][seen],
                    rows["cal_mean"][seen],
                    yerr=rows["cal_std"][seen],
                    color=fit_color,
                    markerfacecolor="white",
                    linestyle="none",
                    marker="s",
                    markersize=4.3,
                    capsize=2.0,
                    elinewidth=1.0,
                    label="Calibration-level held-out",
                    zorder=3,
                )
                top.plot(
                    [-5, 105],
                    [-5, 105],
                    color=identity_color,
                    linestyle=(0, (4, 2)),
                    linewidth=1.1,
                    label="Identity",
                    zorder=1,
                )
                top.set_title(f"Measure {session}")
                top.set_xlim(-5, 105)
                top.set_ylim(-5, 105)
                top.set_ylabel(
                    r"Fitted $^{6}$Li percentage (%)" if column == 0 else ""
                )
                top.grid(color="#D8D8D8", linewidth=0.55, alpha=0.7)
                top.tick_params(labelbottom=False)

                residual = rows["cal_mean"] - rows["true"]
                bottom.axhline(
                    0,
                    color=identity_color,
                    linestyle=(0, (4, 2)),
                    linewidth=1.0,
                )
                bottom.plot(
                    rows["true"][unseen],
                    residual[unseen],
                    color=fit_color,
                    linestyle="none",
                    marker="o",
                    markersize=4.4,
                )
                bottom.plot(
                    rows["true"][seen],
                    residual[seen],
                    color=fit_color,
                    markerfacecolor="white",
                    linestyle="none",
                    marker="s",
                    markersize=4.3,
                )
                bottom.set_ylim(-8, 8)
                bottom.set_xlabel(r"Standard $^{6}$Li percentage (%)")
                bottom.set_ylabel("Mean residual (pp)" if column == 0 else "")
                bottom.grid(color="#D8D8D8", linewidth=0.55, alpha=0.7)

            for label, axis in zip(
                ("a", "b", "c", "d"), axes_top + axes_bottom
            ):
                axis.text(
                    -0.12,
                    1.03,
                    label,
                    transform=axis.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="bottom",
                )

            handles = [
                Line2D(
                    [0],
                    [0],
                    color=identity_color,
                    linestyle=(0, (4, 2)),
                    linewidth=1.1,
                    label="Identity",
                ),
                Line2D(
                    [0],
                    [0],
                    color=fit_color,
                    linestyle="none",
                    marker="o",
                    markersize=4.4,
                    label="Unseen standard",
                ),
                Line2D(
                    [0],
                    [0],
                    color=fit_color,
                    markerfacecolor="white",
                    linestyle="none",
                    marker="s",
                    markersize=4.3,
                    label="Calibration-level held-out",
                ),
            ]
            figure.legend(
                handles=handles,
                loc="outside upper center",
                ncol=3,
                frameon=False,
                columnspacing=1.25,
                handletextpad=0.5,
            )
            _save_figure(figure, output_path)
        finally:
            plt.close(figure)

    return output_path
