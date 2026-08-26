"""Render the measured joint Li–Ne residual-refinement comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DATA_FILE = Path("joint_refinement") / "per_spectrum.csv"
CALIBRATION_LEVELS = np.array(
    [0, 15, 30, 45, 60, 75, 90, 100], dtype=float
)
NETWORK_COLOR = "#0072B2"
JOINT_COLOR = "#D55E00"
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
}


def _read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "mapped_li_ne_joint"
        ]

    rows.sort(key=lambda row: int(row["array_index"]))
    if len(rows) != 260 or len({row["array_index"] for row in rows}) != 260:
        raise ValueError("Expected 260 unique joint Li–Ne result rows")
    if any(row["success"] != "True" for row in rows):
        raise ValueError("The joint Li–Ne results contain an unsuccessful fit")

    source_columns = {
        "array_index": "array_index",
        "session": "session",
        "mix_n": "mix_n",
        "true_li6": "true_li6",
        "initial_rmse": "initial_li_rmse",
        "initial_isotope_percent": "network_isotope_percent",
        "joint_rmse": "fitted_li_rmse",
        "joint_isotope_percent": "fitted_isotope_percent",
    }
    columns: dict[str, np.ndarray] = {}
    for target, source in source_columns.items():
        if target == "session":
            columns[target] = np.asarray([row[source] for row in rows])
        else:
            columns[target] = np.asarray(
                [float(row[source]) for row in rows], dtype=float
            )
    return columns


def _draw_half_violin(
    axis: plt.Axes,
    values: np.ndarray,
    position: int,
    side: str,
    color: str,
    rng: np.random.Generator,
) -> None:
    """Draw one horizontal half-violin with points, IQR, and median."""

    violins = axis.violinplot(
        [values],
        positions=[position],
        orientation="horizontal",
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = violins["bodies"][0]
    vertices = body.get_paths()[0].vertices
    if side == "upper":
        vertices[:, 1] = np.maximum(vertices[:, 1], position)
        point_y = position + rng.uniform(0.035, 0.20, size=values.size)
        summary_y = position + 0.105
    elif side == "lower":
        vertices[:, 1] = np.minimum(vertices[:, 1], position)
        point_y = position - rng.uniform(0.035, 0.20, size=values.size)
        summary_y = position - 0.105
    else:
        raise ValueError(f"Unknown violin side: {side!r}")

    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.18)
    body.set_linewidth(0.8)
    axis.scatter(
        values,
        point_y,
        s=7,
        alpha=0.25,
        color=color,
        edgecolors="none",
        zorder=2,
    )
    first_quartile, median, third_quartile = np.percentile(
        values, [25, 50, 75]
    )
    axis.plot(
        [first_quartile, third_quartile],
        [summary_y, summary_y],
        color=color,
        linewidth=4.5,
        solid_capstyle="round",
        zorder=4,
    )
    axis.scatter(
        [median],
        [summary_y],
        s=27,
        facecolor="white",
        edgecolor=color,
        linewidth=1.2,
        zorder=5,
    )


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
    """Render Figure 9 from all 260 successful measured joint fits."""

    output_path = Path(output_path)
    data = _read_columns(Path(data_root) / DATA_FILE)
    primary = ~np.isin(data["mix_n"], CALIBRATION_LEVELS)
    if not np.all(primary):
        raise ValueError(
            "Joint Li–Ne input unexpectedly includes calibration levels"
        )

    joint_rmse = data["joint_rmse"]
    joint_isotope_percent = data["joint_isotope_percent"]
    initial_error = np.abs(
        data["initial_isotope_percent"] - data["true_li6"]
    )
    joint_error = np.abs(joint_isotope_percent - data["true_li6"])

    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(PLOT_STYLE)
        figure, axes = plt.subplots(
            2, 2, figsize=(7.2, 5.8), constrained_layout=True
        )
        try:
            positions = [0, 1]
            axis = axes[0, 0]
            rng = np.random.default_rng(20260728)
            for position, session in zip(positions, ("A", "B")):
                mask = primary & (data["session"] == session)
                _draw_half_violin(
                    axis,
                    data["initial_rmse"][mask],
                    position,
                    "upper",
                    NETWORK_COLOR,
                    rng,
                )
                _draw_half_violin(
                    axis,
                    joint_rmse[mask],
                    position,
                    "lower",
                    JOINT_COLOR,
                    rng,
                )
            rmse_all = np.concatenate(
                (data["initial_rmse"][primary], joint_rmse[primary])
            )
            rmse_padding = 0.06 * np.ptp(rmse_all)
            axis.set_xlim(
                rmse_all.min() - rmse_padding,
                rmse_all.max() + rmse_padding,
            )
            axis.set_yticks(positions, labels=("Measure A", "Measure B"))
            axis.set_ylim(-0.48, 1.48)
            axis.set_xlabel("Reconstruction RMSE")
            axis.set_title("Reconstruction RMSE distribution")
            axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)

            axis = axes[0, 1]
            rng = np.random.default_rng(20260729)
            for position, session in zip(positions, ("A", "B")):
                mask = primary & (data["session"] == session)
                _draw_half_violin(
                    axis,
                    initial_error[mask],
                    position,
                    "upper",
                    NETWORK_COLOR,
                    rng,
                )
                _draw_half_violin(
                    axis,
                    joint_error[mask],
                    position,
                    "lower",
                    JOINT_COLOR,
                    rng,
                )
            error_all = np.concatenate(
                (initial_error[primary], joint_error[primary])
            )
            error_padding = 0.06 * np.ptp(error_all)
            axis.set_xlim(
                min(0, error_all.min() - error_padding),
                error_all.max() + error_padding,
            )
            axis.set_yticks(positions, labels=("Measure A", "Measure B"))
            axis.set_ylim(-0.48, 1.48)
            axis.set_xlabel(r"Absolute error in $^{6}$Li percentage (pp)")
            axis.set_title(r"$^{6}$Li absolute-error distribution")
            axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)

            for column, session in enumerate(("A", "B")):
                axis = axes[1, column]
                session_mask = primary & (data["session"] == session)
                levels = np.unique(data["mix_n"][session_mask])
                true = []
                initial_mean = []
                joint_mean = []
                for level in levels:
                    mask = session_mask & np.isclose(data["mix_n"], level)
                    true.append(data["true_li6"][mask][0])
                    initial_mean.append(
                        np.mean(data["initial_isotope_percent"][mask])
                    )
                    joint_mean.append(np.mean(joint_isotope_percent[mask]))
                true = np.asarray(true)
                initial_mean = np.asarray(initial_mean)
                joint_mean = np.asarray(joint_mean)

                axis.plot(
                    true,
                    initial_mean,
                    color=NETWORK_COLOR,
                    linestyle="-",
                    marker="^",
                    markersize=4.8,
                    markeredgecolor=NETWORK_COLOR,
                    markeredgewidth=0.6,
                    label="The network solution",
                )
                axis.plot(
                    true,
                    joint_mean,
                    color=JOINT_COLOR,
                    linestyle="-",
                    marker="o",
                    markersize=3.7,
                    label="Joint Li–Ne residual protocol (shared shift)",
                )
                axis.plot(
                    [-5, 105],
                    [-5, 105],
                    color="#222222",
                    linestyle=(0, (4, 2)),
                    linewidth=1,
                    label="Identity",
                )
                axis.set(xlim=(-5, 105), ylim=(-5, 105))
                axis.set_xlabel(r"Standard $^{6}$Li percentage (%)")
                axis.set_ylabel(
                    r"Mean fitted $^{6}$Li percentage (%)"
                    if column == 0
                    else ""
                )
                axis.set_title(f"Measure {session}: unseen standards")
                axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)

            for label, axis in zip(("a", "b", "c", "d"), axes.ravel()):
                axis.text(
                    0.015,
                    0.985,
                    label,
                    transform=axis.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.80,
                        "pad": 0.8,
                    },
                )

            method_handles = [
                Line2D(
                    [0],
                    [0],
                    color=NETWORK_COLOR,
                    linestyle="-",
                    marker="^",
                    markersize=4.8,
                    markeredgecolor=NETWORK_COLOR,
                    markeredgewidth=0.6,
                    label="The network solution",
                ),
                Line2D(
                    [0],
                    [0],
                    color=JOINT_COLOR,
                    linestyle="-",
                    marker="o",
                    markersize=3.7,
                    label="Joint Li–Ne residual protocol (shared shift)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#222222",
                    linestyle=(0, (4, 2)),
                    label="Identity",
                ),
            ]
            figure.legend(
                handles=method_handles,
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
