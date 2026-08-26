"""Render the joint lithium–neon multistart comparison used as Figure 5."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("network", "li_only", "joint")
METHOD_LABELS = (
    "The network\nsolution",
    "Li-only residual\nprotocol\n(free shift)",
    "Joint Li–Ne\nresidual protocol\n(shared shift)",
)
METHOD_COLORS = ("#009E73", "#777777", "#D55E00")
LEGEND_LABELS = ("The network solution", "Li-only", "Joint Li–Ne")
PARAMETERS = (
    ("absorbance", "Peak optical depth"),
    ("gaussian_fwhm_GHz", "Gaussian linewidth (GHz)"),
    (
        "core_region_fwhm_GHz",
        "Lorentzian linewidth\n(core-region plasma; GHz)",
    ),
    (
        "edge_region_fwhm_GHz",
        "Lorentzian linewidth\n(edge-region plasma; GHz)",
    ),
    ("absorption_offset_pm", "E–A offset (pm)"),
    ("isotope_percent", "$^6$Li percentage (%)"),
    ("shift_pixel", "Shift (pixel)"),
)

def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    return _repository_root() / "artifacts" / "evaluation"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _select_rows(
    baseline: list[dict[str, str]], joint_rows: list[dict[str, str]]
) -> dict[str, dict[int, dict[str, str]]]:
    selected: dict[str, dict[int, dict[str, str]]] = {
        method: {} for method in METHODS
    }
    spectrum_ids = sorted({int(row["spectrum_id"]) for row in baseline})
    for spectrum_id in spectrum_ids:
        learned = next(
            row
            for row in baseline
            if int(row["spectrum_id"]) == spectrum_id
            and row["start_type"] == "learned"
        )
        random_rows = [
            row
            for row in baseline
            if int(row["spectrum_id"]) == spectrum_id
            and row["start_type"] == "random"
        ]
        joint_candidates = [
            row
            for row in joint_rows
            if int(row["spectrum_id"]) == spectrum_id
            and row["method"] == "li_ne_joint"
            and row["start_type"] == "random"
        ]
        if not random_rows or not joint_candidates:
            raise ValueError(
                f"Spectrum {spectrum_id} has incomplete multistart candidates."
            )
        selected["network"][spectrum_id] = learned
        selected["li_only"][spectrum_id] = min(
            random_rows, key=lambda row: float(row["fitted_rmse"])
        )
        selected["joint"][spectrum_id] = min(
            joint_candidates,
            key=lambda row: float(row["joint_objective_rmse"]),
        )
    return selected


def _row_value(method: str, row: Mapping[str, str], name: str) -> float:
    prefix = "initial" if method == "network" else "fitted"
    return float(row[f"{prefix}_{name}"])


def _row_rmse(method: str, row: Mapping[str, str]) -> float:
    if method == "network":
        return float(row["initial_rmse"])
    if method == "li_only":
        return float(row["fitted_rmse"])
    return float(row["fitted_li_rmse"])


def _minmax_row(values: np.ndarray) -> np.ndarray:
    lower = float(np.min(values))
    upper = float(np.max(values))
    return (values - lower) / (upper - lower + 1e-8)


def _joint_reconstruction(frozen_path: Path) -> np.ndarray:
    """Load the frozen fitted curve used by the approved figure."""

    if not frozen_path.is_file():
        raise FileNotFoundError(frozen_path)
    reconstruction = np.asarray(
        np.load(frozen_path, allow_pickle=False), dtype=np.float64
    ).squeeze()

    if reconstruction.shape != (512,):
        raise ValueError(
            "Joint reconstruction must contain exactly 512 samples; "
            f"received shape {reconstruction.shape}."
        )
    return reconstruction


def _paired_distribution(axis: plt.Axes, values: np.ndarray) -> None:
    positions = np.arange(3)
    violin = axis.violinplot(
        [values[:, index] for index in range(3)],
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], METHOD_COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.17)
    for spectrum_values in values:
        axis.plot(
            positions,
            spectrum_values,
            color="#999999",
            linewidth=0.65,
            alpha=0.35,
        )
    for position, color in zip(positions, METHOD_COLORS):
        data = values[:, position]
        axis.scatter(
            np.full(data.size, position),
            data,
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )
        q25, median, q75 = np.percentile(data, [25, 50, 75])
        axis.vlines(
            position, q25, q75, color="#222222", linewidth=3.0, zorder=4
        )
        axis.scatter(
            position,
            median,
            s=48,
            marker="D",
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.0,
            zorder=5,
        )
    axis.set_xticks(positions, METHOD_LABELS)
    axis.set_xlim(-0.55, 2.55)
    axis.set_ylabel("Li spectral RMSE")
    axis.set_title("Li measurement-space residual")


def render(
    output_path: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    """Render Figure 5 from frozen multistart results.

    The representative joint reconstruction is a frozen evaluation artifact;
    checkpoint interpretation is tested by the separate inference verifier.
    """

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Figure 5 output must be a PNG path.")
    source_root = _data_root(data_root) / "synthetic_multistart"
    baseline = _read_csv(source_root / "baseline_runs.csv")
    summaries = _read_csv(source_root / "spectra.csv")
    joint_rows = _read_csv(source_root / "joint_runs.csv")
    arrays_path = source_root / "arrays.npz"
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)

    selected = _select_rows(baseline, joint_rows)
    spectrum_ids = sorted(selected["network"])
    spectrum_count = len(spectrum_ids)
    rmse = np.column_stack(
        [
            [
                _row_rmse(method, selected[method][spectrum_id])
                for spectrum_id in spectrum_ids
            ]
            for method in METHODS
        ]
    )

    normalized_error = np.empty((3, spectrum_count, len(PARAMETERS)))
    for method_index, method in enumerate(METHODS):
        for spectrum_index, spectrum_id in enumerate(spectrum_ids):
            row = selected[method][spectrum_id]
            network_row = selected["network"][spectrum_id]
            for parameter_index, (name, _label) in enumerate(PARAMETERS):
                span = float(network_row[f"bound_upper_{name}"]) - float(
                    network_row[f"bound_lower_{name}"]
                )
                normalized_error[
                    method_index, spectrum_index, parameter_index
                ] = abs(
                    _row_value(method, row, name) - float(row[f"true_{name}"])
                ) / span
    normalized_mae = normalized_error.mean(axis=1)

    representative_summary = max(
        summaries, key=lambda row: float(row["near_best_isotope_span_pp"])
    )
    representative_id = int(representative_summary["spectrum_id"])
    if representative_id not in selected["network"]:
        raise ValueError(
            f"Representative spectrum {representative_id} is absent from runs."
        )
    network_row = selected["network"][representative_id]
    lithium_row = selected["li_only"][representative_id]
    joint_row = selected["joint"][representative_id]

    with np.load(arrays_path, allow_pickle=False) as arrays:
        observation = np.asarray(
            arrays["observation"][representative_id], dtype=np.float64
        )
        network_reconstruction = np.asarray(
            arrays["one_pass_reconstruction"][representative_id],
            dtype=np.float64,
        )
        match = (
            (arrays["spectrum_id"] == representative_id)
            & (arrays["start_type"] == "random")
            & (arrays["start_id"] == int(lithium_row["start_id"]))
        )
        matches = np.where(match)[0]
        if len(matches) != 1:
            raise ValueError(
                "Expected one Li-only reconstruction for the representative "
                f"spectrum; found {len(matches)}."
            )
        lithium_reconstruction = np.asarray(
            arrays["reconstruction"][matches[0]], dtype=np.float64
        )

    joint_reconstruction = _joint_reconstruction(
        source_root / "joint_reconstruction.npy"
    )

    figure = plt.figure(figsize=(11.0, 7.7))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.12))
    residual_axis = figure.add_subplot(grid[0, 0])
    error_axis = figure.add_subplot(grid[0, 1])
    spectrum_axis = figure.add_subplot(grid[1, :])
    _paired_distribution(residual_axis, rmse)

    maximum = max(0.20, float(normalized_mae.max()) * 1.02)
    heatmap = error_axis.imshow(
        normalized_mae,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=maximum,
        aspect="auto",
        interpolation="nearest",
    )
    error_axis.set_xticks(
        np.arange(len(PARAMETERS)),
        (
            "Peak optical\ndepth",
            "Gaussian\nlinewidth",
            "Core-region\nLorentzian\nlinewidth",
            "Edge-region\nLorentzian\nlinewidth",
            "E–A\noffset",
            "$^6$Li\npercentage",
            "Spectral\nshift",
        ),
        fontsize=6.1,
    )
    error_axis.set_yticks(np.arange(3), METHOD_LABELS, fontsize=7.8)
    error_axis.set_title("Normalized MAE against generator ground truth")
    error_axis.set_xticks(np.arange(-0.5, len(PARAMETERS), 1), minor=True)
    error_axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    error_axis.grid(which="minor", color="white", linewidth=1.5)
    error_axis.tick_params(which="minor", bottom=False, left=False)
    threshold = 0.56 * maximum
    for row_index in range(3):
        for column_index in range(len(PARAMETERS)):
            value = normalized_mae[row_index, column_index]
            error_axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="semibold",
                color="white" if value >= threshold else "#202020",
            )
    colorbar = figure.colorbar(
        heatmap,
        ax=error_axis,
        orientation="horizontal",
        fraction=0.075,
        pad=0.20,
        aspect=30,
    )
    colorbar.set_label("MAE / decoder span  (lower is better)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)

    pixel = np.arange(512)
    spectrum_axis.plot(
        pixel,
        _minmax_row(observation),
        color="#202020",
        linewidth=1.5,
        label="Noisy synthetic observation",
    )
    method_rows = (network_row, lithium_row, joint_row)
    reconstructions = (
        network_reconstruction,
        lithium_reconstruction,
        joint_reconstruction,
    )
    line_styles = (":", "--", "-.")
    for method, row, reconstruction, color, line_style in zip(
        METHODS,
        method_rows,
        reconstructions,
        METHOD_COLORS,
        line_styles,
    ):
        spectrum_axis.plot(
            pixel,
            reconstruction,
            color=color,
            linewidth=1.25,
            linestyle=line_style,
            label=(
                LEGEND_LABELS[METHODS.index(method)]
                + f" — {_row_value(method, row, 'isotope_percent'):.1f}%; "
                + f"{_row_value(method, row, 'shift_pixel'):.2f} px; "
                + f"{_row_rmse(method, row):.4f}"
            ),
        )
    spectrum_axis.set(
        xlabel="Pixel", ylabel="Normalized intensity", xlim=(140, 360)
    )
    spectrum_axis.set_title(
        "Representative spectrum: ground-truth "
        f"$^6$Li percentage = {float(network_row['true_isotope_percent']):.1f}%; "
        f"shift = {float(network_row['true_shift_pixel']):.2f} px",
        y=1.085,
        pad=0,
    )
    spectrum_axis.legend(
        frameon=False,
        fontsize=7.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        columnspacing=1.2,
        handlelength=2.5,
    )

    for label, axis in zip(
        "abc", (residual_axis, error_axis, spectrum_axis)
    ):
        axis.text(
            -0.09,
            1.075 if label == "c" else 1.02,
            label,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(alpha=0.22)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.95,
        wspace=0.28,
        hspace=0.46,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output
