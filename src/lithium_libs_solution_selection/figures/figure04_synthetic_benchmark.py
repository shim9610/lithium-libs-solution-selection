"""Render the synthetic parameter-recovery benchmark used as Figure 4."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


GENERATED_COLOR = "#2C6F9E"
FITTED_COLOR = "#D55E00"
ERROR_COLOR = "#2C6F9E"
MEDIAN_COLOR = "#D55E00"
CORRELATION_COLOR = "#258A72"

PARAMETERS = (
    (
        "Intensity scale",
        "true_intensity_scale",
        "pred_intensity_scale",
        "normalized scale",
        1.0,
    ),
    (
        "Peak optical depth",
        "true_absorbance",
        "pred_absorbance",
        "optical depth",
        1.0,
    ),
    (
        "Gaussian linewidth",
        "true_gaussian_fwhm_li7_GHz",
        "pred_gaussian_fwhm_li7_GHz",
        "GHz",
        2.0,
    ),
    (
        "Core-region Lorentzian linewidth",
        "true_lorentz_hot_total_fwhm_GHz",
        "pred_lorentz_hot_total_fwhm_GHz",
        "GHz",
        1.0,
    ),
    (
        "Edge-region Lorentzian linewidth",
        "true_lorentz_cold_fwhm_GHz",
        "pred_lorentz_cold_fwhm_GHz",
        "GHz",
        1.0,
    ),
    (
        "Emission–absorption offset",
        "true_absorption_offset_nm",
        "pred_absorption_offset_nm",
        "pm",
        1000.0,
    ),
    (
        r"$^6$Li percentage",
        "true_isotope_percent",
        "pred_isotope_percent",
        "%",
        1.0,
    ),
    (
        "Spectral shift",
        "true_shift_pixel",
        "pred_shift_pixel",
        "pixel",
        1.0,
    ),
)

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 450,
    "svg.fonttype": "none",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    return _repository_root() / "artifacts" / "evaluation"


def _load_data(path: Path) -> dict[str, np.ndarray]:
    columns = {"attenuation_fraction_true"}
    for _, true_key, fitted_key, _, _ in PARAMETERS:
        columns.update((true_key, fitted_key))

    raw = {key: [] for key in columns}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = columns.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Synthetic benchmark is missing columns: {sorted(missing)}"
            )
        for row in reader:
            for key in columns:
                raw[key].append(float(row[key]))

    data = {key: np.asarray(values, dtype=float) for key, values in raw.items()}
    row_count = len(data["attenuation_fraction_true"])
    if row_count < 240:
        raise ValueError(
            "Synthetic benchmark must contain at least 240 spectra for the "
            "published deterministic point sample."
        )
    return data


def _violin_panel(
    axis: plt.Axes,
    generated: np.ndarray,
    fitted: np.ndarray,
    title: str,
    unit: str,
    panel_label: str,
) -> None:
    rng = np.random.default_rng(7100 + ord(panel_label))
    for values, side, color in (
        (generated, "upper", GENERATED_COLOR),
        (fitted, "lower", FITTED_COLOR),
    ):
        parts = axis.violinplot(
            [values],
            positions=[0.0],
            orientation="horizontal",
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=180,
            bw_method=0.18,
        )
        body = parts["bodies"][0]
        vertices = body.get_paths()[0].vertices
        if side == "upper":
            vertices[:, 1] = np.maximum(vertices[:, 1], 0.0)
            point_y = rng.uniform(0.035, 0.19, size=240)
            summary_y = 0.105
        else:
            vertices[:, 1] = np.minimum(vertices[:, 1], 0.0)
            point_y = -rng.uniform(0.035, 0.19, size=240)
            summary_y = -0.105

        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(0.8)
        selected = rng.choice(values.size, size=240, replace=False)
        axis.scatter(
            values[selected],
            point_y,
            s=4.5,
            alpha=0.22,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.plot(
            [q25, q75],
            [summary_y, summary_y],
            color=color,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=4,
        )
        axis.scatter(
            [median],
            [summary_y],
            s=17,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=5,
        )

    all_values = np.concatenate((generated, fitted))
    lower, upper = np.quantile(all_values, [0.001, 0.999])
    padding = 0.08 * max(upper - lower, np.finfo(float).eps)
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(-0.43, 0.43)
    axis.set_yticks([])
    axis.axhline(0.0, color="#777777", linewidth=0.55, alpha=0.5)
    axis.set_xlabel(unit)
    axis.set_title(title, pad=4)
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    axis.text(
        -0.17,
        1.03,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )


def _binned_metrics(
    minimum_transmittance: np.ndarray,
    generated_width: np.ndarray,
    fitted_width: np.ndarray,
) -> list[dict[str, float]]:
    error = fitted_width - generated_width
    edges = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, float]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            selected = (minimum_transmittance >= lower) & (
                minimum_transmittance <= upper
            )
        else:
            selected = (minimum_transmittance >= lower) & (
                minimum_transmittance < upper
            )
        if not np.any(selected):
            continue
        absolute_error = np.abs(error[selected])
        rows.append(
            {
                "mean_transmittance": float(
                    np.mean(minimum_transmittance[selected])
                ),
                "mae": float(np.mean(absolute_error)),
                "median_absolute_error": float(np.median(absolute_error)),
                "pearson_r": float(
                    np.corrcoef(
                        generated_width[selected], fitted_width[selected]
                    )[0, 1]
                ),
            }
        )
    return rows


def render(
    output_path: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    """Render Figure 4 from the frozen synthetic benchmark CSV."""

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Figure 4 output must be a PNG path.")
    source = _data_root(data_root) / "synthetic_benchmark" / "random_spectra.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = _load_data(source)

    with plt.rc_context(STYLE):
        figure = plt.figure(
            figsize=(18.0 / 2.54, 19.2 / 2.54), constrained_layout=True
        )
        grid = figure.add_gridspec(
            4, 3, height_ratios=[1.0, 1.0, 1.0, 1.18]
        )
        figure.get_layout_engine().set(
            w_pad=0.04, h_pad=0.04, wspace=0.10, hspace=0.12
        )

        wrapped_titles = {
            "Core-region Lorentzian linewidth": (
                "Core-region Lorentzian\nlinewidth"
            ),
            "Edge-region Lorentzian linewidth": (
                "Edge-region Lorentzian\nlinewidth"
            ),
            "Emission–absorption offset": "Emission–absorption\noffset",
        }
        for index, (title, true_key, fitted_key, unit, scale) in enumerate(
            PARAMETERS
        ):
            axis = figure.add_subplot(grid[index // 3, index % 3])
            _violin_panel(
                axis,
                data[true_key] * scale,
                data[fitted_key] * scale,
                wrapped_titles.get(title, title),
                unit,
                chr(ord("a") + index),
            )

        legend_axis = figure.add_subplot(grid[2, 2])
        legend_axis.axis("off")
        legend_axis.legend(
            handles=[
                Patch(
                    facecolor=GENERATED_COLOR,
                    alpha=0.52,
                    label="Generated",
                ),
                Patch(
                    facecolor=FITTED_COLOR,
                    alpha=0.52,
                    label="Fitted",
                ),
                Line2D(
                    [],
                    [],
                    color="#555555",
                    linewidth=4,
                    marker="o",
                    markerfacecolor="white",
                    label="IQR and median",
                ),
            ],
            frameon=False,
            loc="center",
            fontsize=8.0,
        )
        spectrum_count = len(data["attenuation_fraction_true"])
        legend_axis.text(
            0.5,
            0.25,
            f"Each violin contains all {spectrum_count:,} spectra.",
            ha="center",
            va="center",
            fontsize=7.0,
            transform=legend_axis.transAxes,
        )

        minimum_transmittance = 1.0 - data["attenuation_fraction_true"]
        generated_width = data["true_lorentz_cold_fwhm_GHz"]
        fitted_width = data["pred_lorentz_cold_fwhm_GHz"]
        rows = _binned_metrics(
            minimum_transmittance, generated_width, fitted_width
        )

        axis = figure.add_subplot(grid[3, :])
        x = np.asarray([row["mean_transmittance"] for row in rows])
        mae = np.asarray([row["mae"] for row in rows])
        median = np.asarray([row["median_absolute_error"] for row in rows])
        correlation = np.asarray([row["pearson_r"] for row in rows])
        axis.plot(x, mae, color=ERROR_COLOR, marker="o", label="MAE")
        axis.plot(
            x,
            median,
            color=MEDIAN_COLOR,
            marker="s",
            linestyle="--",
            label="Median absolute error",
        )
        axis.axvspan(0.0, 0.2, color=CORRELATION_COLOR, alpha=0.09)
        axis.axvline(
            0.2, color=CORRELATION_COLOR, linestyle=":", linewidth=1.2
        )
        strong = minimum_transmittance < 0.2
        strong_r = float(
            np.corrcoef(generated_width[strong], fitted_width[strong])[0, 1]
        )
        strong_mae = float(
            np.mean(np.abs(fitted_width[strong] - generated_width[strong]))
        )
        mathtext_count = f"{int(strong.sum()):,}".replace(",", "{,}")
        axis.text(
            0.23,
            0.45,
            r"$T_{\min}<0.2$"
            + "\n"
            + rf"$n={mathtext_count}$, $r={strong_r:.3f}$, "
            + f"MAE={strong_mae:.2f} GHz",
            transform=axis.transAxes,
            ha="left",
            va="center",
            color=CORRELATION_COLOR,
            zorder=10,
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel(r"True minimum total transmittance, $T_{\min}$")
        axis.set_ylabel("Edge-region linewidth error (GHz)")
        axis.set_title("Absorption-dependent edge-region linewidth error")
        axis.grid(alpha=0.18, linewidth=0.5)

        right_axis = axis.twinx()
        right_axis.plot(
            x,
            correlation,
            color=CORRELATION_COLOR,
            marker="^",
            linewidth=1.2,
            label="Generated–fitted Pearson r",
        )
        right_axis.set_ylim(-0.05, 1.05)
        right_axis.set_ylabel("Pearson r", color=CORRELATION_COLOR)
        right_axis.tick_params(axis="y", colors=CORRELATION_COLOR)
        right_axis.spines["right"].set_color(CORRELATION_COLOR)

        axis.legend(
            handles=[
                Line2D([], [], color=ERROR_COLOR, marker="o", label="MAE"),
                Line2D(
                    [],
                    [],
                    color=MEDIAN_COLOR,
                    marker="s",
                    linestyle="--",
                    label="Median absolute error",
                ),
                Line2D(
                    [],
                    [],
                    color=CORRELATION_COLOR,
                    marker="^",
                    label="Generated–fitted Pearson r",
                ),
            ],
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            columnspacing=1.2,
            handlelength=2.2,
        )
        axis.text(
            -0.035,
            1.04,
            "i",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

        figure.savefig(output, facecolor="white")
        plt.close(figure)
    return output
