"""Render the component-supervision checkpoint ablation used as Figure 6."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


PARAMETERS = (
    (
        "intensity_scale",
        "true_intensity_scale",
        "pred_intensity_scale",
        1.0,
    ),
    ("absorbance", "true_absorbance", "pred_absorbance", 1.0),
    (
        "gaussian_fwhm",
        "true_gaussian_fwhm_li7_GHz",
        "pred_gaussian_fwhm_li7_GHz",
        2.0,
    ),
    (
        "core_region_fwhm",
        "true_lorentz_hot_total_fwhm_GHz",
        "pred_lorentz_hot_total_fwhm_GHz",
        1.0,
    ),
    (
        "edge_region_fwhm",
        "true_lorentz_cold_fwhm_GHz",
        "pred_lorentz_cold_fwhm_GHz",
        1.0,
    ),
    (
        "ea_offset",
        "true_absorption_offset_nm",
        "pred_absorption_offset_nm",
        1000.0,
    ),
    (
        "isotope",
        "true_isotope_percent",
        "pred_isotope_percent",
        1.0,
    ),
    ("shift", "true_shift_pixel", "pred_shift_pixel", 1.0),
)

MODEL_KEYS = ("full_objective", "without_component_supervision")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root)
    return _repository_root() / "artifacts" / "evaluation"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _finite_pair(
    data: dict[str, np.ndarray],
    true_key: str,
    predicted_key: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(data[true_key], dtype=float) * scale
    fitted = np.asarray(data[predicted_key], dtype=float) * scale
    keep = np.isfinite(generated) & np.isfinite(fitted)
    return generated[keep], fitted[keep]


def _parameter_metrics(
    data: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name, true_key, predicted_key, scale in PARAMETERS:
        generated, fitted = _finite_pair(
            data, true_key, predicted_key, scale
        )
        metrics[name] = {
            "mae": float(np.mean(np.abs(fitted - generated))),
            "pearson_r": float(pearsonr(generated, fitted).statistic),
        }
    return metrics


def render(
    output_path: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    """Render Figure 6 from the frozen paired checkpoint evaluations.

    No checkpoint is loaded and no model evaluation is performed.
    """

    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Figure 6 output must be a PNG path.")
    source_root = _data_root(data_root) / "component_supervision"
    datasets = {
        "full_objective": _load_npz(source_root / "full_objective.npz"),
        "without_component_supervision": _load_npz(
            source_root / "without_component_supervision.npz"
        ),
    }
    metrics = {
        model_key: _parameter_metrics(data)
        for model_key, data in datasets.items()
    }
    full = datasets["full_objective"]
    without_supervision = datasets["without_component_supervision"]

    figure = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 0.92))
    paired_axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
    )
    trend_axis = figure.add_subplot(grid[1, 0:3])
    ratio_axis = figure.add_subplot(grid[1, 3:6])

    comparisons = (
        (
            paired_axes[0],
            "recon_s_rmse",
            "Total-spectrum RMSE",
            "Mean ratio",
            "Spectral RMSE",
        ),
        (
            paired_axes[1],
            "emission_component_relative_l1",
            "Emission-component error",
            r"Relative $L_1$ mean ratio",
            r"Relative $L_1$ error",
        ),
        (
            paired_axes[2],
            "absorption_transmittance_component_relative_l1",
            "Absorption-component error",
            r"Transmittance relative $L_1$ mean ratio",
            r"Transmittance relative $L_1$ error",
        ),
    )
    violin_colors = ("#2166ac", "#b2182b")
    for axis, key, title, subtitle, ylabel in comparisons:
        full_values = np.asarray(full[key], dtype=float)
        without_values = np.asarray(without_supervision[key], dtype=float)
        if full_values.shape != without_values.shape:
            raise ValueError(
                f"Paired arrays for {key} have different shapes: "
                f"{full_values.shape} and {without_values.shape}."
            )
        keep = np.isfinite(full_values) & np.isfinite(without_values)
        full_values = full_values[keep]
        without_values = without_values[keep]
        values = (full_values, without_values)
        violins = axis.violinplot(
            values,
            positions=(0, 1),
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method="scott",
        )
        for body, color in zip(violins["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(0.9)
            body.set_alpha(0.55)
        for position, sample, color in zip((0, 1), values, violin_colors):
            percentile_05, q1, median, q3, percentile_95 = np.percentile(
                sample, (5, 25, 50, 75, 95)
            )
            axis.plot(
                (position, position),
                (percentile_05, percentile_95),
                color=color,
                linewidth=1.2,
                zorder=3,
            )
            axis.plot(
                (position, position),
                (q1, q3),
                color=color,
                linewidth=5.0,
                solid_capstyle="round",
                zorder=4,
            )
            axis.scatter(
                position,
                median,
                s=24,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=5,
            )
        lower = float(min(np.min(full_values), np.min(without_values)))
        upper = float(max(np.max(full_values), np.max(without_values)))
        span = max(upper - lower, 1e-8)
        axis.set_ylim(max(0.0, lower - 0.04 * span), upper + 0.04 * span)
        axis.set_xlim(-0.62, 1.62)
        axis.set_xticks(
            (0, 1),
            ("Full-objective\ncheckpoint", "Ablation\ncheckpoint"),
        )
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"{title}\n{subtitle}="
            f"{np.mean(without_values) / np.mean(full_values):.3f}",
            fontsize=8.2,
        )
        axis.grid(axis="y", alpha=0.17)
        axis.tick_params(labelsize=7.2)
        axis.yaxis.label.set_size(7.5)

    names = [parameter[0] for parameter in PARAMETERS]
    labels = [
        "Intensity",
        "Peak optical\ndepth",
        "Gaussian",
        "Core Lorentzian",
        "Edge Lorentzian",
        "E-A offset",
        r"$^6$Li",
        "Shift",
    ]
    positions = np.arange(len(names))
    width = 0.38

    full_correlation = [
        metrics["full_objective"][name]["pearson_r"] for name in names
    ]
    without_correlation = [
        metrics["without_component_supervision"][name]["pearson_r"]
        for name in names
    ]
    full_bars = trend_axis.bar(
        positions - width / 2,
        full_correlation,
        width,
        color="#2166ac",
        label="Full-objective checkpoint",
    )
    without_bars = trend_axis.bar(
        positions + width / 2,
        without_correlation,
        width,
        color="#b2182b",
        label="Ablation checkpoint",
    )
    trend_axis.set_ylim(0, 1.06)
    trend_axis.set_ylabel("Generated-fitted Pearson $r$")
    trend_axis.set_xticks(positions, labels, rotation=34, ha="right")
    trend_axis.set_title("Generated-to-fitted trend recovery", fontsize=8.8)
    trend_axis.tick_params(labelsize=7.2)
    trend_axis.yaxis.label.set_size(7.5)
    trend_axis.grid(axis="y", alpha=0.2)

    mae_ratio = [
        metrics["without_component_supervision"][name]["mae"]
        / max(metrics["full_objective"][name]["mae"], 1e-12)
        for name in names
    ]
    ratio_axis.bar(positions, mae_ratio, color="#595959")
    ratio_axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    ratio_axis.set_ylabel("Ablation/full-objective MAE ratio")
    ratio_axis.set_xticks(positions, labels, rotation=34, ha="right")
    ratio_axis.set_title(
        "Parameter MAE ratio (above 1 favors full-objective checkpoint)",
        fontsize=8.8,
    )
    ratio_axis.tick_params(labelsize=7.2)
    ratio_axis.yaxis.label.set_size(7.5)
    ratio_axis.grid(axis="y", alpha=0.2)

    all_axes = (*paired_axes, trend_axis, ratio_axis)
    for label, axis in zip(("a", "b", "c", "d", "e"), all_axes):
        axis.text(
            -0.18,
            1.13,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
    figure.legend(
        handles=(full_bars, without_bars),
        labels=("Full-objective checkpoint", "Ablation checkpoint"),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output
