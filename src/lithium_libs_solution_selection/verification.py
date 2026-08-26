"""Exact-first visual regression checks with an explicit tolerance fallback."""

from __future__ import annotations

import json
import platform
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import PIL
from matplotlib import ft2font
from PIL import Image
from scipy.ndimage import gaussian_filter

from .artifacts import sha256_file


MAX_NORMALIZED_RMSE = 0.010
MAX_NORMALIZED_MAE = 0.0025
MAX_CHANGED_PIXEL_FRACTION = 0.01
MIN_SSIM = 0.995


@dataclass(frozen=True)
class ImageMetrics:
    dimensions_equal: bool
    pixel_equal: bool
    normalized_mae: float | None
    normalized_rmse: float | None
    changed_fraction_gt_8: float | None
    ssim: float | None


@dataclass(frozen=True)
class FigureCheck:
    figure: str
    status: str
    semantic_status: str
    reference_sha256: str
    generated_sha256: str
    metrics: ImageMetrics
    reason_code: str
    inferred_reason: str


def _rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _ssim(reference: np.ndarray, generated: np.ndarray) -> float:
    reference_gray = (
        0.2126 * reference[..., 0]
        + 0.7152 * reference[..., 1]
        + 0.0722 * reference[..., 2]
    ) / 255.0
    generated_gray = (
        0.2126 * generated[..., 0]
        + 0.7152 * generated[..., 1]
        + 0.0722 * generated[..., 2]
    ) / 255.0
    sigma = 1.5
    mu_x = gaussian_filter(reference_gray, sigma=sigma, mode="reflect")
    mu_y = gaussian_filter(generated_gray, sigma=sigma, mode="reflect")
    sigma_x = gaussian_filter(reference_gray * reference_gray, sigma, mode="reflect") - mu_x * mu_x
    sigma_y = gaussian_filter(generated_gray * generated_gray, sigma, mode="reflect") - mu_y * mu_y
    sigma_xy = gaussian_filter(reference_gray * generated_gray, sigma, mode="reflect") - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, np.finfo(float).eps)))


def _png_chunks(path: Path) -> list[str]:
    chunks: list[str] = []
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            return chunks
        while True:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                break
            length = struct.unpack(">I", raw_length)[0]
            name = stream.read(4).decode("ascii", errors="replace")
            chunks.append(name)
            stream.seek(length + 4, 1)
            if name == "IEND":
                break
    return chunks


def compare_images(reference: Path, generated: Path) -> ImageMetrics:
    ref = _rgba(reference)
    got = _rgba(generated)
    if ref.shape != got.shape:
        return ImageMetrics(False, False, None, None, None, None)
    equal = bool(np.array_equal(ref, got))
    difference = np.abs(ref.astype(np.int16) - got.astype(np.int16))
    return ImageMetrics(
        dimensions_equal=True,
        pixel_equal=equal,
        normalized_mae=float(np.mean(difference) / 255.0),
        normalized_rmse=float(
            np.sqrt(np.mean(difference.astype(float) ** 2)) / 255.0
        ),
        changed_fraction_gt_8=float(
            np.mean(np.any(difference > 8, axis=-1))
        ),
        ssim=_ssim(ref, got),
    )


def verify_figure(
    reference: Path,
    generated: Path,
    *,
    semantic_ok: bool,
    static_raster_source: bool = False,
) -> FigureCheck:
    reference_hash = sha256_file(reference)
    generated_hash = sha256_file(generated)
    metrics = compare_images(reference, generated)
    semantic_status = "PASS" if semantic_ok else "FAIL"

    if not semantic_ok:
        status = "FAIL_SEMANTIC"
        reason_code = "INPUT_OR_PLOT_DATA_MISMATCH"
        reason = "The frozen scientific inputs or plotted values did not match their manifest."
    elif reference_hash == generated_hash:
        status = "PASS_BYTE_EXACT"
        reason_code = "STATIC_RASTER_SOURCE" if static_raster_source else "IDENTICAL_PINNED_STACK"
        reason = (
            "The approved raster artwork was copied without transformation."
            if static_raster_source
            else "The pinned inputs and rendering stack reproduced the PNG byte-for-byte."
        )
    elif metrics.pixel_equal:
        status = "PASS_PIXEL_EXACT"
        reason_code = "PNG_ANCILLARY_OR_COMPRESSION_ONLY"
        reference_chunks = _png_chunks(reference)
        generated_chunks = _png_chunks(generated)
        reason = (
            "Decoded pixels are identical; the PNG container differs in ancillary metadata "
            f"or compression (reference chunks={reference_chunks}, generated chunks={generated_chunks})."
        )
    elif (
        metrics.dimensions_equal
        and metrics.normalized_rmse is not None
        and metrics.normalized_rmse <= MAX_NORMALIZED_RMSE
        and metrics.normalized_mae is not None
        and metrics.normalized_mae <= MAX_NORMALIZED_MAE
        and metrics.changed_fraction_gt_8 is not None
        and metrics.changed_fraction_gt_8 <= MAX_CHANGED_PIXEL_FRACTION
        and metrics.ssim is not None
        and metrics.ssim >= MIN_SSIM
    ):
        status = "PASS_TOLERANCE"
        reason_code = "RENDERER_ENVIRONMENT_DIFFERENCE"
        reason = (
            "Scientific inputs match and the same-size image stays within the locked visual "
            "thresholds; the remaining edge-level differences are consistent with font or "
            "renderer-version variation."
        )
    else:
        status = "FAIL_VISUAL_UNKNOWN"
        reason_code = "UNEXPLAINED_RENDER_DIFFERENCE"
        reason = "The image difference is too large or insufficiently explained for fallback acceptance."

    return FigureCheck(
        figure=reference.name,
        status=status,
        semantic_status=semantic_status,
        reference_sha256=reference_hash,
        generated_sha256=generated_hash,
        metrics=metrics,
        reason_code=reason_code,
        inferred_reason=reason,
    )


def environment_record() -> dict[str, str]:
    imagemagick = "unavailable"
    executable = shutil.which("magick")
    if executable is not None:
        try:
            completed = subprocess.run(
                [executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = (completed.stdout or completed.stderr).splitlines()
            if first_line:
                imagemagick = first_line[0].removeprefix("Version: ").strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "matplotlib_freetype": ft2font.__freetype_version__,
        "pillow": PIL.__version__,
        "zlib": zlib.ZLIB_RUNTIME_VERSION,
        "imagemagick": imagemagick,
    }


def write_report(path: Path, checks: list[FigureCheck]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "environment": environment_record(),
        "thresholds": {
            "normalized_rmse_max": MAX_NORMALIZED_RMSE,
            "normalized_mae_max": MAX_NORMALIZED_MAE,
            "changed_fraction_gt_8_max": MAX_CHANGED_PIXEL_FRACTION,
            "ssim_min": MIN_SSIM,
        },
        "figures": [
            {
                **{key: value for key, value in asdict(check).items() if key != "metrics"},
                "image_metrics": asdict(check.metrics),
            }
            for check in checks
        ],
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
