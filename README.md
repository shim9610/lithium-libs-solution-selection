# Lithium LIBS solution selection

This repository reproduces the nine paper figures from frozen checkpoints and
preprocessed evaluation artifacts. It also verifies artifact integrity, checks
rendered figures against the approved paper PNGs, and confirms that the
published checkpoint still reproduces stored inference results. The independent
SciPy generator, two-channel preprocessing, seven-term objective, online data
loading, Adam/cosine training loop, and component-supervision ablation are also
included under descriptive public names.

Figure 8 is tied to the authoritative **LiOH mole-fraction-corrected,
warm-start measurement evaluation**. The included reference PNG, evaluation
array, and plotting module all refer to that same result set.

## Quick start

Python 3.12, Git LFS, and ImageMagick are required. Git LFS carries the large
checkpoints and evaluation arrays. ImageMagick is used only to rasterize the
editable SVG sources for Figures 1 and 2.

```text
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Activate the virtual environment using the command appropriate for your shell,
then run the checks in this order:

```text
lithium-libs artifacts
lithium-libs figures --all
lithium-libs verify --all --existing
lithium-libs verify-inference
lithium-libs verify-training
```

The first command checks every frozen file against `artifacts/manifest.json`
before scientific results are used. Generated figures are written beneath
`outputs/figures`; machine-readable verification reports are written beneath
`outputs/reports`.

The same interface is available without the installed console script:

```text
python -m lithium_libs_solution_selection artifacts
python -m lithium_libs_solution_selection figures --all
python -m lithium_libs_solution_selection verify --all --existing
python -m lithium_libs_solution_selection verify-inference
python -m lithium_libs_solution_selection verify-training
```

To rebuild or verify one figure, replace `--all` with `--figure N`, where `N`
is a number from 1 through 9.

The wheel contains the reusable Python code, while the Git checkout contains
the large LFS-managed scientific artifacts. Run commands from the checkout
root, or point an installed command at it explicitly with
`lithium-libs --root PATH COMMAND`.

## What is included

- `artifacts/checkpoints`: the paper model and component-supervision ablation
  checkpoint.
- `artifacts/evaluation`: frozen, preprocessed synthetic and measured
  evaluation results used by the plots.
- `artifacts/training`: both complete histories plus source/checkpoint
  provenance for the full and component-supervision-ablation objectives.
- `figure_sources`: editable SVG sources for Figures 1–3.
- `reference_figures`: approved high-resolution paper PNGs.
- `src/lithium_libs_solution_selection`: the learned fitter, independent
  generator, datasets, exact loss, online trainer, figure generation,
  integrity checks, inference verification, and command-line entry point.

The raw `resources.tar` archive is intentionally excluded. It is not an input to
the public reproduction pipeline and duplicates material represented by the
included preprocessed artifacts.

## Figure reproduction map

| Figure | Public source | Reproduction path | Expected result in the pinned environment |
| --- | --- | --- | --- |
| 1 | `figure_sources/figure_01_reference_conditioned_fitting.svg` | ImageMagick rasterization | `PASS_PIXEL_EXACT` |
| 2 | `figure_sources/figure_02_training_gradient_routing.svg` | ImageMagick rasterization | `PASS_PIXEL_EXACT` |
| 3 | Approved PNG plus editable `figure_sources/figure_03_measurement_setup.svg` | Byte-preserving copy of the approved raster | `PASS_BYTE_EXACT` |
| 4 | `artifacts/evaluation/synthetic_benchmark` | Synthetic benchmark plotting module | `PASS_BYTE_EXACT` |
| 5 | `artifacts/evaluation/synthetic_multistart` | Joint Li–Ne multistart comparison module | `PASS_BYTE_EXACT` |
| 6 | `artifacts/evaluation/component_supervision` | Paired checkpoint-ablation plotting module | `PASS_BYTE_EXACT` |
| 7 | `artifacts/evaluation/measured_decomposition` | Model-conditioned measured decomposition module | `PASS_BYTE_EXACT` |
| 8 | `artifacts/evaluation/measured_standards` | Corrected warm-start isotope calibration module | `PASS_BYTE_EXACT` |
| 9 | `artifacts/evaluation/joint_refinement` | Joint Li–Ne measured-refinement module | `PASS_BYTE_EXACT` |

Figures 1 and 2 reproduce the decoded pixels exactly; only PNG container data
or compression differs. Figure 3 is a deliberately static source: the original
rasterizer and font environment cannot be established, so the package does not
claim that its editable SVG independently regenerates the approved PNG.

## Public entry points

The `lithium-libs` command is the recommended entry point. For programmatic use:

```python
from lithium_libs_solution_selection.paths import ProjectPaths
from lithium_libs_solution_selection.model import load_checkpoint

paths = ProjectPaths.discover()
model = load_checkpoint(
    paths.checkpoints / "solution_selector.pth",
    device="cpu",
)
model.eval()
```

Each module under `lithium_libs_solution_selection.figures` also exposes a
`render(...)` function. Repository-relative paths are resolved through
`ProjectPaths`; no machine-specific source location is required.

## Training reproduction

Inspect the resolved paper settings without starting a run:

```text
lithium-libs training-spec --objective full
lithium-libs training-spec --objective component-supervision-ablation
```

The exact default full run is launched with:

```text
lithium-libs train --objective full --device cuda
```

One online update generates 16,384 fresh spectra and performs 16 Adam steps;
the configured 20,000 updates therefore contain 320,000 optimizer steps. The
historical runs did not record fixed RNG states. Pass `--seed` for a repeatable
new run, without implying byte-identical regeneration of the released weights.
`--smoke-test` reduces the run to one generated training and validation sample
and is intended only for installation QA.

The full numerical contract—including the per-channel minimum subtraction,
joint input normalization, historical Gaussian-width convention, target order,
seven loss terms, reconstruction mask, and ablation definition—is documented
in [docs/training.md](docs/training.md).

## Model compression

Export a trusted checkpoint without optimizer state, or prepare a small
physics-supervised student distillation run:

```text
lithium-libs export-checkpoint --output outputs/training/teacher_inference.pth
lithium-libs distill --config configs/distillation_small.json --device cuda --dry-run
```

The default student has 2,664,722 parameters (three blocks, width 144, MLP
512/256). Dry-run validates settings without loading models or starting
training. Remove `--dry-run` on the compute machine to launch a fresh run.
See [docs/compression.md](docs/compression.md) for setup, loss definitions,
checkpoint formats, output files and evaluation requirements. Compression
accuracy and speed have not been measured.
한국어 단계별 실행 방법은 [docs/compression-ko.md](docs/compression-ko.md)에
정리했습니다. 교사 파일 내보내기, dry-run, GPU 확인 실행, 본 학습과
학생 모델 추론 예제를 포함합니다.

## Exact-first verification

Visual verification tries a byte-for-byte match first, then decoded-pixel
identity, and finally a locked tolerance check. Every accepted fallback records
both its pass class and an inferred reason. The meanings and numerical limits
are documented in [docs/verification.md](docs/verification.md).

The reference environment is fixed to:

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| NumPy | 2.4.2 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.8 |
| Matplotlib FreeType | 2.6.1 |
| Pillow | 12.3.0 |
| zlib | 1.3.2 |
| ImageMagick | 7.1.0-47 |

See [docs/architecture.md](docs/architecture.md) for module boundaries,
[docs/provenance.md](docs/provenance.md) for the figure-to-artifact chain, and
[docs/verification.md](docs/verification.md) for acceptance rules.

## License status

No public license has been selected. This repository must not be published as a
licensed release until the rights holder chooses terms for the code, model
weights, evaluation data, and figure assets. See
[PUBLICATION_RIGHTS_STATUS.md](PUBLICATION_RIGHTS_STATUS.md); it is a status
notice, not a license.
