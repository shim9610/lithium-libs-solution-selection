# Architecture

The package separates immutable scientific inputs, pure figure rendering,
model inference, and verification. This keeps paper reproduction independent
of private experiment directories and makes every public execution path start
from repository-relative inputs.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| Command line | `lithium_libs_solution_selection.cli` | Exposes artifact, figure, inference, training-calculation, and online-training entry points. |
| Path resolution | `lithium_libs_solution_selection.paths` | Discovers the repository root and provides stable paths for artifacts, references, sources, and outputs. |
| Artifact integrity | `lithium_libs_solution_selection.artifacts` | Verifies file size and SHA-256 against the frozen manifest. |
| Model | `lithium_libs_solution_selection.model` | Defines the reference-conditioned spectrum model, strictly loads checkpoints, and exposes continuous decoding. |
| Forward generator | `lithium_libs_solution_selection.physics`, `simulation` | Reproduces the independent SciPy generator, two-channel Li--Ne inputs, and hybrid canonical/input-frame targets. |
| Training data | `lithium_libs_solution_selection.datasets` | Provides online synthetic batches and the process-fixed validation set. |
| Objective | `lithium_libs_solution_selection.losses` | Implements the seven-term full objective and component-supervision ablation. |
| Training | `lithium_libs_solution_selection.training` | Runs the 20,000 online-update Adam/cosine protocol and checkpoint selection. |
| Training verification | `lithium_libs_solution_selection.training_verification` | Locks the migrated simulator and loss to frozen-source golden calculations. |
| Figure registry | `lithium_libs_solution_selection.figures` | Maps figure numbers to public renderer modules and their required inputs. |
| Visual verification | `lithium_libs_solution_selection.verification` | Applies semantic checks and the byte-exact, pixel-exact, then tolerance decision sequence. |
| Inference verification | `lithium_libs_solution_selection.inference_verification` | Re-runs a deterministic evaluation subset and compares model outputs with frozen results. |

## Data flow

1. `lithium-libs artifacts` validates all immutable inputs against
   `artifacts/manifest.json`.
2. `lithium-libs figures` resolves a figure through the public registry and
   writes a PNG beneath `outputs/figures`.
3. The renderer reads only the artifacts declared for that figure. Figure 5
   uses its frozen representative reconstruction; checkpoint interpretation
   is independently covered by the inference verifier.
4. `lithium-libs verify` first confirms the scientific input/plot-data
   semantics, then compares the generated PNG with the approved raster.
5. `lithium-libs verify-inference` strictly loads the published checkpoint and
   compares deterministic predictions with the stored measurement outputs.
6. `lithium-libs verify-training` compares a seeded generated sample and every
   loss term with golden results calculated from the frozen historical source.
7. `lithium-libs train` creates fresh spectra online and applies the published
   optimizer, scheduler, validation, and objective-selection protocol.

No renderer discovers or imports an experiment directory. The public package
does not add a runtime dependency between a user interface, a separate source
of truth, and the interpretation engine.

The public training modules likewise contain no machine-specific experiment
path or development-version selector. Their numerical contract and historical
limitations are documented in [training.md](training.md).

## Figure modules

| Figures | Module | Input contract |
| --- | --- | --- |
| 1–3 | `figures.diagrams` | SVG sources for Figures 1–2; approved raster copy for Figure 3. |
| 4 | `figures.figure04_synthetic_benchmark` | Frozen synthetic benchmark table. |
| 5 | `figures.figure05_joint_multistart` | Multistart arrays, selected baseline/joint runs, summaries, and representative reconstruction. |
| 6 | `figures.figure06_checkpoint_ablation` | Paired precomputed evaluations for the full and ablated objectives. |
| 7 | `figures.figure07_decomposition` | Measured component overlays, reconstruction overlaps, and corrected isotope labels. |
| 8 | `figures.figure08_isotope_calibration` | Corrected warm-start measured-standard output array. |
| 9 | `figures.figure09_joint_refinement` | Per-spectrum joint Li–Ne refinement results. |

Every plotting module exposes `render(...)`. The command-line registry supplies
the standard artifact and output paths, while direct callers may pass explicit
repository-relative roots.

## Immutable and generated content

The following directories are treated as immutable inputs:

- `artifacts/checkpoints`
- `artifacts/evaluation`
- `figure_sources`
- `reference_figures`

Generated files belong beneath `outputs`. Verification never overwrites an
approved reference image. The artifact and figure manifests are the
machine-readable contracts for file identity and figure dependencies.

The built wheel contains code only. Checkpoints, evaluation records, diagram
sources, and approved references remain Git LFS-managed repository assets. An
installed command can locate them from the checkout working directory or from
the explicit global `--root` option.
