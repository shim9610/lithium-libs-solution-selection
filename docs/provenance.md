# Artifact and figure provenance

This package freezes only the inputs required to reproduce and verify the
published results: model checkpoints, preprocessed evaluation records, editable
diagram sources, and approved PNGs. File sizes and SHA-256 digests in
`artifacts/manifest.json` are the authoritative identity record.

## Checkpoints

| Artifact | Role |
| --- | --- |
| `artifacts/checkpoints/solution_selector.pth` | Published reference-conditioned solution-selection model used by the inference harness and retained with the evaluations derived from it. |
| `artifacts/checkpoints/component_supervision_ablation.pth` | Checkpoint trained without component supervision, retained for audit and strict-loading verification. |

Figure 6 is rendered from the paired frozen evaluation arrays, rather than
rerunning the full evaluation sweep. This makes plot reproduction deterministic
while the included checkpoints preserve the model states being compared.

## Training provenance

`artifacts/training/provenance.json` records the authoritative historical
source revision, role-level source hashes, checkpoint-selection updates, and
the absence of a recorded random seed. The two complete 20,000-row histories
are retained as `full_objective_history.csv` and
`component_supervision_ablation_history.csv`.

The public modules reproduce the full training calculation without exposing
private filenames or development-version labels. A seeded generator golden is
byte-identical to the frozen source, and the public loss produces the same total
and term values on a fixed tensor fixture. These checks are run by
`lithium-libs verify-training`.

The full-objective source is present in the authoritative revision. The
ablation checkpoint and history are present, but its separate historical
launcher was not preserved. The public ablation therefore implements the
paper-specified objective—both reconstruction terms plus isotope KL—without
claiming that an unavailable launcher was recovered.

## Evaluation artifacts

| Directory | Contents and use |
| --- | --- |
| `artifacts/evaluation/synthetic_benchmark` | Random synthetic spectra table, generation seeds, and sampling metadata for Figure 4. |
| `artifacts/evaluation/synthetic_multistart` | Reconstruction arrays, baseline runs, per-spectrum summaries, joint runs, and a cached representative joint reconstruction for Figure 5. |
| `artifacts/evaluation/component_supervision` | Matched evaluation arrays for the complete training objective and the objective without component supervision, used by Figure 6. |
| `artifacts/evaluation/measured_decomposition` | Session A/B reconstruction overlaps, component overlays, and corrected isotope labels used by Figure 7. |
| `artifacts/evaluation/measured_standards` | Full measured-output array, calibration record, per-file and summary tables, and corrected isotope labels used by Figure 8 and inference verification. |
| `artifacts/evaluation/joint_refinement` | Per-spectrum and summary outputs from the joint Li–Ne refinement used by Figure 9. |

The measured-standard artifact used by Figure 8 is the authoritative result of
the LiOH mole-fraction correction and warm-start evaluation. It is kept paired
with its approved reference PNG so the plot and the stated analysis cannot
silently refer to different result sets.

## Figure chain

| Figure | Scientific or design input | Generator | Approved reproduction status |
| --- | --- | --- | --- |
| 1 | Reference-conditioned fitting SVG | ImageMagick SVG rasterization | Pixel-exact |
| 2 | Training and gradient-routing SVG | ImageMagick SVG rasterization | Pixel-exact |
| 3 | Approved measurement-setup PNG; SVG retained for editing | Byte-preserving file copy | Byte-exact static source |
| 4 | Synthetic benchmark table | `figure04_synthetic_benchmark.render` | Byte-exact |
| 5 | Synthetic multistart evaluation | `figure05_joint_multistart.render` | Byte-exact |
| 6 | Paired component-supervision evaluations | `figure06_checkpoint_ablation.render` | Byte-exact |
| 7 | Corrected measured decomposition tables | `figure07_decomposition.render` | Byte-exact |
| 8 | Corrected warm-start measured-standard array | `figure08_isotope_calibration.render` | Byte-exact |
| 9 | Measured joint Li–Ne per-spectrum results | `figure09_joint_refinement.render` | Byte-exact |

Figures 1 and 2 have identical decoded pixels to the approved PNGs. Their file
bytes differ because of PNG-container metadata or compression; this explanation
is an inference supported by decoded-pixel equality. Figures 4–9 reproduce
byte-for-byte in the pinned environment.

The editable SVG for Figure 3 is useful for inspection and future editing, but
the rasterizer command, fonts, and exact environment used for the approved
raster could not be recovered. Re-rasterizing that SVG with a guessed tool
would imply unsupported provenance. The public path therefore copies the
approved raster exactly and labels the result as a static source.

## Deliberate exclusions

The raw `resource.tar` archive is not included. It is not consumed by any public
figure or verification entry point and duplicates source material already
represented by the preprocessed evaluation artifacts. Its omission does not
remove an input from the documented reproduction chain.

No private absolute path, experiment folder name, or internal development label
is part of the public provenance contract.
