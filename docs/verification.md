# Verification protocol

The verification process separates file integrity, scientific semantics,
rendering identity, and checkpoint inference. A visually similar image is not
accepted until the frozen inputs and plotted values pass their own checks.

## Recommended sequence

```text
lithium-libs artifacts
lithium-libs figures --all
lithium-libs verify --all --existing
lithium-libs verify-inference
```

1. **Artifact integrity** checks the size and SHA-256 of every frozen input
   against `artifacts/manifest.json`.
2. **Figure generation** writes fresh PNGs beneath `outputs/figures` without
   modifying approved references.
3. **Figure verification** confirms semantic inputs and then applies the
   exact-first image comparison below.
4. **Inference verification** strictly loads the published model and compares
   predictions on a deterministic subset with values stored in the measured
   evaluation artifact.

Each command exits unsuccessfully if a required check fails. Reports include
the comparison class, image metrics, input state, and an inferred reason for
any accepted non-byte-exact result.

## Exact-first decision sequence

The generated and reference images are compared without resizing, alignment,
cropping, or color correction.

| Status | Acceptance rule | Interpretation |
| --- | --- | --- |
| `PASS_BYTE_EXACT` | SHA-256 values are identical. | The complete PNG files are identical. For Figure 3, this is explicitly a byte-preserving approved-raster copy. |
| `PASS_PIXEL_EXACT` | File bytes differ, but decoded RGBA arrays are identical. | Only PNG container metadata or compression differs; scientific and visible content is unchanged. |
| `PASS_TOLERANCE` | Dimensions match and every fixed metric limit below passes. | Small renderer- or font-environment differences are accepted and recorded, not hidden. |

Semantic failure takes precedence over all three pass classes. A file also
fails if its dimensions differ or if any tolerance metric is outside its fixed
limit.

## Locked tolerance limits

Pixel differences are calculated on decoded RGBA channels in the common,
unchanged image dimensions.

| Metric | Required value |
| --- | --- |
| Normalized mean absolute error | at most `0.0025` |
| Normalized root-mean-square error | at most `0.010` |
| Fraction of pixels with any RGBA-channel absolute difference greater than 8 | at most `0.01` |
| Structural similarity index | at least `0.995` |

All four conditions must pass for `PASS_TOLERANCE`. There is no best-effort
alignment step and no automatic threshold relaxation.

## Reason reporting

The report distinguishes the result from its explanation:

| Reason category | Meaning |
| --- | --- |
| Identical pinned stack | Pinned data and rendering software produced the same PNG bytes. |
| Static raster source | The approved raster was copied without transformation because an independent rasterization path is not supportable. |
| PNG ancillary or compression only | Decoded pixels are equal, while PNG chunks or compression differ. |
| Renderer environment difference | Scientific inputs match and differences satisfy every fixed visual limit; font or renderer variation is the inferred cause. |
| Input or plot-data mismatch | Frozen scientific inputs or plotted values do not match their manifest or semantic checks. |
| Unexplained render difference | The mismatch is too large or insufficiently explained to accept. |

The reason attached to a fallback is explicitly an inference. It never upgrades
a failed metric into a pass.

## Expected results in the reference environment

| Figures | Expected class | Basis |
| --- | --- | --- |
| 1–2 | `PASS_PIXEL_EXACT` | SVG rasterization reproduces identical decoded pixels; PNG bytes differ. |
| 3 | `PASS_BYTE_EXACT` | Approved static raster is copied byte-for-byte. |
| 4–9 | `PASS_BYTE_EXACT` | Frozen preprocessed inputs and pinned plotting stack reproduce the approved files exactly. |

The reference stack is Python 3.12.13, PyTorch 2.11.0+cu128, NumPy 2.4.2,
SciPy 1.17.1, Matplotlib 3.10.8 with FreeType 2.6.1, Pillow 12.3.0,
zlib 1.3.2, and ImageMagick 7.1.0-47.

## Inference verification

`lithium-libs verify-inference` performs a strict checkpoint load and evaluates
a fixed set of 17 evenly distributed rows from the stored, calibrated measured
spectra. It compares the recomputed isotope prediction and fitted shift with
the frozen values. The locked absolute limits
are `0.01` percentage point for the isotope prediction and `0.001` pixel for
the fitted shift. The command records the device, limits, maximum observed
differences, artifact hashes, and pass/fail result in its report.

The checkpoint SHA-256 is checked before PyTorch deserializes it. A changed
checkpoint is rejected rather than loaded.

This check is separate from figure generation: a byte-exact plot verifies the
plotting path, while inference verification confirms that the supplied model
weights and public model definition still interpret the same inputs
consistently.
