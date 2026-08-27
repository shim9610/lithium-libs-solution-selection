# Training and objective reproduction

The public training path is a descriptive-name transcription of the frozen
source that produced the released learned-fitter checkpoint. It preserves the
historical numerical operations while removing development-version switches
and machine-specific paths.

The authoritative historical source revision is
`03f1193ab5e4e2981a3aadf4d70c44215eac0ae4`. Role-specific source hashes,
checkpoint hashes, selected online updates, and both complete histories are
recorded in `artifacts/training`.

## Reproduction scope

The package reproduces the generator distribution, preprocessing, architecture,
loss calculation, Adam update sequence, cosine schedule, validation procedure,
and checkpoint-selection rule. The historical run did not record Python,
NumPy, PyTorch, or worker RNG states. Consequently, a newly seeded run is
repeatable but cannot be claimed to recreate the published checkpoint bytes.

Run the frozen-source numerical checks with:

```text
lithium-libs verify-training
```

Print the full configuration and resolved loss weights with:

```text
lithium-libs training-spec --objective full
lithium-libs training-spec --objective component-supervision-ablation
```

## Input and preprocessing

Each generated input has shape `[2, 512]`:

1. the lithium measurement window on CCD coordinates 577--1088;
2. the simultaneous neon wavelength-reference window on coordinates 1088--1600.

The same sampled frame shift is applied to both coordinate grids. The generator
subtracts the minimum of each channel separately. The learned fitter then
performs one sample-wise min--max normalization over all 1,024 input values,
thereby retaining the Li/Ne relative amplitude after the two baseline offsets
have been removed.

## Fitting rows and global decoder coordinates

The prediction head produces eight 300-bin rows. The first seven enter the
physical decoder; the eighth is the spectral shift.

| Row | Internal coordinate | Bin support |
| ---: | --- | --- |
| 0 | normalized emission-component scale | `[1, 5]` |
| 1 | peak optical depth | `[0, 2]` |
| 2 | shared Gaussian standard deviation in decoder coordinates | `[1e-8, 0.05]` |
| 3 | nonnegative core--edge Lorentzian HWHM increment | `[1e-8, 0.5001]` |
| 4 | edge-region absorption Lorentzian HWHM | `[1e-8, 0.1001]` |
| 5 | emission--absorption offset | `[-0.02, 0.02]` |
| 6 | lithium-6 percentage | `[0, 100]` |
| 7 | spectral shift | `[-10, 10]` pixel |

The core-region emission HWHM is row 3 plus row 4. The checkpoint also contains
two model-wide trainable scalars: the mapped reference centre and transition
spacing. Both remain part of the public architecture and strict checkpoint
contract.

The lithium-6 expectation used to set transition amplitudes is detached from
the component-rendering graph. Absorption centres are the detached emission
centres plus the learned emission--absorption offset. The canonical
reconstruction is also detached before the circular first-order interpolation
that produces the input-frame reconstruction; therefore the latter loss routes
its direct gradient through the shift row.

## Online generator support

For each training instance the generator samples:

| Coordinate | Distribution |
| --- | --- |
| lithium-6 percentage | `Uniform(0, 100)` |
| peak optical depth | `Uniform(0, 2)` |
| normalized raw LIBS intensity | fixed at `10,000` before model-input normalization |
| historical Gaussian width-scale variable | `Uniform(300, 20,000)` K |
| core--edge Lorentzian linewidth increment | `Uniform(0.1, 120)` GHz |
| edge-region absorption Lorentzian linewidth | `Uniform(0.1, 20)` GHz |
| emission--absorption offset | `Uniform(-0.007, 0.007)` nm |
| frame shift | `Uniform(-10, 10)` pixel |
| Gaussian detector-noise standard deviation | `Uniform(0, 180)` |
| scaled-Poisson factor | `Uniform(0, 15)` |

The wavelength map is

```text
lambda(p) = -2.8581e-8 p^2 + 0.00167 p + 669.43069 nm.
```

The neon line is centred at 671.70430 nm, with a nominal 0.008846 nm FWHM
varied by plus or minus 5%. Li and Ne share the two sampled noise-level scalars,
but their noise realizations are independent.

The original SciPy generator accidentally omitted the leading factor 2 when it
converted a Doppler FWHM to the Gaussian standard deviation required by
`voigt_profile`: it used `FWHM / sqrt(2 log(2))` instead of the conventional
`FWHM / (2 sqrt(2 log(2)))`. Its supplied Gaussian standard deviation was
therefore twice the conventional value, and its rendered Gaussian FWHM was
twice the nominal Doppler FWHM. The released checkpoint was trained on these
rendered profiles, so the historical calculation is deliberately retained to
preserve checkpoint and training-distribution integrity.

This convention does not affect the paper's stated linewidth task because the
generated and fitted quantities are compared as effective rendered Gaussian
FWHM in GHz, without inferring temperature. The sampled 300--20,000 values are
only the K-valued internal driver of the historical width formula and must not
be reported as a validated plasma-temperature range. A study that interprets
them physically or converts fitted widths back to kelvin must correct the
conversion, regenerate the training distribution, retrain the model, and repeat
the evaluations.

An unused reference-lithium spectrum and its detector noise are also evaluated
before the neon noise draw. The result is not added to the input, but the draw
order is retained so a fixed Python/NumPy seed reproduces the frozen generator
array byte for byte.

## Target tensor

The generator returns ten target rows:

1. rows 0--3: canonical emission components;
2. rows 4--7: canonical absorption optical-depth components;
3. row 8: input-frame reconstruction;
4. row 9: canonical reconstruction.

The lithium-6 target is a 300-bin truncated Gaussian with nominal standard
deviation 1 percentage point and zero support outside plus or minus 20
percentage points. Its underlying continuous mean is solved so truncation to
0--100 has the requested expectation, followed by discrete normalization.

## Full objective

The active full objective is the sum of seven unit-weight terms:

```text
component MSE
+ component prediction-relative L1
+ component peak-height relative L1
+ component soft-position error
+ canonical reconstruction error
+ input-frame reconstruction error
+ KL(target || prediction)
```

The central crop is Python slice `50:-50`, indices 50--461 inclusive (412
samples). Component MSE and relative L1 use only the target-dominant D2 pair:
rows 0 and 4 below 50% lithium-6, rows 1 and 5 at or above 50%.

The component relative term is intentionally

```text
mean(abs((target - prediction) / (prediction + 1e-8))).
```

It has no 0.1 threshold. Peak height uses the full 512-point maximum and a
ground-truth denominator floored at `1e-4`. The position term uses a
temperature-1 soft argmax over component rows 0--6 and divides pixel distance
by 512; the historical implementation excludes row 7.

Both reconstruction terms use target values at least 0.1 and compute a global
mean over all selected batch pixels:

```text
mean((prediction - target)^2) + mean(abs(prediction - target) / target).
```

The prediction is cropped and then min--max normalized inside the crop. The
target was min--max normalized over all 512 generator samples before cropping
and is not normalized again.

## Component-supervision ablation

The component-supervision ablation retains only:

```text
canonical reconstruction + input-frame reconstruction + isotope KL.
```

The four removed terms are component MSE, component relative L1, peak height,
and component position. The public implementation gates these terms
individually; setting a shared reconstruction scale to zero would incorrectly
remove the canonical reconstruction term as well.

The ablation checkpoint and its 20,000-row history are preserved. A separate
historical ablation launcher was not preserved, so the public ablation path is
the explicit paper-defined objective reconstruction rather than a claim that a
lost launcher file was copied byte for byte.

## Optimizer and online updates

- fresh initialization;
- dropout forced to 0;
- Adam, learning rate `6e-4`, default betas `(0.9, 0.999)`, epsilon `1e-8`,
  no weight decay, FP32, no AMP;
- batch size 1,024;
- 16,384 fresh spectra per online update, hence 16 Adam steps per update;
- 20,000 configured online updates, hence 320,000 configured Adam steps;
- cosine annealing once per online update, `T_max=20,000`, `eta_min=1e-7`, no
  warmup;
- global gradient-norm cap 1,000 after the first online update; the first 16
  Adam steps are not clipped;
- 4,096 validation spectra generated once at process start;
- best checkpoint selected by minimum validation loss.

The released learned-fitter checkpoint was selected after 18,737 completed
online updates (299,792 Adam steps). The component-supervision ablation was
selected after 18,660 completed online updates (298,560 Adam steps).

To launch an exact-default new run:

```text
lithium-libs train --objective full --device cuda
```

Pass `--seed` to make the new run repeatable. Use `--smoke-test` only to verify
installation and gradient/update plumbing; it deliberately reduces the sample
and update counts.
