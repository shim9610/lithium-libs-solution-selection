# Checkpoint export and student distillation

The compression workflow is separate from reproducing the released model.
It keeps the original checkpoints, evaluation artifacts and manifests intact.
The implementation and small CPU unit checks are prepared; no student has been
trained, and no compression accuracy or speedup has been measured.

## Prepare the compute environment

Use Python 3.12 and the repository's pinned scientific dependencies. Fetch
the branch and LFS objects before loading the released teacher:

```bash
git fetch origin feat/checkpoint-export-distillation
git switch feat/checkpoint-export-distillation
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
lithium-libs artifacts
```

Install the PyTorch 2.11.0 build appropriate for the target device using the
official PyTorch package index. A CPU-only wheel cannot train on CUDA.
For GPU training, verify `torch.cuda.is_available()` in that environment.
Start with a batch size that fits the target device; neither the teacher's
physical decoder nor the student uses AMP in this implementation.

## Remove optimizer state

```bash
lithium-libs export-checkpoint \
  --output outputs/training/teacher_inference.pth
```

The default source is `artifacts/checkpoints/solution_selector.pth`. Its release
SHA-256 is checked before deserialization. For another trusted checkpoint:

```bash
lithium-libs export-checkpoint \
  --checkpoint /path/to/trusted_training_checkpoint.pth \
  --expected-sha256 SOURCE_SHA256 \
  --output /path/to/new_inference_checkpoint.pth
```

The command prints source/output hashes, file sizes, parameter count and tensor
identity. It creates a new file and refuses to overwrite an existing file,
the source file, or frozen input directories. Legacy training files require
pickle loading, so only use trusted sources. The export contains CPU tensors,
`model_config`, format and schema metadata; no optimizer, scheduler, epoch or
training-history objects. It supports `torch.load(..., weights_only=True)`.
This saves disk space without changing model parameters, forward calculations,
or weight memory. It cannot resume Adam training.

Both the released architecture and newly trained students can be loaded with:

```python
from lithium_libs_solution_selection.model import load_checkpoint

model = load_checkpoint("outputs/training/teacher_inference.pth", device="cpu")
```

Self-describing files restore their architecture automatically. For an existing
custom bare state dictionary with no metadata, supply `--model-config` pointing
to an architecture JSON using the constructor keys (`base_dim`, `num_heads`,
`num_layers`, `classifier_hidden_dims`, etc.). Legacy bare release weights still
default to the published architecture. Explicit programmatic `model_config`
overrides saved settings and is subject to strict state-dictionary validation.

## Configure the student without doing any training

```bash
lithium-libs distill \
  --config configs/distillation_small.json \
  --device cuda \
  --output-dir outputs/training/student_small \
  --dry-run
```

Dry-run validates settings and prints the resolved configuration and optimizer
step count. It does not load weights, instantiate models, generate synthetic
data, create the output directory or start training. It does not establish
checkpoint integrity, device availability or available GPU memory.

The default student uses embedding width 144, three self/cross-attention blocks,
eight heads, and MLP hidden widths 512/256: **2,664,722 parameters**, versus
17,596,994 in the released model. Input `[B, 2, 512]`, all eight 300-bin output
rows, the Ne reference CNN and physical decoder retain their meanings.

The JSON merges omitted fields with `DistillationConfig` defaults; unknown
fields and invalid dimensions/hyperparameters are rejected. For a more
conservative student, change `base_dimension` to 192, keeping eight heads,
three blocks and MLP widths 512/256 (3,892,130 parameters). The dimensions of
both MLP hidden layers must be changed explicitly: shrinking embedding width
alone leaves the large published MLP in place.

The provided experiment uses batch size 32, 512 fresh samples per online update,
20,000 updates, 1,024 fixed synthetic validation spectra and seed 42. This means
16 Adam steps per update, 320,000 optimizer steps and 10,240,000 generated
training spectra. These are configurable starting values, not validated
student hyperparameters or a promise that the full run fits any given device.

## Run on the compute machine

After inspecting/editing the configuration, remove `--dry-run`:

```bash
lithium-libs distill \
  --config configs/distillation_small.json \
  --device cuda \
  --output-dir outputs/training/student_small
```

The released checkpoint is the default teacher, verified against its release
hash. To use an exported teacher, add `--teacher /path/to/teacher_inference.pth`
and preferably `--expected-sha256` with the export report's output hash.
The output directory must be new or empty. This command starts a **fresh**
student run; automatic resume is not implemented. Use a different directory
for each experiment.

Teacher parameters are frozen with `requires_grad=False`, dropout is disabled,
and the teacher stays in evaluation mode. Teacher calculations use
`torch.no_grad()`. Both models encode each input once, expose all parameter
logits, and use their physical decoder. The student retains the historical
isotope/position/observed-reconstruction detach points and Gaussian-width
convention. No measured evaluation artifacts are used during training.

The loss is:

```text
supervised_weight * original seven-term physical objective
+ logits_weight * T² * mean_over_batch_and_rows KL(teacher(T) || student(T))
+ shift_weight * MSE(student_shift / shift_max, teacher_shift / shift_max)
+ reconstruction_weight * MSE(student_observed_reconstruction, teacher_observed_reconstruction)
+ components_weight * MSE(student_components, teacher_components)
```

KL distills **all eight rows**, not only isotope logits. Its averaging is over
batch and rows after summing bins. Defaults are `T=2`, all weights 1 except
`components_weight=0`. Teacher targets are detached even when the loss helper
is used directly. Seven-term supervision must have positive weight; the
component-supervision ablation is intentionally disallowed in this workflow.
The shift term is dimensionless after division by the common +/-10 pixel
support. Reconstruction/component MSEs use the raw model outputs. Hyperparameter
weights therefore have to be tuned for this task.

Adam and the per-update cosine schedule follow the existing training code.
Distillation clips gradients from the first optimizer step, unlike the paper
runner's first-update exception. Nonfinite losses or gradients stop the run.
Validation is fixed across updates, and the best model is selected using the
**supervised physical validation loss**, rather than teacher agreement alone.

Outputs:

| File | Contents |
|---|---|
| `distillation_config.json` | Resolved experiment settings, teacher path/hash/architecture |
| `distillation_history.csv` | Train total, validation physical/KD terms, isotope MAE, timing and learning rate |
| `best_model.pth` | Student architecture/weights, optimizer/scheduler, completed steps, settings and metrics |
| `best_model_inference.pth` | Student architecture and CPU tensors, no optimizer/training state |
| `checkpoint_latest.pth` | Periodic training state; interval defaults to 10 updates |
| `checkpoint_update_N.pth` | Permanent periodic training state; defaults to 1,000 updates |

Checkpoint payloads retain training state for inspection or future custom
resume code; the CLI currently does not restore optimizer, scheduler or RNG
state. The inference checkpoint is the deployment artifact. A trained student
can subsequently use `export-checkpoint` too.

## Verify before deployment

Implementation checks, without real training or the simulator:

```bash
python -m pytest tests/test_compression.py -q
```

These use small CPU tensors/models and a stubbed runner; they check export
identity, architecture restoration, teacher gradient isolation, configuration
errors, dry-run and checkpoint selection. They do not validate a CUDA training
run. Existing frozen figure/training golden failures documented during
onboarding remain separate from these checks.

On the compute machine, first run a short experiment with JSON overrides such
as `online_updates=1`, `samples_per_online_update=32`, `batch_size=32`,
`validation_samples=32` and `number_of_workers=0`, then restore the intended
settings for longer training. Evaluate the resulting student against held-out
synthetic data and the measured spectra: isotope error/bias, shift, line widths,
component peaks and reconstruction quality. Measure file size, peak memory and
latency on the intended deployment hardware. Avoid fitting the student or
hyperparameters to the frozen measured test set.

The public `verify-inference` command specifically validates the released model
and its hash; it is not a student evaluation command. Rendering frozen figures
also cannot establish student accuracy. Neither a trained student nor the
export file replaces the published artifacts or their manifest identities.
