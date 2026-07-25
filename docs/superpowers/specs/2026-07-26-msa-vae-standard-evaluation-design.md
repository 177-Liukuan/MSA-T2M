# MSA-VAE Standard Evaluation Pipeline Design

## Objective

Add an isolated, one-command evaluation pipeline for HumanML3D-272 MSA-VAE
checkpoints. Given a checkpoint path, the pipeline must reconstruct one
deterministic test set and report:

- FID (lower is better)
- MPJPE in millimetres (lower is better)
- P-MPJPE in millimetres (lower is better)
- ACCEL in millimetres per frame squared (lower is better)
- foot skating as a percentage (lower is better)
- text-to-motion R@1, R@2, R@3 and median rank
- motion-to-text R@1, R@2, R@3 and median rank

The new pipeline must not change the code path, command-line defaults, metric
definitions, or output of the existing MotionStreamer, MSA-T2M, TAE, BABEL, or
legacy MSA-VAE evaluators.

## Considered Approaches

### 1. Extend `eval_msa_vae.py`

This would reuse the most code, but that file is also the current legacy
HumanML/BABEL entrypoint. Changing its loader and retrieval behaviour would
silently invalidate historical results and risks affecting training-time
evaluation.

### 2. Add an isolated MSA-VAE metrics pipeline

This is the selected approach. A new dataset, metric module, Python entrypoint,
and uppercase shell launcher will reuse the frozen 272-D evaluator and
MSA-VAE model definitions while leaving all old entrypoints untouched.

### 3. Run the cloned MLD, OmniControl, and TMR repositories directly

This appears closest to the papers, but their data representations,
checkpoints, frame rates, and dependency stacks do not match the repository's
272-D evaluator. Direct composition would mix incompatible embedding spaces
and make the result harder to reproduce. Their metric definitions will be
ported or adapted into focused, tested local functions instead; the cloned
third-party trees remain unmodified and untracked.

## User Interface

The primary command is:

```bash
bash EVAL_msa_vae_metrics.sh /path/to/checkpoint.pth
```

Additional arguments may follow the checkpoint path, for example:

```bash
bash EVAL_msa_vae_metrics.sh /path/to/checkpoint.pth \
  --batch-size 32 \
  --output-dir output/msa_vae_metrics/my_ablation
```

The launcher resolves paths relative to the repository root, invokes the
existing `mgpt` environment when it is not already active, and does not create
or replace the legacy `Evaluator_272` symlinks.

The Python entrypoint accepts a required checkpoint path and optional dataset,
evaluator, output, device, worker, seed, and model-architecture overrides.
Default paths are repository-relative rather than machine-specific.

## Checkpoint Configuration Contract

The pipeline loads model weights strictly. It resolves architecture values in
this order:

1. explicit command-line overrides;
2. checkpoint `metadata.training_args` when present;
3. the first valid JSON argument block in `run.log` beside a legacy
   checkpoint;
4. state-dict shape inference where the value is structurally identifiable;
5. the documented mainline MSA-VAE defaults for values such as attention-head
   count and dilation that cannot be inferred from tensor shapes.

The resolved configuration and its source are written into the result
manifest. A missing key, unexpected key, or tensor-shape mismatch is a fatal
error; evaluation must never continue after a partial checkpoint load.

## Deterministic Evaluation Dataset

The new loader reads `humanml3d_272/split/test.txt` in file order. Each valid
test ID contributes at most one candidate:

- load the complete source motion;
- keep the existing HumanML3D-272 length contract of at least 60 and fewer
  than 300 frames;
- select the first caption tagged as applying to the complete motion
  (`from=0`, `to=0`);
- truncate only the tail so the length is divisible by the MSA-VAE temporal
  unit length;
- normalize with the existing HumanML3D-272 mean and standard deviation.

Tagged subclips are not added as extra candidates. There is no random caption
selection, random crop, shuffle, repeated evaluation, or dropped final batch.
The collator pads only within a batch and returns the true lengths and stable
sample IDs. All physical metrics slice each sample to its true valid length.

This contract evaluates complete test motions and produces an exact diagonal
between the text and motion candidate lists. The result manifest records the
included sample IDs and a stable hash of their ordered identities so two
checkpoints can be verified against the same candidate set.

## Reconstruction and Embedding Data Flow

For each batch:

1. run one seeded MSA-VAE reconstruction pass in evaluation/inference mode;
2. inverse-normalize the valid ground-truth and reconstructed 272-D motions;
3. recover 22-joint global XYZ positions;
4. accumulate physical reconstruction metrics on valid frames;
5. encode ground-truth motions, reconstructed motions, and captions with the
   existing frozen HumanML3D-272 TEMOS/TMR evaluator.

Ground-truth motion embeddings and reconstructed motion embeddings feed FID.
Caption embeddings and reconstructed motion embeddings feed retrieval.
Retrieval therefore measures the semantic fidelity of the reconstructed
motion, not similarity between text and the MSA-VAE's internal alignment
projection.

The evaluator checkpoint is loaded strictly and frozen. The pipeline does not
train or fine-tune an evaluator.

## Metric Definitions

### FID

Compute the mean and covariance of all ground-truth motion embeddings and all
reconstructed motion embeddings from the frozen 272-D evaluator, then use the
existing numerically guarded Fréchet-distance implementation. FID is evaluated
once on the deterministic complete candidate set.

### MPJPE

For every valid frame, subtract joint 0 from all predicted joints and from all
reference joints independently. Compute the Euclidean position error per
joint, average over joints, then average over all valid frames. Multiply metres
by 1000 and report millimetres. This matches MLD's HumanML3D root-aligned
`calc_mpjpe(..., align_inds=[0])` aggregation.

### P-MPJPE

For every valid frame, align the 22 predicted joints to the reference joints
with the MLD similarity-transform Procrustes procedure (translation, rotation,
and scale), then compute mean per-joint Euclidean error. Average over all valid
frames and report millimetres.

### ACCEL

For each sequence, compute the second finite difference
`x[t] - 2*x[t+1] + x[t+2]` for prediction and reference. Compute their
per-joint Euclidean difference, average over joints, then average over exactly
`sum(length - 2)` valid acceleration frames. Report millimetres per frame
squared, without multiplying by FPS squared, matching MLD's published
implementation.

### Skating (%)

Use the OmniControl foot-skating rule on reconstructed global joints:

- left and right foot indices: 10 and 11;
- up axis: Y; horizontal plane: XZ;
- contact: foot height is below 0.05 metres at both adjacent frames;
- instantaneous and smoothed horizontal speed must both exceed 0.50 m/s;
- either foot sliding marks the transition as skating.

The HumanML3D-272 stream is 30 FPS. Horizontal displacement is therefore
multiplied by 30, and OmniControl's five-frame window at 20 FPS is converted to
an eight-frame window to preserve approximately 0.25 seconds of smoothing.
For each sequence, divide its skating transitions by its own number of valid
transitions. The final metric is 100 times the mean of those per-sequence
ratios, matching OmniControl's sample-wise aggregation while excluding padded
transitions. The output manifest records the FPS, joint indices, height
threshold, velocity threshold, and smoothing window.

### Bidirectional Retrieval

L2-normalize all caption embeddings and reconstructed-motion embeddings, then
form one full cosine-similarity matrix over the ordered test candidates. Use
TMR's `normal` exact-diagonal protocol:

- text-to-motion ranks each row of the matrix;
- motion-to-text ranks each row of its transpose;
- ties use TMR's average-rank rule;
- R@1, R@2, and R@3 are reported as percentages in `[0, 100]`;
- median rank is the median zero-based rank plus one.

Semantically equivalent duplicate captions remain negatives, as they do in
TMR's `normal` protocol. No Guo batch-32 sampling, Euclidean distance, or
threshold-based false-negative filtering is mixed into these primary metrics.

## Output Contract

The default output directory is
`output/msa_vae_metrics/<experiment-name>/<checkpoint-stem>/`, which is already
excluded from source control and avoids collisions between equally named
checkpoints from different experiments. Each run writes:

- `evaluation.log` with progress and a compact final table;
- `metrics.json` with unrounded metrics, units, protocol version, resolved
  model configuration, checkpoint identity, evaluator identity, seed, sample
  count, ordered-sample hash, and skating parameters;
- `metrics.csv` with one flat row suitable for an ablation table.

The terminal summary groups reconstruction realism and both retrieval
directions. It labels all units and explicitly states `TMR-full-normal` so the
new numbers cannot be confused with historical Guo-32 R-precision.

## Isolation and Compatibility

Implementation is additive:

- do not modify `eval_msa_vae.py`, `EVAL_msa_vae.sh`, or
  `utils/eval_trans.py`;
- do not change training-time evaluation calls;
- do not change the official MSA-T2M or TAE launchers;
- do not edit or commit any file under the three cloned third-party
  exploration repositories;
- import the existing evaluator through an explicit repository-root path,
  without changing the process working directory;
- keep all new names MSA-VAE-specific to avoid accidental imports by legacy
  workflows.

## Failure Behaviour

Before allocating the model on CUDA, the entrypoint validates the checkpoint,
data split, motion/text directories, normalization files, evaluator
dependencies, and evaluator checkpoint. Errors name the missing or mismatched
artifact and terminate with a non-zero exit code.

The pipeline also rejects:

- an empty deterministic candidate set;
- duplicate stable sample IDs;
- a sequence shorter than three valid frames after unit-length truncation;
- non-finite recovered joints, embeddings, covariance statistics, or final
  metrics;
- mismatched counts or dimensions among captions, sample IDs, and embeddings.

CUDA is the default device for a real evaluation, but metric and dataset unit
tests remain CPU-only.

## Test Strategy

Tests are additive and must not invoke a full dataset evaluation.

1. Dataset tests create tiny temporary HumanML-style files and verify stable
   ordering, first-full-caption selection, deterministic tail truncation,
   filtering, padding, lengths, IDs, and absence of tagged subclip candidates.
2. Reconstruction metric tests use synthetic joints with known translation,
   scale, acceleration, and foot sliding to verify MLD aggregation, units,
   valid-length exclusion, and the 30 FPS OmniControl adaptation.
3. Retrieval tests use small cosine matrices to verify both directions,
   R@1/2/3, one-based median rank, TMR tie averaging, and percentage units.
4. Checkpoint resolver tests cover metadata, adjacent `run.log`, state-shape
   inference, explicit overrides, and fatal strict-load mismatches.
5. Pipeline tests use fake models/evaluators to prove there is one
   reconstruction pass, one deterministic candidate set, correct JSON/CSV
   schemas, and finite-value validation.
6. Launcher and isolation tests verify the positional checkpoint interface,
   shell syntax, repository-relative paths, and that no existing evaluation
   entrypoint imports or calls the new pipeline.

Validation includes targeted unit tests, the full existing test suite,
`py_compile` for changed Python files, `bash -n` for the new launcher, and
`git diff --check`. A real full-test-set CUDA evaluation is not part of
code-edit validation unless explicitly requested.
