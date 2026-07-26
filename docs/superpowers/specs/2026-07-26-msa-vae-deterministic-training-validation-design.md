# MSA-VAE Deterministic Training Validation Design

## Objective

Replace the HumanML3D MSA-VAE training-time validation path with a
deterministic complete-motion validation protocol that is consistent with the
standalone standard evaluator, does not perturb training randomness, and
continues saving FID- and MPJPE-selected diagnostic checkpoints.

This repair applies only to the HumanML3D MSA-VAE workflow. The separate BABEL
sparse-global validation path remains unchanged.

## Confirmed Problem

The current HumanML3D training path constructs validation data with
`humanml3d_272.dataset_eval_t2m.DATALoader(..., is_test=False)`. That legacy
dataset:

- includes complete motions and eligible caption-derived motion segments;
- chooses captions, temporal rounding, and crop offsets randomly;
- shuffles validation samples;
- drops the final incomplete batch;
- pads every motion to 300 frames.

The MSA-VAE posterior also samples fresh Gaussian noise during reconstruction.
Consequently, successive validation calls use different motions, crops,
captions, order, and reconstruction noise. Validation consumes the same
process RNG streams used by training, so changing `eval_iter` can change the
subsequent optimization trajectory.

For the same Phase-1 smoke checkpoint, the legacy validation path evaluated
864 samples and reported FID 1.0131, while the deterministic complete-motion
validation protocol evaluated 802 samples and reported FID 0.954017. On the
same Phase-2 smoke checkpoint, complete-motion validation reported FID
0.949756 on the 802-sample validation split and FID 0.509117 on the
2,480-sample test split. Stable MPJPE around 21--22 mm excludes a
normalization or fixed-TAE corruption.

The approximately 0.95 validation FID and approximately 0.51 test FID are
different protocol outcomes. This repair must make validation reproducible; it
must not force validation FID to match the test value.

## Selected Approach

Use the standalone standard evaluator's in-process metric core during
training.

Alternatives were rejected:

- repairing the legacy validation loop would preserve two independent metric
  implementations that can drift;
- spawning the standalone evaluator as a subprocess would isolate state, but
  repeatedly serializing and loading large checkpoints would add unnecessary
  overhead.

The selected in-process approach shares the deterministic dataset, collation,
motion padding, TMR embedding, reconstruction, and retrieval implementations
with final evaluation while retaining training-controlled logging and
checkpoint saving.

## Deterministic Validation Dataset

HumanML3D training validation will use
`MSAVAEMetricsDataset` with `humanml3d_272/split/val.txt`.

The validation population is:

- one complete motion per eligible validation ID;
- the first complete-motion caption;
- lengths rounded down to the MSA-VAE temporal unit;
- no caption-derived segments;
- no temporal crop;
- no shuffle;
- no dropped final batch.

The loader uses a fixed batch size of 32 and the existing standard dynamic
collator. The frozen TMR motion encoder uses dynamic masks (`max_len=-1`) so
its mask width matches each dynamically padded batch.

The validation sample count and ordered sample hash will be logged once at
startup. An empty or malformed validation dataset remains a fatal preflight
error.

## RNG Isolation

Every HumanML3D training validation call will execute inside an RNG isolation
guard:

1. save Python `random`, NumPy, PyTorch CPU, and all available CUDA RNG states;
2. seed those RNGs with a fixed validation seed;
3. run the complete validation pass;
4. restore every saved RNG state in a `finally` block.

The fixed validation seed defaults to the existing evaluation seed contract,
123. It is independent of the training seed so all training seeds are measured
under the same stochastic reconstruction condition.

Restoration occurs even when validation raises. The model's pre-validation
training/evaluation mode is also restored. Validation must therefore neither
change subsequent training randomness nor leave the model in evaluation mode.

The fixed validation batch size and validation seed become checkpoint
training-identity fields. The three-seed aggregator will consequently reject
runs trained under different internal validation conditions.

## Metrics and Checkpoint Selection

Each validation call computes the same metric family as standalone standard
evaluation:

- FID;
- MPJPE and P-MPJPE;
- acceleration;
- skating percentage;
- bidirectional TMR retrieval.

Training logs and TensorBoard receive the validation values under an explicit
complete-validation namespace. The log message identifies the split,
deterministic sample count, validation seed, and batch size so the value cannot
be confused with the 2,480-sample test result.

Both Phase 1 and Phase 2 retain all existing diagnostic checkpoint artifacts:

- `net_best_fid.pth` when deterministic validation FID strictly improves;
- `net_best_mpjpe.pth` when deterministic validation MPJPE strictly improves;
- `net_last.pth` at every validation boundary and at the end of training.

Iteration 0 participates in both best comparisons, preserving current
behavior. Checkpoints keep the complete training metadata and fixed-TAE
lineage.

The formal alignment--realism table continues to evaluate the fixed-iteration
Phase-2 `net_last.pth`. Best checkpoints are diagnostic artifacts and are not
accepted as substitutes by the formal experiment protocol.

## Visualization

Training-time numeric validation will not render GIFs or TensorBoard videos.
This removes the multi-minute headless rendering step and its XDG/ALSA
warnings from every validation interval.

Motion visualization remains available through existing explicit evaluation
or visualization entrypoints; this repair does not delete those facilities.

## Distributed Execution

The unwrapped MSA-VAE and frozen evaluator remain available on every rank, but
only the main rank executes the complete HumanML3D validation loader and
writes metrics or checkpoints. All ranks synchronize immediately before and
after validation.

RNG state is saved and restored on every rank. Non-main ranks do not consume
validation data or model randomness while waiting, so distributed training
resumes from the same per-rank RNG state it had before validation.

The validation loader is not prepared or sharded by Accelerate because the
main rank must evaluate the complete ordered validation set.

## Failure Handling

Validation fails closed on:

- zero evaluated samples;
- non-finite metric output;
- TMR embedding/sample count mismatch;
- missing or malformed validation assets;
- failure to restore model mode or RNG state.

No best or last checkpoint is published from a failed validation call.
Existing checkpoints are not removed.

## Testing

Implementation follows test-driven development.

1. A dataset/loader test proves the training validation population contains
   only deterministic complete motions, has stable order, keeps the final
   partial batch, and exposes a stable sample hash.
2. An RNG-guard test proves Python, NumPy, PyTorch CPU, and CUDA states are
   restored after both success and exception.
3. A validation-helper test proves repeated calls with the same model and
   seed return identical metrics and restore the model's prior mode.
4. A checkpoint-selection test proves both phases save
   `net_best_fid.pth`, `net_best_mpjpe.pth`, and `net_last.pth` with metadata,
   using strict improvement for best files.
5. A logging test proves the validation split, sample count, seed, and batch
   size are recorded.
6. A regression test proves numeric validation never calls GIF/video
   rendering.
7. Checkpoint metadata and aggregation tests prove internal validation seed
   and batch size are required and compatible across the three training
   seeds.
8. Run the complete unit suite, Python compilation, shell syntax checks, and
   `git diff --check`.
9. Run a one-call GPU diagnostic on the existing smoke checkpoint and
   validation split. It must reproduce the deterministic validation range and
   leave an explicitly captured RNG probe unchanged.

No full training run starts as part of implementation validation.
