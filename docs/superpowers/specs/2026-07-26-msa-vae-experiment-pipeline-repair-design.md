# MSA-VAE Experiment Pipeline Repair Design

## Objective

Make the HumanML3D MSA-VAE alignment--realism experiment pipeline runnable
and reproducible from fresh two-GPU Phase-1 training through Phase-2
fine-tuning, per-checkpoint standard evaluation, and three-seed ablation-table
aggregation.

The target table columns are:

| Variant | FID | MPJPE | P-MPJPE | ACCEL | Skating | T2M R@1 | T2M R@5 | T2M MedR | M2T R@1 | M2T R@5 | M2T MedR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

All formal MSA-VAE runs start from the same approved Causal TAE checkpoint.
Existing MSA-VAE checkpoints and their measurements are excluded from the
paper results.

## Scope

The repair has five bounded parts:

1. make full-HumanML training tolerate source motions that cannot produce one
   temporal latent unit;
2. make the standalone evaluator's dynamic batch padding compatible with the
   frozen TMR motion encoder;
3. add text-to-motion and motion-to-text R@5 to the standard evaluation
   artifacts;
4. aggregate three per-seed result manifests into one mean-and-standard-
   deviation ablation row;
5. parameterize the existing two-stage launchers so four two-GPU jobs can run
   concurrently without output collisions.

The model architecture, 272-D representation, causal latent convention,
normalization, frozen evaluator weights, TAE checkpoint, and legacy
evaluation entrypoints remain unchanged.

## Confirmed Failures

### Full-split training failure

`humanml3d_272/split/train.txt` includes motion `000990`, whose stored motion
has three frames. With a four-frame temporal latent unit,
`MSAVAEDataset` raises `MotionSequenceTooShortError` and aborts dataset
construction on every DDP rank. The official `--no_ft_split` Phase-1 launcher
therefore fails before the first optimizer step.

### Evaluator padding failure

`collate_msa_vae_metrics` pads only to the longest motion in a batch. The
frozen `ActorAgnosticEncoder` is currently constructed with `max_len=300`, so
it builds a 300-frame padding mask even when the input tensor has fewer
frames. The first real batch produced 288 input frames, a 290-token
Transformer input, and a 302-token mask.

A diagnostic run with dynamic evaluator masks completed all 2,480 test
motions. On the same batch, dynamic padding and explicit zero-padding to 300
produced bit-identical evaluator embeddings.

### Output-schema mismatch

The evaluator currently exports R@1, R@2, and R@3, while the requested table
requires R@1 and R@5. There is no three-seed result aggregator.

### Launcher collision risk

The Phase-1 and Phase-2 launchers hard-code experiment names and alignment
weights. Concurrent ablations would target the same output directories unless
users bypass the authoritative launchers with manually duplicated commands.

## Design

### 1. Deterministic invalid-motion filtering

`MSAVAEDataset` will skip a source record when its motion has fewer than
`unit_length` frames. It will not special-case a sample ID or modify the
dataset split.

The dataset will count skipped sub-unit motions and include that count in its
construction summary. All DDP ranks read the same split and apply the same
deterministic rule, so sampler lengths remain consistent.

Motions that are shorter than the 64-frame replay window but contain at least
one latent unit remain valid in Phase-1 full-sequence mode. Phase-2 mixed mode
retains its existing behavior of filtering them when it constructs the shared
replay-capable dataset. Changing that Phase-2 sampling population is outside
this repair.

Unexpected I/O, shape, caption, or embedding problems keep the existing
handling unless a focused test demonstrates that they belong to this repair.
This change only replaces the fatal sub-unit-motion branch with an explicit,
counted skip.

### 2. Dynamic TMR motion masks

The standalone MSA-VAE standard evaluator will construct its frozen TMR motion
encoder with `max_len=-1`. This makes the encoder derive the padding-mask width
from the true maximum `lengths` value, which equals the dynamically padded
input width.

The third-party evaluator source will not be edited. The standalone evaluator
will continue zeroing reconstructed padding before embedding, and all physical
metrics will continue slicing to true lengths.

A regression test will exercise a real `ActorAgnosticEncoder`-compatible
dynamic batch whose padded width is less than 300 and prove that the mask
matches the input. A second assertion will preserve the already confirmed
equivalence between dynamic input and explicit 300-frame padding.

### 3. Additive R@5 reporting

The retrieval implementation will calculate R@5 from the same full
L2-normalized cosine similarity matrix and the same average-tie ranks used for
the existing metrics:

`R@5 = mean(rank < 5) * 100`.

R@2 and R@3 remain in machine-readable output for compatibility. R@5 is added
to:

- the metric dictionary;
- required manifest keys;
- unit metadata;
- terminal and durable log summaries;
- flat CSV output;
- retrieval and artifact tests.

Because the schema gains new required fields, the protocol version will be
bumped from `msa-vae-standard-v1` to `msa-vae-standard-v2`. Retrieval
semantics remain `TMR-full-normal`.

### 4. Three-seed aggregation

A separate MSA-VAE-specific command will consume exactly three
`metrics.json` files for one named variant. It will fail closed unless:

- all inputs use protocol `msa-vae-standard-v2`;
- evaluator SHA-256, ordered sample hash, sample count, and skating
  configuration match;
- checkpoint SHA-256 values are distinct;
- the training seeds are distinct and recorded in the fresh checkpoint
  metadata copied into each evaluation manifest;
- every target table metric is finite.

For each requested metric, the aggregator will report arithmetic mean and
sample standard deviation (`ddof=1`). It will write:

- one flat CSV row containing numeric `<metric>_mean` and `<metric>_std`
  fields;
- one Markdown row formatted as `mean ± std`;
- a JSON manifest listing all source evaluations and compatibility identities.

The aggregator will not average R@2/R@3 because they are not part of the
requested table.

### 5. Safe two-GPU launcher parameterization

The existing authoritative Phase-1 and Phase-2 launchers will retain their
current defaults while accepting environment overrides for at least:

- `EXP_NAME`;
- `GLOBAL_ALIGN_WEIGHT`;
- `LOCAL_ALIGN_WEIGHT`;
- `SEED`;
- `TOTAL_ITER`;
- `EVAL_ITER`;
- `OUT_DIR`;
- the Phase-1 source directory for Phase 2.

Each launcher will print the resolved experiment name, seed, alignment
weights, output root, and GPU count before launching. It will reject an empty
experiment name and invalid negative alignment weights.

Formal scheduling will set one unique experiment name per
variant/weight/seed and bind four non-overlapping pairs:

- GPUs 0--1;
- GPUs 2--3;
- GPUs 4--5;
- GPUs 6--7.

The launchers will not assign GPU IDs internally; the scheduler or caller will
set `CUDA_VISIBLE_DEVICES` so the same launcher remains portable.

## Checkpoint and Result Contract

Fresh Phase-1 checkpoints continue to initialize from:

`Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth`

Phase 2 resumes the matching Phase-1 `net_last.pth`. Formal comparison uses
the fixed-iteration Phase-2 `net_last.pth`, avoiding FID- or MPJPE-biased
checkpoint selection.

Checkpoint metadata will record the training seed, global alignment weight,
local alignment weight, experiment name, and complete model-construction
arguments needed by the evaluator and aggregator. The existing tensor state
keys remain unchanged.

Per-checkpoint evaluation uses a fixed evaluation seed and writes an identity
manifest containing checkpoint SHA-256, evaluator SHA-256, sample hash,
protocol version, units, metric values, and the checkpoint's recorded training
identity. This makes the three-seed aggregator independent of experiment
directory names or adjacent logs for newly trained checkpoints; adjacent logs
remain only a legacy diagnostic fallback and are not accepted for the formal
fresh-run aggregation.

## Validation

Implementation follows test-driven development.

1. Add a failing dataset test that combines a valid full-sequence sample with
   a three-frame sample and expects one retained record plus one counted skip.
2. Add a failing real-mask regression test for a batch shorter than 300.
3. Add failing R@5 tests, including ranks 4 and 5 and average-tie behavior.
4. Add failing artifact tests for the v2 R@5 fields.
5. Add failing aggregation tests for mean/std output and every compatibility
   rejection.
6. Add failing launcher tests for override forwarding and unique names.
7. Run the focused unit suite, Python compilation, shell syntax checks, and
   `git diff --check`.
8. Run a real two-GPU one-step Phase 1 from the fixed TAE checkpoint.
9. Run a real two-GPU one-step Phase 2 from that newly produced Phase-1
   checkpoint.
10. Run the unpatched standard evaluator on the new Phase-2 smoke checkpoint
    and verify v2 JSON, CSV, and log artifacts contain all target metrics.

No full 50k training or formal result collection begins until this repair
validation passes.
