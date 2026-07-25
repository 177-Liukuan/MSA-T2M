# MSA-VAE Domain-Specific Sparse Supervision Design

**Date:** 2026-07-25
**Status:** Approved

## Objective

Validate that MSA-VAE can learn from partially available semantic supervision
without mixing incompatible motion normalization domains.

The work produces two independent experiments:

1. HumanML3D full training, where global supervision is always present and
   local BABEL supervision is available only for the overlap subset.
2. BABEL sparse-global training, where local supervision is always present
   and global HumanML3D supervision is available only for an overlap bridge
   subset.

The BABEL experiment targets reconstruction and local semantic alignment. It
does not add a BABEL text-to-motion generation pipeline.

## Current behavior

The authoritative HumanML3D launchers already pass `--no_ft_split`. Therefore
the current mainline is not trained only on the HumanML3D/BABEL intersection:

- all HumanML3D training motions are loaded;
- every sample has HumanML3D global text supervision;
- only samples with a valid file in `humanml3d_272/t5_enc_single/` receive
  local supervision;
- `has_local` masks local loss for HumanML3D-only samples.

The current implementation does not load BABEL-only motions into MSA-VAE.

The existing local T5 cache contains 7,056 training files corresponding to
`humanml3d_272/split/train_ft.txt`. Its arrays remain at approximately 20 FPS;
`dataset_msa_vae.py` upsamples them to the 30 FPS motion length at load time.

## Reference from UniMotion

UniMotion provides the intended supervision semantics:

- `HumanML3D_Full`: global text is always available and local text may be
  missing;
- `Babel_full`: local text is always available and global text may be missing.

For BABEL training, UniMotion combines:

- HumanML3D/BABEL overlap samples with both global and local supervision;
- BABEL-only samples with local supervision and a disabled global loss.

It does not require an unreliable `seq_*` to HumanML3D ID mapping. This design
ports that behavior to the repository's 272-dimensional, 30 FPS data.

## Experiment boundaries

### HumanML3D full

This remains the official mainline behavior:

- motions: `humanml3d_272/motion_data/`;
- split: `humanml3d_272/split/train.txt`;
- normalization: `humanml3d_272/mean_std/`;
- Causal TAE: the existing HumanML3D checkpoint;
- global target: HumanML3D SentenceT5 embedding for every valid sample;
- local target: existing overlap cache when available;
- supervision flags:
  - `has_global=True`;
  - `has_local=True` only when the local cache is valid.

Existing HumanML3D training and evaluation entrypoints must retain their
current behavior and checkpoint compatibility.

### BABEL sparse-global

The BABEL training dataset has two sources with one shared sample contract.

#### Overlap bridge source

- IDs: `humanml3d_272/split/train_ft.txt`;
- motions: `humanml3d_272/motion_data/`;
- global target: existing HumanML3D SentenceT5 embedding;
- local target: existing `humanml3d_272/t5_enc_single/`;
- flags: `has_global=True`, `has_local=True`.

Bridge samples are required to have both targets. Missing or malformed
supervision is an error, not a reason to silently downgrade a bridge sample.

#### BABEL-only source

- motions: `babel_272_stream/train_stream/`;
- local annotation: `babel_272_stream/train_stream_text/`;
- local target: a new offline 30 FPS SentenceT5 cache;
- global target: a zero placeholder used only for collation;
- flags: `has_global=False`, `has_local=True`.

The BABEL workflow uses:

- normalization: `babel_272/t2m_babel_mean_std/`, which contains joint
  HumanML3D/BABEL statistics;
- pretrained Causal TAE:
  `Experiments/causal_TAE_t2m_babel_272_h100_20260205/net_best_mpjpe.pth`.

This experiment is named **BABEL sparse-global**, rather than pure BABEL,
because its bridge subset contains HumanML3D overlap motions.

## Offline BABEL local target cache

Add `scripts/prepare_babel_stream_t5.py`.

Each BABEL-stream text record contains two action descriptions and the first
segment's ending frame. The preprocessor:

1. parses both descriptions and the boundary;
2. collects and deduplicates all action strings;
3. encodes each unique string once with the configured SentenceT5 model;
4. repeats the first embedding before the boundary;
5. repeats the second embedding from the boundary to the motion end;
6. asserts that the result is `(motion_frames, text_embed_dim)`;
7. writes float32 arrays under:
   - `babel_272_stream/t5_enc_single/train/`;
   - `babel_272_stream/t5_enc_single/val/`.

The cache is local data and remains ignored by Git.

The preprocessor also writes a manifest containing:

- format version;
- SentenceT5 model identifier/path signature;
- embedding dimension;
- source motion and text roots;
- split and file count;
- input signatures;
- valid and rejected sample counts.

Training rejects an absent, incomplete, incompatible, or stale manifest.
Existing cache files are not overwritten unless explicitly requested.

## Dataset and training interfaces

Add `humanml3d_272/dataset_msa_vae_babel.py`. It owns the bridge and
BABEL-only adapters and returns the same tuple contract as
`dataset_msa_vae.py`:

- normalized motion window;
- caption or placeholder;
- global embedding and `has_global`;
- latent-rate local embedding and `has_local`;
- total frame count;
- pooled local embedding.

Keeping this loader separate avoids introducing BABEL branches into the
authoritative HumanML3D loader.

Extend the MSA-VAE options with an explicit data mode:

- `humanml_full`;
- `babel_sparse_global`.

`train_msa_vae.py` selects the dataset and validation path from this mode.
Model construction and state-dict keys remain unchanged.

Add independent entrypoints:

- `TRAIN_msa_vae_babel_phase1.sh`;
- `TRAIN_msa_vae_babel_phase2.sh`;
- `EVAL_msa_vae_babel.sh`.

The BABEL launchers use distinct experiment names and must not load a
HumanML3D Causal TAE or MSA-VAE checkpoint.

## Loss semantics

Every sample always contributes the phase-appropriate motion objectives:

- reconstruction;
- KL divergence;
- root reconstruction;
- Transformer latent reconstruction.

Semantic losses use per-sample validity masks:

```text
L = L_recon + L_KL + lambda_root * L_root
  + lambda_latent * L_latent
  + lambda_global * masked_mean(L_global, has_global)
  + lambda_local * masked_mean(L_local, has_local)
```

The command-line `lambda_global` and `lambda_local` values remain fixed.
Missing supervision disables only the corresponding sample contribution; it
does not mutate the weight for the whole batch.

For distributed training:

- global alignment is normalized by the total valid global sample count
  across all ranks;
- local alignment is normalized by the total valid local token count across
  all ranks;
- a globally empty mask produces a differentiable zero;
- rank-local valid counts must not change the effective loss scale.

Log:

- global and local valid ratios;
- global valid sample count;
- local valid token count;
- masked global and local losses.

## Validation

### HumanML3D

Keep the current HumanML3D FID, MPJPE, R-Precision, and checkpoint selection
behavior unchanged.

### BABEL sparse-global

Use a dedicated BABEL-stream validation loader with the same joint
normalization and offline local cache. Report:

- reconstruction MPJPE;
- reconstruction loss;
- KL divergence;
- latent reconstruction loss;
- mean cosine similarity over valid local tokens;
- local alignment loss;
- global and local supervision coverage.

During Phase 1, the CNN reconstruction path is frozen, so MPJPE cannot
meaningfully select the Transformer/semantic checkpoint. Select
`net_best_semantic.pth` by the validation latent-plus-local-alignment
objective. During Phase 2, select `net_best_mpjpe.pth` by BABEL validation
MPJPE. Retain `net_last.pth` in both phases. Do not use HumanML3D TMR/FID to
select BABEL checkpoints.

Checkpoint payloads keep the existing `net` state dict. Additional metadata
records the data mode, normalization paths, cache manifest identity,
supervision coverage, loss weights, and training arguments.

## Failure handling

Fail before training when:

- required motion, cache, normalization, or Causal TAE paths are missing;
- a bridge sample lacks either semantic target;
- a BABEL-only sample lacks a valid local target;
- a BABEL record cannot be parsed or has an invalid boundary;
- cached embedding shape, dimension, or length does not match the motion;
- the cache manifest does not match the requested data or SentenceT5 setup;
- distributed ranks resolve different cache manifests or data modes.

Missing global text on BABEL-only samples is expected and must not be treated
as an error.

The cache builder returns a nonzero status with a deterministic rejection
report rather than silently skipping malformed data.

## Testing and verification

Add CPU/synthetic tests for:

- BABEL two-segment text parsing;
- boundary expansion to exact 30 FPS motion length;
- unique-text deduplication before SentenceT5 encoding;
- cache manifest validation and corrupt-cache rejection;
- bridge and BABEL-only supervision flags;
- mixed-batch collation and shapes;
- all-global, partial-global, and no-global masked losses;
- all-local and no-local masked losses;
- single-rank and simulated distributed normalization equivalence;
- preservation of HumanML3D loader behavior;
- preservation of model state-dict keys and checkpoint loading.

Run proportional static checks:

```bash
conda run -n mgpt python -m py_compile <changed-python-files>
bash -n <changed-shell-files>
git diff --check
```

Do not launch full training or TMR evaluation as implementation validation.

## Experiment interpretation

The two runs test the same hypothesis from opposite directions:

- HumanML3D full: dense global supervision plus sparse local supervision;
- BABEL sparse-global: dense local supervision plus sparse global
  supervision.

This supports the claim that MSA-VAE can use heterogeneous annotations through
explicit missing-supervision masks without inventing labels or mixing
incompatible checkpoints.

The experiments remain domain-specific. Results must not be presented as a
single HumanML3D/BABEL union model.
