# MSA-VAE Full-Sequence Semantic Training Design

**Date:** 2026-07-26
**Status:** Approved

## Objective

Align MSA-VAE training with its downstream use:

- retain the existing 64-frame Causal CNN-VAE pretraining checkpoint as the
  stable local-motion initialization;
- train the Semantic Transformer on complete HumanML3D motions;
- fine-tune the complete MSA-VAE primarily on full sequences while replaying
  periodic 64-frame physical batches;
- preserve the 272-D representation, causal latent convention, model
  state-dict keys, and official evaluation path.

The implementation changes training behavior only. It does not change the
MSA-VAE latent dimensions, Transformer architecture, RAG architecture, or
checkpoint tensor names.

## Current behavior and motivation

`MSAVAEDataset` currently discards motions shorter than `window_size` and
returns a random fixed-size crop. With the authoritative `window_size=64` and
fourfold temporal downsampling, the Semantic Transformer always receives 16
local latent tokens during training.

Official reconstruction evaluation and `get_msa_latent.py` instead encode each
complete motion. HumanML3D sequences can reach approximately 300 frames, or
75 latent tokens. This creates a length-distribution mismatch for positional
encoding, CLS aggregation, and Transformer latent reconstruction.

The global SentenceT5 target describes the complete motion. Applying it to a
random 64-frame crop introduces semantic target noise when the crop contains
only one part of a multi-action sequence. Complete-sequence semantic training
removes that mismatch and makes the role of `h_cls` consistent with its use as
the global retrieval representation.

The existing Phase-1 checkpoint selection is also structurally mismatched.
Phase 1 freezes the CNN reconstruction path, so reconstruction FID and MPJPE
cannot select a trained Semantic Transformer. Phase 2 must consume the final
Phase-1 checkpoint rather than a reconstruction-selected checkpoint.

## Training curriculum

### Existing Causal CNN-VAE pretraining

Keep the existing pretrained Causal TAE and its 64-frame training unchanged.
This preserves the current local-motion initialization and produces a clean
ablation against the old MSA-VAE schedule.

The default CNN has a theoretical receptive field longer than 64 frames.
Multi-length CNN pretraining is therefore a possible future ablation, but it
is not part of this change.

### Phase 1: full-sequence semantic training

Freeze `cnn_encoder`, `cnn_decoder`, and `decode_proj`.

For every sample:

1. trim the motion length down to an exact multiple of
   `unit_length = stride_t ** down_t`;
2. normalize the complete valid sequence;
3. pad only within the batch;
4. extract deterministic CNN posterior means `mu`;
5. mask padded latent tokens in the Transformer encoder and decoder;
6. optimize masked Transformer latent reconstruction, global alignment, and
   local alignment.

Use a semantic-only model forward that omits posterior sampling and the CNN
decoder. The CNN is frozen and no physical reconstruction objective is
computed in this phase.

The CLS global target is the complete HumanML3D text embedding. Full-sequence
training sets Spotlight interpolation to zero; local pooled text is not mixed
into the global target. Local semantic supervision remains an independent
token-level objective.

Phase 1 writes `net_last.pth` at every evaluation boundary and after the final
optimizer step. The Phase-2 launcher loads that checkpoint. Reconstruction
metrics may still be reported for monitoring but do not define the Phase-1
handoff.

### Phase 2: full-sequence joint tuning with window replay

Use two views over one loaded training dataset:

- `full`: complete trimmed motion, padded per batch;
- `window`: random 64-frame crop with the existing local target construction.

The deterministic schedule uses a configurable positive
`window_replay_interval`. Step numbers divisible by the interval use the
window view; all other steps use the full view. The authoritative default is
4, giving 75% full-sequence steps and 25% replay steps without rank-local
random decisions.

Full steps optimize all six objectives:

```text
L_full = L_recon + L_KL + lambda_root * L_root
       + lambda_latent * L_latent
       + lambda_global * L_global
       + lambda_local * L_local
```

Replay steps preserve the physical/local representation:

```text
L_replay = L_recon + L_KL + lambda_root * L_root
         + lambda_local * L_local
```

The model still executes its standard forward on replay steps. Transformer
and global outputs contribute differentiable zero terms so distributed
training does not acquire rank-dependent unused parameters.

CNN parameters retain the existing differential learning-rate group. The
authoritative launcher keeps `cnn_lr_scale=0.1`; the value remains explicit
and overridable for experiments.

## Dataset and batch contract

Extend `humanml3d_272/dataset_msa_vae.py` instead of creating a parallel
HumanML loader.

The dataset owns the loaded records and exposes:

```python
get_item(index: int, sequence_mode: str) -> tuple
```

where `sequence_mode` is `full` or `window`. Lightweight dataset views select
one mode without duplicating the loaded motion records.

Every item returns the existing fields plus its valid motion length:

```text
motion
caption
global_text_embedding
has_global
local_text_embedding_at_latent_rate
has_local
total_source_frames
pooled_local_text_embedding
motion_length
```

The collator pads motion to the maximum frame length and local text to the
maximum latent length in the batch. Window items remain exactly 64 frames and
are compatible with the same contract.

The full-sequence loader uses length-bucketed shuffled batches to reduce
padding. Full and window batch sizes are separate command-line options because
their memory costs differ.

## Length and mask semantics

The causal strided convolutions produce floor-divided output lengths. Valid
latent lengths are:

```text
latent_length = floor(floor(motion_length / stride_t) / stride_t)
```

for the current two downsampling layers. Input lengths are trimmed to a
multiple of the aggregate unit length, so no valid partial latent token is
created.

Construct:

- a frame-valid mask `(B, T)` for reconstruction and root objectives;
- a latent-valid mask `(B, T')` for KL, Transformer latent reconstruction,
  and local semantic alignment;
- the inverse latent mask for Transformer padding.

Padding must never affect a loss, target average, logged count, or gradient.
Right padding cannot affect earlier valid outputs of the causal CNN, but its
outputs still require masking.

## Length-invariant losses

The current reconstruction and root losses sum every tensor element, while KL
sums every latent element. That makes complete-sequence samples and larger
padded batches change objective scale.

For full-sequence training, each dense objective uses:

1. a mean over valid time and feature dimensions for each sample;
2. a mean over valid samples.

This applies to:

- optimal-sigma Gaussian reconstruction;
- optimal-sigma root reconstruction;
- KL divergence;
- Transformer latent MSE;
- local cosine alignment.

Global alignment remains a per-sample masked mean. A globally empty semantic
mask returns a differentiable zero. Local alignment combines `has_local` with
the latent-valid mask.

The new normalized objectives intentionally make loss scale independent of
sequence length and full-sequence batch size. Existing checkpoints remain
loadable, but training curves are not numerically comparable to the legacy
summed reconstruction curves.

## Model interface

Add posterior-statistics extraction without changing parameterized modules:

```python
CausalEncoder.encode_stats(x) -> (mu, logvar)
MSA_VAE.forward_semantic(x, lengths) -> output_dict
```

`CausalEncoder.forward` calls `encode_stats` and then reparameterizes, so all
state-dict keys remain identical.

`MSA_HumanVAE.forward` accepts a `semantic_only` flag. Keeping the behavior
behind the normal forward call allows Accelerate/DDP to observe the executed
trainable parameters.

The existing inference and evaluation calls without `semantic_only` remain
unchanged.

## Options and authoritative launchers

Add:

- `--sequence_mode {window,full,mixed}`;
- `--full-seq-batch-size`;
- `--window-replay-interval`;
- `--length-bucket-size`.

Defaults preserve direct legacy invocation with `sequence_mode=window`.
Authoritative scripts pass explicit values:

- Phase 1: `sequence_mode=full`;
- Phase 2: `sequence_mode=mixed`, replay interval 4;
- evaluation: unchanged complete-sequence evaluation.

Launchers expose the full-sequence batch size and replay interval through
overridable environment variables. They do not hard-code GPU IDs or
machine-specific paths.

Phase 2 loads `${PHASE1_DIR}/net_last.pth`. Experiment names state
`fullseq` or `fullseq_replay` so legacy and new checkpoints are not confused.

## Checkpoint metadata

Keep the existing `{"net": state_dict}` payload and add a `metadata` object:

- phase;
- sequence mode;
- window size;
- full-sequence batch size;
- replay interval;
- aggregate temporal unit length;
- normalized-loss version.

Loading continues to read the `net` key and remains compatible with legacy
payloads that have no metadata. New training validates structural metadata
when resuming a full MSA-VAE checkpoint and rejects incompatible temporal
downsampling or latent dimensions.

## Evaluation and downstream artifacts

Official reconstruction evaluation already processes valid complete
sequences one sample at a time and remains authoritative.

After training, all MSA-derived local latents, deterministic `mu` latents, and
`h_cls` retrieval features must be regenerated from the selected Phase-2
checkpoint before RAG training. Old latent directories must not be mixed with
the new checkpoint.

For research comparison, report reconstruction and downstream T2M metrics by
motion-length bins:

- up to 64 frames;
- 65 to 128 frames;
- more than 128 frames.

The implementation adds the length-bin utility and logging contract but does
not run full evaluation or RAG retraining.

## Failure handling

Fail before training when:

- Phase 1 or Phase 2 requests a non-window mode with a full-sequence batch size
  below one;
- mixed mode is requested outside Phase 2;
- Phase 1 does not use full mode;
- Phase 2 mixed mode has a replay interval below two;
- a motion cannot produce at least one complete latent token;
- motion and local target lengths cannot be aligned;
- resumed checkpoint metadata conflicts with the requested temporal
  structure.

Missing local supervision remains valid and disables only the local semantic
loss for that sample.

## Testing and validation

Add CPU/synthetic tests for:

- full-sequence trimming and local-target alignment;
- full and window dataset views sharing records;
- variable-length padding and exact returned lengths;
- length-bucketed batches;
- floor-divided latent masks;
- masked reconstruction, root, KL, latent, and alignment losses;
- invariance to padded values and duplicated sequence length;
- semantic-only forward output shapes and absence of CNN decoding;
- unchanged state-dict keys;
- deterministic replay scheduling;
- phase/mode option validation;
- launcher arguments and Phase-1 final-checkpoint handoff;
- checkpoint metadata compatibility.

Run:

```bash
conda run -n mgpt python -m unittest discover -s tests -v
conda run -n mgpt python -m py_compile <changed-python-files>
bash -n TRAIN_msa_vae_phase1.sh
bash -n TRAIN_msa_vae_phase2.sh
git diff --check
```

Do not launch full training, TMR evaluation, SentenceT5 preprocessing, or GPU
benchmarks as implementation validation.
