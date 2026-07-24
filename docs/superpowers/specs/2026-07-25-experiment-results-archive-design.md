# MSA-T2M Exploration Experiment Results Archive Design

## Context

`Experiments/` currently contains official MSA-T2M results, formal ablations,
upstream baselines, and negative or superseded research routes at the same
level. The exploration results occupy approximately 1.1 TB across 35
directories, which makes the official experiment lineage difficult to read.

This change archives only results belonging to clearly separate exploration
methods. Official and historical development results for Causal TAE, MSA-VAE,
and Global-RAG DDPM remain directly under `Experiments/`.

## Goals

- Move results from clearly identified exploration methods into one
  `Experiments/explorations/` tree.
- Keep all Causal TAE, `MSA_VAE*`, Global-RAG DDPM, and formal ablation
  directories at the `Experiments/` root.
- Preserve every checkpoint, log, TensorBoard event, and generated artifact.
- Update runnable exploration entrypoints so they read and write the new
  locations.
- Produce a tracked manifest containing the exact old-to-new mapping and
  post-move verification results.
- Make the operation reversible without old-path compatibility symlinks.

## Non-goals

- Deleting, deduplicating, compressing, or pruning experiment artifacts.
- Renaming individual experiment runs.
- Selecting a new official checkpoint.
- Moving official results into an additional `official/` or `mainline/`
  directory.
- Moving datasets, latent caches, retrieval databases, or checkpoints outside
  `Experiments/`.
- Rewriting historical documents merely because they mention an old command.

## Root Results That Must Remain

The following classes stay directly under `Experiments/`:

- `Causal_TAE` and every `causal_TAE*` directory, including Babel variants.
- Every directory whose name begins with `MSA_VAE`.
- Every Global-RAG DDPM result, including early versions, test-code versions,
  K ablations, and component ablations.
- In particular, `MotionStreamer_t2m_272_msa_rag*` stays at the root unless
  the name explicitly identifies Rectified Flow, MCA, latent retrieval, or
  local-RAG cross-attention.

The retained Global-RAG set includes names containing:

- `k1`, `k2`, `k4`, `k5`, `k7`, or `k9`;
- `no_rag`, `no_local`, `no_global`, or `no_decoupling`;
- `testcode`, `test_no_rag`, or other early Global-RAG DDPM identifiers.

## Destination Layout

```text
Experiments/
├── <Causal TAE, MSA-VAE, Global-RAG DDPM, and ablation results>
└── explorations/
    ├── clip/
    ├── rectified_flow/
    ├── cross_attention/
    │   ├── mca/
    │   ├── latent_retrieval/
    │   └── local_rag/
    ├── qformer/
    ├── representation_experiments/
    ├── motionstreamer_baselines/
    └── misc/
```

No symlink or compatibility directory remains at an old result path.

## Exact Move Manifest

### CLIP

Destination: `Experiments/explorations/clip/`

- `MotionStreamer_t2m_272_baseline_clip`

### Rectified Flow

Destination: `Experiments/explorations/rectified_flow/`

- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_100000Iter_addEMA`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_200000Iter`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_tuned`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_tuned_addLR`

### MCA Cross-Attention

Destination: `Experiments/explorations/cross_attention/mca/`

- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose_fix`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5_Flamingo`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5_Flamingo_gateclose`

### Latent-Retrieval Cross-Attention

Destination:
`Experiments/explorations/cross_attention/latent_retrieval/`

- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_6layer_top3_ddpm`
- `MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_late_after_sa_every1layer_top3_ddpm_cfg_saca_dropout01`

### Local-RAG Cross-Attention

Destination: `Experiments/explorations/cross_attention/local_rag/`

- `MotionStreamer_t2m_272_msa_rag_local_L16_k3_sa_ca`
- `MotionStreamer_t2m_272_msa_rag_local_L4_k3`
- `MotionStreamer_t2m_272_msa_rag_local_L4_k3_crossattn`
- `MotionStreamer_t2m_272_msa_rag_local_L8_k3`
- `MotionStreamer_t2m_272_msa_rag_local_L8_k3_crossattn`

### Q-Former

Destination: `Experiments/explorations/qformer/`

- `QFormer_t2m_272_v1`
- `QFormer_t2m_272_v2`
- `QFormer_t2m_272_v3`
- `QFormer_t2m_272_v4`
- `QFormer_t2m_272_v5`

### Representation Experiments

Destination: `Experiments/explorations/representation_experiments/`

- `SAE_v1_t2m_272`
- `TAE_GAN_Loss_`

### MotionStreamer Baselines

Destination: `Experiments/explorations/motionstreamer_baselines/`

- `t2m_model`
- `MotionStreamer_vaebyh100_t2m_h100_20260204`
- `motionstreamer_model_causal_TAE_t2m_babel_272_h100_20260205_20260209`
- `MotionStreamer_8gpus_distributed`
- `MotionStreamer_8gpus_distributed_mp`
- `MotionStreamer_t2m_272_cached_embeddings_8gpu_bf16`
- `MotionStreamer_vae_causal_TAE_t2m_272_h100_20260203_t2m_h100_20260206`

### Miscellaneous

Destination: `Experiments/explorations/misc/`

- `.ipynb_checkpoints`

## Path Update Rules

Runnable exploration entrypoints under `explorations/` are updated as follows:

- Default checkpoint paths for the 35 moved runs receive their route-specific
  `Experiments/explorations/...` prefix.
- Exploration training shell scripts write new runs beneath the matching route
  directory rather than directly beneath `Experiments/`.
- Cross-route dependencies on official Causal TAE, MSA-VAE, or Global-RAG
  checkpoints keep their existing root paths.
- Historical scripts already documented as non-runnable remain historical;
  the archive does not invent replacements for missing programs.
- Official root training, evaluation, and inference entrypoints are unchanged
  except for comments that would otherwise direct users to a moved exploration
  checkpoint.

`explorations/README.md` is updated with the result-root convention. A tracked
post-move manifest is written to:

`docs/experiments/2026-07-25-exploration-results-archive.md`

The manifest records source, destination, route, byte size, file count, and
checkpoint filenames. It also contains the reverse mapping needed for manual
rollback.

## Move Safety

Before any move:

1. Verify all 35 source paths exist and are directories.
2. Verify all 35 destination paths do not exist.
3. Verify every source appears exactly once in the manifest.
4. Verify the source directories and `Experiments/` share the same filesystem
   device.
5. Record byte size, file count, and checkpoint filenames for every source.
6. Stop before mutation if any precondition fails.

The move is performed one directory at a time with explicit source and
destination paths. No wildcard is passed directly to a destructive or
overwriting command. Because all paths are on the same filesystem, each move
is a metadata rename rather than a 1.1 TB copy.

After each move:

1. Verify the source path is absent.
2. Verify the destination path is present and is a directory.
3. Recompute byte size and file count and compare them with the pre-move
   manifest.
4. Stop immediately on the first mismatch.

No destination is overwritten. No checkpoint content is rewritten.

## Validation

### Result layout

- All 35 sources are absent.
- All 35 destinations exist.
- Their aggregate byte size and file count match the pre-move manifest.
- No directory matching RF, MCA, latent-retrieval, local-RAG, Q-Former, CLIP
  baseline, SAE, TAE-GAN, or upstream MotionStreamer result patterns remains
  directly under `Experiments/`.
- Every Causal TAE and `MSA_VAE*` directory recorded before the move remains at
  the root.
- Every ordinary Global-RAG DDPM and formal ablation directory recorded before
  the move remains at the root.

### Code and documentation

- Search runnable entrypoints for stale references to the 35 old paths.
- Compile every modified Python entrypoint.
- Run `bash -n` on every modified shell launcher.
- Run the repository test suite.
- Run `git diff --check`.
- Verify `paper writing/Research-Paper-Writing-Skills` remains the only
  unrelated dirty path.

No training, evaluation, checkpoint loading, cache building, or dataset
mutation is part of this organization task.

## Rollback

Rollback uses the tracked manifest in reverse order:

1. Verify every old root path is absent.
2. Verify every archived destination exists.
3. Move each destination back to its recorded root path.
4. Restore the corresponding code-path commit.
5. Re-run size, file-count, and root-layout verification.

Rollback never overwrites an existing root directory.

## Expected Outcome

`Experiments/` continues to expose the complete Causal TAE, MSA-VAE,
Global-RAG DDPM, and formal ablation history at its root. Results for clearly
separate exploration methods live under one route-based
`Experiments/explorations/` directory, and runnable exploration scripts use
those locations for both existing checkpoints and future output.
