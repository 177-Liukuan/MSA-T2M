# MSA-T2M Exploration Code Archive Design

## Context

The repository root currently contains more than eighty tracked Python and
Shell entrypoints. Only a small subset belongs to the official MSA-T2M path.
The remainder records CLIP, cross-attention, Rectified Flow, Q-Former,
retrieval-baseline, representation-learning, upstream MotionStreamer, demo, and
diagnostic work.

The repository should present the paper method clearly without deleting the
research history. Archived exploration routes must remain runnable after the
move. Only root-level code and scripts are in scope; shared implementation
modules under `models/`, `humanml3d_272/`, `options/`, `utils/`, and
`visualization/` stay in their current locations.

## Goals

- Leave only official mainline and directly required helper entrypoints at the
  repository root.
- Group exploration entrypoints by research route under one `explorations/`
  tree.
- Preserve each archived route's ability to import shared repository modules
  and invoke its companion scripts.
- Document the status, negative result or intended use, dependencies, and new
  command for every archived route.
- Prevent exploration entrypoints from accumulating at the root again.

## Non-goals

- Refactoring or relocating shared model and dataset modules.
- Changing model behavior, hyperparameters, checkpoint formats, or experiment
  defaults.
- Deleting exploration code.
- Moving papers, PDFs, review material, datasets, checkpoints, generated
  artifacts, or external reference repositories.
- Making asset-dependent or GPU-dependent experiments runnable without their
  original data, checkpoints, and dependencies.

## Root-level Official Entry Points

The following code and scripts remain at the repository root.

### Causal TAE

- `train_causal_TAE.py`
- `TRAIN_causal_TAE.sh`
- `eval_causal_TAE.py`
- `EVAL_causal_TAE.sh`

### MSA-VAE

- `train_msa_vae.py`
- `TRAIN_msa_vae_phase1.sh`
- `TRAIN_msa_vae_phase2.sh`
- `eval_msa_vae.py`
- `EVAL_msa_vae.sh`

### Official preprocessing and cache construction

- `dataset_clip2t5.py`
- `get_text_latent_t5.py`
- `get_msa_latent.py`
- `build_msa_rag_cache.py`

`dataset_clip2t5.py` remains because it is the documented migration path from
the historical local-label files to the official T5 local features; its name
does not make the trained model a CLIP model.

### Global-RAG DDPM training, evaluation, and inference

- `train_t2m_rag.py`
- `TRAIN_t2m_rag.sh`
- `eval_msa_t2m_rag_t5.py`
- `EVAL_t2m_rag_t5.sh`
- `msa_gen_motion.py`
- `output_vis.py`

All non-code root files and all shared source directories remain in place.

## Archive Layout and File Mapping

The archive is a Python package so archived Python entrypoints can be invoked
with `python -m` from the repository root.

```text
explorations/
├── README.md
├── __init__.py
├── ablations/
│   └── no_rag/
├── clip/
├── cross_attention/
│   ├── local_rag/
│   ├── mca/
│   └── latent_retrieval/
├── rectified_flow/
├── qformer/
├── motionstreamer_baselines/
├── representation_experiments/
├── retrieval_baselines/
├── demos_and_diagnostics/
└── project_history/
```

Every Python-containing directory receives `__init__.py`.

### `explorations/ablations/no_rag/`

- `demo_msa_t2m_no_rag_t5.py`
- `DEMO_msa_t2m_no_rag_t5.sh`
- `eval_msa_t2m_no_rag_t5.py`
- `EVAL_t2m_no_rag_t5.sh`
- `TRAIN_t2m_no_rag.sh`

These are retained as mainline ablation records, but are not official
full-method entrypoints.

### `explorations/clip/`

- `demo_msa_t2m_clip.py`
- `eval_t2m_clip_baseline.py`
- `EVAL_t2m_clip_baseline.sh`
- `eval_t2m_rag.py`
- `EVAL_t2m_rag.sh`
- `get_text_latent_clip.py`
- `train_t2m_baseline_clip.py`
- `TRAIN_t2m_baseline_clip.sh`

This directory contains early 512-dimensional CLIP conditioning and the early
CLIP-dimension global-RAG evaluator.

### `explorations/cross_attention/local_rag/`

- `eval_msa_t2m_rag_local.py`
- `EVAL_t2m_rag_local.sh`
- `msa_gen_motion_local.py`
- `train_t2m_rag_local.py`
- `TRAIN_t2m_rag_local.sh`
- `TRAIN_THEN_EVAL_t2m_rag_local.sh`

### `explorations/cross_attention/mca/`

- `eval_msa_t2m_rag_mca.py`
- `EVAL_t2m_rag_mca.sh`
- `get_text_token_latent_t5.py`
- `msa_gen_motion_mca.py`
- `msa_gen_motion_mca_op.py`
- `train_t2m_rag_multi_text_token.py`
- `Train_t2m_rag_multi_text_token.sh`

### `explorations/cross_attention/latent_retrieval/`

- `build_latent_retr_library.py`
- `eval_msa_t2m_rag_latent_retr.py`
- `EVAL_t2m_rag_latent_retr.sh`
- `eval_msa_t2m_rag_latent_retr_addcfg.py`
- `EVAL_t2m_rag_latent_retr_addcfg.sh`
- `precompute_latent_retr_lookup.py`
- `train_t2m_rag_latent_retr.py`
- `Train_t2m_rag_latent_retr.sh`

The cross-attention implementation modules remain under `models/` and the
specialized datasets remain under `humanml3d_272/`.

### `explorations/rectified_flow/`

- `eval_msa_t2m_rag_t5_rf.py`
- `EVAL_t2m_rag_t5_rf.sh`
- `Train_t2m_rag_rf.sh`

The shared `models/diffloss.py` remains in place because the official model
imports its compatibility wrapper even when configured for DDPM.

### `explorations/qformer/`

- `build_rag_db.py`
- `PREPARE_text_embeddings.sh`
- `train_qformer_rag.py`
- `TRAIN_qformer_rag.sh`

`PREPARE_text_embeddings.sh` is kept as a historical companion script and is
documented as broken because it calls the absent `prepare_text_embeddings.py`.
Archiving must not misrepresent it as a verified preprocessing path.

### `explorations/motionstreamer_baselines/`

- `demo_t2m.py`
- `eval_t2m.py`
- `EVAL_t2m.sh`
- `get_latent.py`
- `motionstreamer_gen_motion.py`
- `train_motionstreamer.py`
- `TRAIN_motionstreamer.sh`
- `train_t2m.py`
- `TRAIN_t2m.sh`
- `Train_t2m_multi.sh`
- `train_t2m_cached.py`
- `TRAIN_t2m_cached.sh`
- `TRAIN_evaluator_272.sh`

These preserve upstream MotionStreamer and cached-training baselines used for
comparison and debugging.

### `explorations/representation_experiments/`

- `demo_msa_vae_sample.py`
- `eval_sae_v1.py`
- `EVAL_sae_v1.sh`
- `train_sae_v1.py`
- `TRAIN_sae_v1.sh`
- `train_tae_gan_v1.py`
- `TRAIN_tae_gan_v1.sh`
- `EVAL_tae_gan_v1.sh`
- `TRAIN_msa_vae.sh`
- `TRAIN_msa_vae_multi.sh`

This contains the one-shot MSA-VAE entrypoint and alternate SAE/GAN
representation-learning attempts.

### `explorations/retrieval_baselines/`

- `demo_retrieval.py`
- `RAG2Motion.py`
- `remodiffuse_gen_motion.py`

### `explorations/demos_and_diagnostics/`

- `demo_msa_t2m_t5.py`
- `demo_msa_t2m_t5_02.py`
- `demo_verify_dataset.py`
- `demo_verify_t5_conversion.py`
- `generate_motion.py`
- `inspect_latent_shapes.py`
- `msa_gen_motion_batch.py`
- `render_smpl_aitviewer_pos.py`
- `render_smpl_aitviewer_rot.py`
- `representation_272_to_bvh.py`
- `smoke_test.py`
- `verify_setup.py`
- `visualize_t2m_generation.py`

### `explorations/project_history/`

- `IMPLEMENTATION_SUMMARY.py`
- `WORKFLOW_GUIDE.py`
- `run.sh`
- `run_training.sh`
- `sedbash`
- `TRAIN_msa_vae_phase1.sh.bak`
- `TRAIN_msa_vae_phase2.sh.bak`

The two `run*.sh` files call the absent `train_t2m_msa.py`; `sedbash` is empty.
They are preserved and explicitly marked non-runnable historical artifacts
rather than silently repaired into a different experiment.

## Invocation and Import Rules

Archived Python entrypoints are run from the repository root:

```bash
python -m explorations.clip.train_t2m_baseline_clip --help
python -m explorations.cross_attention.local_rag.train_t2m_rag_local --help
```

This keeps the repository root on `sys.path`, so imports from `models`,
`humanml3d_272`, `options`, `utils`, and `visualization` continue to work.
Imports between files moved into the same archive route become explicit package
imports, for example:

```python
from explorations.qformer.train_qformer_rag import TAEFeatureExtractor
```

Archived shell scripts use this preamble:

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
```

The number of `..` components is set from each shell's actual archive depth.
Python targets moved into the archive are invoked with `python -m`; official
root targets use their root path. Cross-shell calls use an absolute path rooted
at `REPO_ROOT`. Existing environment variables, GPU counts, arguments, and
experiment defaults remain unchanged.

No compatibility wrapper or symlink remains at the old root path. The archive
index provides the old-to-new command mapping.

## Documentation

`explorations/README.md` contains:

- the official-versus-exploration warning;
- one table per route with status (`ABLATION`, `EXPERIMENTAL`,
  `NEGATIVE RESULT`, `BASELINE`, or `HISTORY`);
- old root path, new module or shell command, known result, and required
  external assets;
- the rule that archived scripts are launched from the repository root;
- explicit warnings for scripts that were already broken before the move.

The root README code-lineage tables link to the archive index and replace old
root paths with the new locations. `AGENTS.md` states that official work should
not modify `explorations/` unless an exploration or ablation is explicitly in
scope.

## Safety and Validation

### Structure manifest

A checked-in test defines the exact allowed root-level Python/Shell entrypoint
set listed in this design. It fails when:

- an official entrypoint is missing;
- an unclassified Python or Shell entrypoint appears at the root;
- one source file is assigned to multiple archive destinations;
- an archived file listed in the mapping is missing.

### Python validation

- Compile every moved `.py` file.
- Import each archived module that is safe to import without executing work.
- Run `python -m ... --help` only for modules whose argument parsing is guarded
  by `if __name__ == "__main__"` and does not initialize models, load data, or
  allocate CUDA before parsing.
- Validate explicit cross-archive imports against the new module names.

Scripts that execute work at import time are documented and compile-checked,
but are not imported by the smoke suite until separately refactored.

### Shell validation

- Run `bash -n` on every moved shell file.
- Use stub `python`, `accelerate`, and nested shell commands to verify that
  launchers change to the repository root and resolve the intended new target
  before attempting training or evaluation.
- Preserve arguments and environment-variable expansion through the move.

### Mainline regression

After all moves:

- run the existing 21 mainline acceleration tests;
- compile all official root Python entrypoints;
- run `bash -n` on all official root launchers;
- rerun the root-structure test;
- run `git diff --check`;
- verify the only unrelated dirty path remains
  `paper writing/Research-Paper-Writing-Skills`.

No training, evaluation, dataset mutation, checkpoint write, or cache rebuild is
part of this directory-only change.

## Expected Outcome

The repository root exposes one coherent reproduction path: Causal TAE,
MSA-VAE, T5 feature extraction, packed global-RAG DDPM training, official TMR
evaluation, and inference. Research history remains available and route-based
under `explorations/`, with executable commands and repaired path resolution,
without mixing abandoned methods into the mainline surface.
