# MSA-T2M Repository Guide

## Project overview

MSA-T2M is a text-to-motion research codebase derived from MotionStreamer. The
main research path has two stages:

1. `models/msa_vae.py` and `train_msa_vae.py` learn the multi-scale semantic
   alignment VAE (MSA-VAE).
2. `models/llama_rag_model.py` and the `train_t2m_rag*.py` entrypoints train
   retrieval-augmented autoregressive diffusion in the learned latent space.

The 272-dimensional HumanML3D data loaders live in `humanml3d_272/`. Shared
model code is in `models/`, command-line definitions are in `options/`, and
evaluation helpers are in `utils/` and `Evaluator_272/`. Uppercase `TRAIN_*.sh`
and `EVAL_*.sh` files are the preferred user-facing entrypoints. Manuscript
sources and research notes live under `paper writing/` and
`msa-t2m设计文档/`.

## Environment

Use the existing environment:

```bash
conda activate mgpt
```

The verified environment uses Python 3.8.11, PyTorch 2.4.1+cu118, and
Accelerate 1.0.1. Do not replace the environment or upgrade core dependencies
unless the task explicitly requires it.

Training assumes CUDA. Do not start a full training or benchmark run merely to
validate a code edit; these jobs are expensive and may require a separately
allocated multi-GPU machine.

## Authoritative workflows

- MSA-VAE training is progressive. Use `TRAIN_msa_vae_phase1.sh` followed by
  `TRAIN_msa_vae_phase2.sh`; `TRAIN_msa_vae.sh` and `train_msa_vae.py` contain
  the shared implementation.
- Extract MSA-VAE motion and retrieval latents with `get_msa_latent.py`.
- Train the global retrieval model with `TRAIN_t2m_rag.sh` /
  `train_t2m_rag.py`.
- Train the global-plus-local retrieval model with
  `TRAIN_t2m_rag_local.sh` / `train_t2m_rag_local.py`.
- Evaluate MSA-VAE reconstruction with `EVAL_msa_vae.sh`.
- Evaluate generation with `EVAL_t2m_rag_t5.sh` or
  `EVAL_t2m_rag_local.sh`, matching the checkpoint architecture.
- Use `TRAIN_THEN_EVAL_t2m_rag_local.sh` only when an end-to-end run was
  explicitly requested.

Treat shell scripts as experiment records as well as launchers. Keep training
and evaluation values consistent for checkpoint paths, latent directories,
text embedding dimension, retrieval `top-K`, local-token count, local latent
dimension, EMA selection, self-attention/cross-attention flags, and generative
head type. A checkpoint must not be evaluated with a structurally different
configuration.

## Change guidelines

- Inspect the relevant model, dataset, option parser, training entrypoint, and
  evaluation entrypoint before changing a shared tensor shape or argument.
- Reuse the existing option and shell-entrypoint patterns. If a new argument
  affects model construction, add it consistently to training, evaluation,
  inference, checkpoint metadata, and launch scripts.
- Preserve the causal latent convention, 272-D motion representation, and
  latent/text dimension contracts unless a task explicitly changes them.
- Do not add a second projection, normalization, or conditioning path without
  first checking whether the underlying LLaMA/MSA module already performs it.
- Do not hard-code machine-specific absolute paths, GPU IDs, usernames, or
  cluster job identifiers. Use command-line arguments or overridable
  environment variables.
- The working tree may contain active experiments and manuscript edits.
  Preserve unrelated user changes and never reset or rewrite them implicitly.
- Third-party projects are submodules. Do not edit, vendor, or commit their
  internal working-tree changes unless the task explicitly targets them.

## Data and artifact policy

Never commit datasets, body models, downloaded text encoders, generated
latents, retrieval caches, checkpoints, TensorBoard events, run logs, rendered
videos, or bulk demo output. In particular, keep `Experiments/`,
`humanml3d_272` data/latent directories, `sentencet5-xxl/`, `body_models/`,
and checkpoint files local.

Small source-controlled figures and manuscript PDFs are allowed. Before making
this repository public, separately audit manuscript/review material for
anonymity, licensing, and publication-policy constraints.

## Validation

Run checks proportional to the change:

```bash
conda run -n mgpt python -m py_compile <changed-python-files>
bash -n <changed-shell-files>
git diff --check
```

For model or dataset changes, add a minimal CPU/import/shape smoke check when
possible. For checkpoint-loading changes, verify the expected state-dict keys
and tensor shapes without launching training. Run a full training or TMR
evaluation only when explicitly requested, and report the exact command,
checkpoint, dataset split, GPU count, and output directory.

## Git remotes

- `origin`: the private MSA-T2M development repository.
- `upstream`: the original `zju3dv/MotionStreamer` repository.

Push project work only to `origin`. Never push local MSA-T2M changes to
`upstream`.
