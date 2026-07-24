# MSA-T2M Mainline Training Pipeline Acceleration Design

## Context

The official generation path is:

```text
TRAIN_t2m_rag.sh
└── train_t2m_rag.py
    ├── humanml3d_272/dataset_msa_rag.py
    ├── models/llama_rag_model.py
    └── models/llama_model.py
```

It trains the paper's global `[CLS]` single-token RAG method with the DDPM
autoregressive head. The current path already uses offline T5 features, BF16,
four persistent DataLoader workers, pinned host memory, and a complete
`no_grad` first pass in the two-forward training algorithm.

Two bottlenecks and one correctness risk remain:

1. Every sampled item reloads its motion and caption arrays from many small
   `.npy` files.
2. Every sampled caption repeats a full CPU cosine search over the same
   `[CLS]` library, although both query captions and the library are static.
3. Multi-GPU training calls `rag_model.module` directly for the gradient-bearing
   forward. This bypasses the DDP wrapper and does not provide a trustworthy
   gradient-reduction boundary.

The accepted equivalence standard permits minor floating-point rounding and
sample-order differences. Retrieval semantics, training objectives, checkpoint
structure, and final evaluation metrics must remain equivalent within normal
statistical variation.

## Goals

- Remove repeated small-file reads and repeated full-library retrieval from
  the training critical path.
- Ensure all gradient-bearing multi-GPU work executes inside one DDP forward.
- Retain the current paper method and checkpoint/evaluation compatibility.
- Make stale or structurally incompatible caches fail safely.
- Keep a non-cached reference path for equivalence checks and rollback.

## Non-goals

- Changing MSA-VAE, DDPM, CFG, EMA, the RAG token definition, or evaluation.
- Adding local retrieval, cross-attention, Rectified Flow, or other abandoned
  research routes to the official path.
- Guaranteeing bitwise-identical training across devices or process counts.
- Running a complete 100K-step training or final TMR evaluation as part of this
  code change.

## Chosen Approach

Use a persistent, self-validating global RAG cache built once before
distributed training. The cache packs source arrays into a small number of
contiguous files and stores per-caption retrieval results. A manifest binds the
cache to the source feature set and retrieval configuration.

The launcher builds or validates the cache in one process before
`accelerate launch`. DataLoader workers then perform only caption sampling,
array indexing, and variable-length motion padding.

This is preferred over rebuilding an in-memory lookup at every launch because
several experiments will reuse the same MSA-VAE and text features. It is
preferred over online GPU retrieval because it does not consume training VRAM
or introduce a new multi-GPU retrieval path.

## Cache Format and Validity

The default cache directory is derived from the MSA-VAE experiment name and is
overridable from `TRAIN_t2m_rag.sh`. It is a generated artifact and must remain
ignored by Git.

The cache contains:

- ordered sample IDs;
- packed float32 motion latents plus per-sample offsets and lengths;
- packed float32 caption embeddings plus per-sample offsets and caption counts;
- the float32 `[CLS]` retrieval library;
- per-caption Top-K library indices;
- per-caption Top-K cosine scores;
- a JSON manifest.

The manifest records:

- cache schema version;
- canonical source directories;
- ordered source sample IDs;
- source file size and nanosecond modification time;
- text dimension;
- requested maximum Top-K;
- self-exclusion setting;
- array shapes and dtypes.

Cache construction writes into a temporary sibling directory and publishes the
completed cache through an atomic rename. An incomplete cache is never accepted.
If the manifest or array validation fails, training stops with a command that
rebuilds the cache; it must not silently mix stale features with a new
experiment.

Top-K retrieval uses the same float32 normalization, cosine similarity,
self-exclusion, descending Top-K selection, and stored similarity values as the
reference dataset. A cache built with maximum `K` may serve any training run
whose requested `K` is no greater than that value.

## Dataset Data Flow

`Text2MotionMSARAGDataset` supports two explicit modes:

- `reference`: current small-file loading and online retrieval, retained for
  regression comparison and emergency rollback;
- `packed`: memory-mapped packed arrays and precomputed retrieval, used by the
  official launcher.

In both modes, `__getitem__`:

1. chooses one caption uniformly with the existing worker-local Python RNG;
2. obtains that caption's T5 embedding;
3. obtains Top-K `[CLS]` vectors and cosine scores using the same sample
   self-exclusion rule;
4. returns the same four logical values and dtypes as before.

The collate function and padding convention remain unchanged. DataLoader keeps
four workers by default, `persistent_workers=True`, and `pin_memory=True`.
Training transfers these tensors with `non_blocking=True`.

## DDP-safe Two-forward Training

A focused `nn.Module` training wrapper owns the existing
`LLaMARAGWrapper` and computes the complete two-forward DDPM loss in its
`forward` method:

1. execute the first Transformer and diffusion prediction under `no_grad`;
2. detach `pred_xstart`;
3. replace the scheduled subset of input latents exactly as before;
4. execute the second Transformer and diffusion loss with gradients;
5. return the scalar loss.

Accelerate/DDP wraps this training module and the training loop calls the
wrapped module exactly once per iteration. The first and second passes therefore
share one valid DDP forward boundary, while the first pass still creates no
autograd graph.

The wrapped research model remains accessible as a child module for loading,
EMA, and checkpoint serialization. Checkpoints continue to expose the existing
top-level fields:

```text
trans
rag
trans_ema
rag_ema
scheduler
optimizer
iter
generative_head_type
use_ema
ema_decay
```

The `trans`, `rag`, and EMA state dictionaries retain their existing parameter
names and tensor shapes. Evaluation continues to instantiate
`LLaMARAGWrapper` directly and therefore requires no model change.

## Minor Synchronization Reductions

The training loop will:

- use `non_blocking=True` for tensors originating in pinned DataLoader memory;
- convert loss to a Python scalar once per iteration and reuse it for
  bookkeeping;
- preserve all current random draws and optimizer/scheduler ordering.

Attention scaling and other shared model internals are outside this change.
They are left untouched to avoid checkpoint or inference drift.

## Error Handling

- Missing source directories or an empty valid sample intersection fail before
  cache construction.
- Mismatched motion/text/`[CLS]` dimensions report the sample ID and expected
  shape.
- A stale, incomplete, wrong-schema, or insufficient-Top-K cache is rejected.
- The launcher prints the selected cache, source directories, Top-K, worker
  count, and whether it is building, validating, or using the cache.
- The reference mode remains selectable through an explicit launcher variable;
  packed mode never silently falls back to online retrieval.

## Verification

### Dataset equivalence

Synthetic fixtures and a sampled real-data check compare reference and packed
modes:

- identical ordered sample IDs;
- identical selected caption under controlled RNG;
- exact Top-K indices;
- cosine scores within float32 tolerance;
- identical text, retrieved `[CLS]`, and motion tensors;
- identical collated shapes, padding, and dtypes.

The cache invalidation tests modify a source file and verify that validation
rejects the cache.

### Loss and gradient equivalence

On a small deterministic model and batch, compare the existing two-forward
calculation with the training module using the same random state:

- loss within floating-point tolerance;
- all trainable parameter gradients within tolerance;
- no gradient from the first pass.

### Distributed correctness

A two-process CPU/Gloo smoke test performs one optimizer step with different
rank-local inputs and verifies that both ranks end with identical parameters.
This test would fail if the gradient-bearing forward bypassed DDP.

### Compatibility and static checks

- Load a representative existing checkpoint into the refactored training path.
- Save a new checkpoint and compare required keys, parameter names, and shapes.
- Run targeted tests, Python compilation, shell syntax checking, and
  `git diff --check`.

### Performance evidence

Measure separately:

- reference versus packed `__getitem__` throughput;
- DataLoader batch wait time;
- a short training-step throughput probe when an allocated GPU is available.

No full training job is started automatically. Final paper metrics still
require the planned controlled training and TMR evaluation runs.

## Expected Outcome

After the one-time cache build, training no longer performs repeated NFS
small-file reads or full-library CPU retrieval. GPU feeding should therefore
match the accelerated pipeline discovered during the cross-attention
experiments, while the official global-RAG method, DDPM objective, checkpoint
contract, and evaluator remain unchanged. Multi-GPU training also gains a
well-defined DDP synchronization boundary instead of relying on the legacy
unwrapped forward path.
