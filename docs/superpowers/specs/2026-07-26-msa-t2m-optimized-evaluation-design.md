# MSA-T2M Protocol-Equivalent Optimized Evaluation Design

## Goal

Add a separate, optimized single-GPU MSA-T2M evaluation pipeline without
modifying any existing evaluation source file or launcher. The new pipeline
must preserve the existing evaluation protocol while batching the expensive
generation path.

Protocol equivalence means preserving the dataset split and batching protocol,
model checkpoints and architecture, FP32 execution, CFG scale, DDPM sampling
steps, retrieval configuration, EOS rule, MSA-VAE decoder, evaluator networks,
and metric definitions. Batched stochastic sampling may change individual
motions and a single run's metric values within the original pipeline's normal
random variation; bitwise identity is not required.

## Scope and Isolation

The implementation creates new files only:

- `EVAL_t2m_rag_t5_optimized.sh`: user-facing optimized launcher.
- `eval_msa_t2m_rag_t5_optimized.py`: model loading and optimized evaluation
  entrypoint.
- `utils/eval_msa_t2m_optimized.py`: batched generation state machine and
  protocol-equivalent evaluation loop.
- `tests/test_eval_msa_t2m_optimized.py`: CPU unit tests for batching, EOS, and
  ordering.

The following existing files remain byte-for-byte unchanged:

- `EVAL_t2m_rag_t5.sh`
- `eval_msa_t2m_rag_t5.py`
- `utils/eval_trans.py`
- `humanml3d_272/dataset_eval_t2m.py`
- all existing model implementations

The new entrypoint may import existing model, dataset, checkpoint, evaluator,
and metric implementations. It must not alter their behavior or defaults.

## Preserved Evaluation Protocol

The optimized pipeline preserves:

- HumanML3D-272 `test` or `val` selection through the existing dataset loader.
- Evaluation batch size 32, existing shuffle behavior, length sorting,
  stochastic caption/crop selection, and `drop_last=True`.
- Sentence-T5 XXL embeddings using the selected online model.
- Top-K dense RAG retrieval and weighted retrieval-token fusion.
- CFG scale 4.0 by default and the existing joint conditional/unconditional
  construction.
- The checkpoint's existing DDPM head with 50 denoising steps per generated
  latent token.
- Ground-truth motion length as the per-sample generation length ceiling.
- Reference-end-latent EOS detection and the configured L2 threshold.
- EOS exclusion: the token that triggers EOS is not appended to the motion
  latent sequence and is never decoded.
- The same MSA-VAE decoder and 272-dimensional normalized motion output.
- The same TEMOS evaluator checkpoint and text/motion encoder architectures.
- Batch-local R-Precision Top-1/2/3 negatives, matching score, FID, and
  diversity formulas from `utils/eval_trans.py`.
- FP32 model execution. BF16, FP16, TF32 policy changes, KV caching,
  `torch.compile`, and multi-GPU sharding are outside this design.

The output log must include `optimized_pipeline=true`, the effective batch
size, text source, CFG scale, stopping threshold, checkpoint paths, and the
same final metric names as the original pipeline. The optimized launcher uses
a distinct default experiment name so its logs cannot overwrite the original
run.

## Batched Generation Architecture

### Batch preparation

For each existing evaluation batch, the optimized evaluator:

1. Encodes all captions with one Sentence-T5 `encode` call.
2. Converts the resulting embeddings to a single FP32 CUDA tensor.
3. Performs one batched Top-K RAG lookup.
4. Computes each sample's maximum latent count as
   `max(1, floor(m_length / unit_length))`.

This removes 32 repeated text-encoder and retrieval invocations while
preserving their mathematical inputs and configuration.

### Active-set autoregressive sampling

All samples start active with an empty motion prefix. At latent step `s`, a
sample is active exactly when:

- it has not previously emitted EOS; and
- `s` is less than its own maximum latent count.

All active samples have generated the same number of accepted latent tokens,
so their prefixes have a common tensor shape and can be passed through
`LLaMARAGWrapper.sample_next_with_cfg` as one batch. Text embeddings and
retrieval tensors are indexed using the same active sample indices.

For every returned candidate token:

1. Compare it with the reference end latent when stopping is enabled.
2. If its distance is below the threshold, mark that sample finished and
   record its stop step without appending the candidate.
3. Otherwise append the candidate to that sample's latent prefix.
4. Remove finished and length-exhausted samples from the next active set.

Because an EOS token is excluded independently for every sample, a sample
that finishes early cannot receive latent tokens generated for other batch
members.

### Empty-motion compatibility

The current serial sampler returns a single zero latent if the first candidate
is EOS and no motion token was accepted. The optimized pipeline preserves this
compatibility behavior, returning one zero latent for that sample. The log
records the number of such fallbacks.

### Batched decoding

Generated latent sequences have variable lengths. The evaluator groups samples
by accepted latent length, stacks each equal-length group, and invokes the
existing MSA-VAE decoder once per group. It then restores decoded motions to
the original evaluation-batch order, crops them to the existing 300-frame
buffer, and records the same predicted lengths used by the evaluator.

Grouping only identical lengths avoids padding context that could change the
causal decoder output.

## Metric Evaluation

The evaluator text and motion encoders continue to run once per original
32-sample evaluation batch. R-Precision and matching score are calculated
within that unchanged batch, so the negative pool and normalization remain
identical to the existing protocol.

Ground-truth and generated motion embeddings are accumulated in original
batch order. FID and diversity are calculated after all batches using the
existing functions. No new metric, repeat count, confidence interval, or
sample inclusion rule is introduced.

## Error Handling

The new entrypoint must fail before evaluation when:

- a checkpoint, evaluator checkpoint, T5 model, retrieval directory, empty
  text embedding, or reference end latent cannot be resolved;
- text, retrieval, or EOS latent dimensions do not match configured model
  dimensions;
- a required RAG checkpoint component is missing;
- the generated batch loses sample ordering or returns a different number of
  samples than its input.

Checkpoint loading follows the current official EMA-selection behavior. The
optimized pipeline does not broaden compatibility by silently changing model
structure or sampling-head type.

## Testing

CPU tests use small deterministic stub components and must cover:

1. A batch with no EOS produces the same per-sample latent sequences as a
   serial reference implementation.
2. The candidate that triggers EOS is excluded from the decoded sequence.
3. Different samples can stop at different steps without receiving tokens
   after their own stop.
4. Per-sample ground-truth length ceilings are enforced.
5. The first-token EOS case returns the documented single-zero-latent
   compatibility fallback.
6. Equal-length decode grouping restores the original sample order.
7. Batched metric accumulation retains original 32-sample batch boundaries.
8. Existing official evaluation files have no diff after implementation.

Repository validation additionally runs Python compilation for all new Python
files, shell syntax validation for the new launcher, focused unit tests, and
`git diff --check`. Full HumanML3D generation evaluation is not run as part of
code validation because it is a multi-hour GPU workload.

## Success Criteria

- No existing evaluation file or script is modified.
- The new launcher constructs the same official MSA-T2M model and evaluator
  configuration.
- CPU tests prove per-sample EOS, maximum-length, and output-order behavior.
- Text encoding and RAG lookup occur once per evaluation batch rather than once
  per sample.
- Autoregressive Transformer and DDPM sampling operate on the current active
  sample set rather than serial batch-size-one calls.
- Metric batch boundaries and formulas are unchanged.
- The new pipeline completes static and focused CPU validation in the existing
  `mgpt` environment.
