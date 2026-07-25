# MSA-T2M EOS Decode Fix

## Problem

MSA-T2M represents EOS as a continuous reference latent. The official
evaluation sampler and the single-motion inference sampler currently append a
generated token to the motion latent sequence before testing whether it is
EOS. When EOS is generated, MSA-VAE therefore decodes that control token as one
latent step, producing four non-motion frames.

## Required behavior

For every newly generated token:

1. Compare it with the reference EOS latent.
2. If it satisfies the stopping threshold, record that loop position by
   terminating generation without appending the token.
3. Otherwise append it to the motion latent prefix.
4. Decode only the retained motion latents.

The sampler return types remain unchanged. The stopping position and EOS token
are internal control information and are not returned to the decoder.

## Scope

Apply the same stop-before-append ordering to:

- `RAGEvalSampler.sample_for_eval_CFG` in `eval_msa_t2m_rag_t5.py`
- `sample_motion_latents_with_stop` in `msa_gen_motion.py`

Do not change training targets, latent extraction, thresholds, archived
explorations, model architecture, or checkpoint formats.

## Verification

Add CPU-only regression tests with deterministic fake samplers that generate
one motion token followed by EOS. Each public sampling path must return only
the motion token. The tests must fail on the pre-fix code and pass after the
minimal ordering change. Also run Python compilation and `git diff --check`.
