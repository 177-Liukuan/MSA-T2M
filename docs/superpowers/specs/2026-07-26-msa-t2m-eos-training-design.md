# MSA-T2M Consistent EOS Training Design

## Problem

The Stage-2 RAG collate function dynamically pads each batch but does not
return the pre-padding sequence lengths. Training infers lengths from non-zero
latent values and then removes the final batch column with
`m_tokens[:, :-1]`. Consequently, the longest sample in each batch loses its
EOS target while shorter samples can retain EOS before padding. EOS
supervision therefore depends on batch composition.

The numeric length inference is also structurally unsafe: a valid continuous
latent is not required to contain a non-zero component.

## Data contract

Every motion latent file produced by `get_msa_latent.py` contains all motion
latents followed by one continuous EOS latent. The RAG dataset returns this
variable-length sequence unchanged.

`collate_fn` must:

1. Compute every sequence length before padding.
2. Right-pad to the longest length in the batch.
3. Return the padded tensor and an integer length tensor.

The returned lengths include EOS and are authoritative in both reference and
packed-cache modes.

## Training contract

The training loop consumes the explicit lengths and passes the complete
padded motion tensor to `RAGTwoForwardLoss`. It must not infer lengths from
latent values, globally remove the last column, or silently clamp invalid
lengths.

The loss validates that every length is positive and no greater than the
padded width. Its per-sample valid mask includes EOS and excludes right
padding.

## Autoregressive alignment

For global RAG, the causal input layout is:

```text
[text, retrieval, motion_0, ..., motion_(T-2), EOS]
```

`motion_condition_slice` begins at `num_condition_tokens - 1`, so the
retrieval position predicts `motion_0`, each motion position predicts the next
motion token, and `motion_(T-2)` predicts EOS. The hidden state produced from
the final EOS input is excluded from the target-aligned slice.

The full transformer sequence must not exceed `block_size`. With one text
token, one global RAG token, and `block_size=78`, at most 76 motion/EOS tokens
are valid. The RAG wrapper must raise a clear error instead of silently
truncating when this bound is exceeded.

## Two-forward replacement

The pseudo-target replacement schedule must use the same per-sample valid
mask as the final loss. Each sample receives the scheduled replacement count
based on its own valid length, and padding positions are never replaced.

The no-gradient pseudo-target pass may still compute predictions for padded
positions because right padding cannot influence earlier valid positions
under causal attention; those predictions are neither inserted nor included
in the gradient-bearing loss.

## Compatibility

The change does not alter latent files, model parameters, state-dict names,
checkpoint formats, inference interfaces, or retrieval features. Existing
checkpoints remain loadable, but a clean Stage-2 retraining is recommended
because previous checkpoints learned batch-dependent EOS supervision.

## Verification

CPU regression tests must prove:

- collate returns exact pre-padding lengths for reference and packed data;
- a valid all-zero latent token remains counted;
- loss masks include EOS and exclude padding;
- pseudo-target replacement never changes padding and uses each sample's
  valid length;
- the RAG wrapper accepts 76 motion tokens with two condition tokens and
  rejects 77;
- existing loss/gradient, cache, DDP, checkpoint, and EOS inference tests
  continue to pass.
