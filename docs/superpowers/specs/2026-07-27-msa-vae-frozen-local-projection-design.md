# MSA-VAE Frozen Local Projection Phase-2 Design

## Objective

Add an explicit Phase-2 training option that freezes the `local_proj`
parameters learned in Phase 1 while preserving gradient flow through that
fixed projection into the CNN encoder. This creates a fixed local semantic
coordinate system for reviewer-facing alignment--realism experiments without
silently changing the existing official, BABEL, or historical pilot
protocols.

The intended local-alignment path is:

```text
L_local
  -> frozen local_proj
  -> mu
  -> CNN encoder
```

Freezing `local_proj` prevents the projection from absorbing additional local
alignment optimization. It does not detach its input and does not place the
forward pass under `torch.no_grad()`, so `L_local` continues to update the CNN
encoder. The local loss still does not directly update the CNN decoder; the
decoder adapts only through reconstruction and root losses.

## Scope

The shared option parser, Phase-2 trainability setup, optimizer construction,
checkpoint metadata, HumanML Phase-2 launcher, focused tests, and concise
training documentation are in scope.

The option is opt-in. Existing commands retain their current behavior unless
the new flag or launcher environment variable is supplied. In particular:

- `TRAIN_msa_vae_phase1.sh` is unchanged;
- the default `TRAIN_msa_vae_phase2.sh` behavior remains fully trainable;
- BABEL Phase 2 remains unchanged;
- the completed 2026-07-26 alignment pilot remains reproducible;
- new reviewer experiments explicitly enable the frozen-head protocol for
  every compared variant.

Choosing the new Phase-2 local-weight sweep, preparing held-out local targets,
logging per-loss gradient diagnostics, and launching expensive training are
separate follow-up tasks. This implementation provides the trainability
contract they require but does not silently choose or run that matrix.

## User Interface

Add the training argument:

```text
--freeze-phase2-local-proj
```

The parser stores it as `args.freeze_phase2_local_proj`. Its default is
`False`. Supplying it with Phase 0 or Phase 1 is rejected because the name and
scientific meaning are Phase-2-specific.

`TRAIN_msa_vae_phase2.sh` accepts:

```text
FREEZE_PHASE2_LOCAL_PROJ=0|1
```

The default is `0`, preserving existing launch behavior. Value `1` appends
`--freeze-phase2-local-proj`; any other value fails before Accelerate is
launched. New reviewer experiment launchers must set the environment variable
to `1` explicitly for all variants rather than relying on inherited shell
state.

The launcher prints the resolved projection policy alongside the other
experiment settings.

## Phase-Aware Trainability

Trainability is configured only after the complete Phase-1 checkpoint has
been loaded. A focused helper applies the following state:

| Module | Phase 2 with frozen-head option |
| --- | --- |
| CNN encoder | trainable |
| CNN decoder | trainable |
| `decode_proj` | trainable |
| `local_proj` | frozen |
| Transformer encoder | trainable |
| Transformer decoder | trainable |
| `global_proj` | trainable when it has parameters |

The helper first restores the expected trainable state for every MSA-VAE
parameter, then applies the phase-specific freeze. This avoids depending on
ambient `requires_grad` values.

Phase 1 continues to freeze the CNN encoder, CNN decoder, and `decode_proj`
while training the semantic modules and projection heads. Phase 0 retains
uniform full-model training.

For the current T5 configuration, `global_proj` is `nn.Identity` because the
Transformer width and text width are both 768, so it has no parameters.
Nothing special is required for it.

## Optimizer Contract

Phase 2 retains differential learning rates:

- CNN encoder, CNN decoder, and `decode_proj` use
  `lr * cnn_lr_scale`;
- every other trainable parameter uses `lr`.

Both parameter groups contain only parameters whose `requires_grad` is
`True`. Consequently, frozen `local_proj` parameters are absent from the
optimizer rather than merely receiving `None` gradients inside it.

The forward implementation in `models/msa_vae.py` remains unchanged:

```python
clip_local_feat = self.local_proj(mu)
```

This ordinary autograd operation is essential. No detach, copied tensor,
custom backward, or second local projection is introduced.

## Artifact Contract

`freeze_phase2_local_proj` is added to the checkpoint `training_args`
identity. New Phase-1 checkpoints record `False`; enabled Phase-2 checkpoints
record `True`. Phase 2 may legitimately resume a Phase-1 parent whose value is
`False`, because the freeze begins only after loading the parent.

The setting does not alter tensor names or tensor shapes. Existing
checkpoints remain loadable, and evaluation model construction is unchanged.
Future reviewer-experiment collectors can require `True` without changing the
historical pilot collector.

## Failure Handling

- Reject the CLI flag outside Phase 2.
- Reject launcher values other than literal `0` or `1`.
- Fail if a requested model does not expose the expected MSA-VAE modules,
  instead of silently freezing nothing.
- Keep at least one trainable parameter in each required Phase-2 optimizer
  group; report a clear configuration error otherwise.
- Log the module-level trainable/frozen policy before optimizer creation.

## Testing

Implementation follows test-driven development.

Focused unit tests use a small real `nn.Module` with the same named module
boundaries as `MSA_HumanVAE` and verify:

1. Phase 2 freezes only `local_proj` when the option is enabled.
2. CNN encoder/decoder, `decode_proj`, Transformer encoder/decoder, and a
   parameterized `global_proj` remain trainable.
3. Backpropagating a local alignment loss leaves
   `local_proj.weight.grad is None` while producing a finite, nonzero CNN
   encoder gradient.
4. An optimizer step changes the CNN encoder but not the frozen local
   projection.
5. Frozen parameters are absent from Phase-2 optimizer groups.
6. Phase 1 and legacy Phase 0 retain their existing trainability behavior.
7. The option parser defaults to disabled, accepts it for Phase 2, and rejects
   it for other phases.
8. Checkpoint metadata records the resolved boolean.
9. The Phase-2 shell launcher defaults to the legacy policy, forwards the
   explicit enabled policy, and rejects malformed values.

Verification uses:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_full_sequence_launchers
conda run -n mgpt python -m py_compile \
  options/option_msa_vae.py \
  utils/msa_vae_training.py \
  train_msa_vae.py
bash -n TRAIN_msa_vae_phase2.sh
git diff --check
```

A minimal CPU gradient smoke test is included in the unit suite. No full
training, benchmark, or GPU allocation is used to validate this change.

## Completion Criteria

The implementation is complete when:

- the opt-in argument and launcher environment variable are available;
- enabled Phase 2 loads the Phase-1 projection and then freezes it;
- local alignment demonstrably backpropagates through the fixed projection
  into the CNN encoder;
- all other requested modules remain trainable;
- the optimizer excludes the projection parameters;
- checkpoint metadata makes the protocol auditable;
- existing default HumanML, BABEL, and historical pilot behavior is
  unchanged;
- focused regression, compile, shell syntax, and diff checks pass.
