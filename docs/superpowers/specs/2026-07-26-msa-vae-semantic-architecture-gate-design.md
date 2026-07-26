# MSA-VAE Semantic Architecture Gate Experiment Design

## Objective

Decide whether the formal MSA-T2M model needs the complete semantic
autoencoder, only its Transformer encoder, or neither Transformer component.
The experiment compares three MSA-VAE semantic architectures:

1. `full_ae`: the current Transformer encoder and Transformer decoder;
2. `encoder_only`: the same Transformer encoder without a semantic decoder;
3. `direct_pool`: mask-aware mean pooling of deterministic local latents,
   followed by one linear projection into the 768-D text space.

The decision uses latent reconstruction, motion reconstruction, semantic
retrieval, model complexity, and paired downstream RAG gain. All three
MSA-VAE variants are trained, but diffusion is trained only for Full AE and
the best simplified representative.

This is a single-training-seed structure gate. Every training run uses seed
123. Results must not be presented as a multi-seed mean or with a
training-seed standard deviation.

## Repository Scope

The implementation lives on branch `exp/semantic-architecture-gate`.

The shared MSA-VAE model, option, checkpoint, evaluation, and extraction
paths receive the smallest changes needed to make semantic architecture an
explicit artifact contract. The official root launchers keep their existing
default behavior. Experiment-specific entrypoints, probes, collectors, and
documentation live in:

```text
explorations/semantic_architecture_gate/
```

Experiment outputs live below:

```text
Experiments/explorations/semantic_architecture_gate_s123/
```

Extracted latents, retrieval features, and RAG caches remain ignored local
artifacts under their existing `humanml3d_272/` roots. No dataset,
checkpoint, latent array, cache, event file, or run log is committed.

The archived third-party projects already present under `explorations/` are
out of scope and must not be modified.

## Architecture Contract

Add one model-construction option:

```text
--semantic-architecture {full_ae,encoder_only,direct_pool}
```

The default is `full_ae`, preserving the current formal architecture and old
command behavior.

### Full AE

`full_ae` preserves the current semantic path:

```text
mu tokens
  -> 6-layer Transformer encoder with [CLS]
  -> 768-D h_cls
  -> 6-layer Transformer decoder
  -> reconstructed mu tokens
```

It uses the native valid-token latent reconstruction loss, global alignment,
and local alignment. The experiment launchers set
`--latent_recon_weight 1.0`.

### Encoder-only

`encoder_only` instantiates the same Transformer encoder, including the same
input projection, positional encoding, `[CLS]` token, layer count, head
count, feed-forward size, and dropout as Full AE. It does not instantiate a
semantic Transformer decoder and has no native latent reconstruction loss.

The experiment launchers set `--latent_recon_weight 0`. Model/training
validation rejects a positive latent reconstruction weight for this
architecture instead of silently ignoring it.

### Direct Pooling

`direct_pool` does not instantiate a semantic Transformer encoder or
decoder. It computes:

```text
pooled_mu = sum(valid_mu_tokens) / number_of_valid_tokens
h_cls = Linear(16, 768)(pooled_mu)
```

Padding is excluded using the same downsampled length mask as the other
architectures. A sample with no valid latent token is rejected. The pooling
path has no MLP, attention, learned token, extra normalization, or second
projection.

It retains the existing local projection used by local alignment. Its
dedicated global linear projection is the only operation after pooling. The
returned `h_cls` is 768-D so retrieval extraction and downstream RAG consume
the same interface for all architectures.

The experiment launchers set `--latent_recon_weight 0`, and configuration
validation rejects a positive value.

### Common Physical Path

All variants retain the same 272-D motion representation, Causal CNN
encoder, posterior `mu`/`logvar`, reparameterized `z_local`, decode
projection, and Causal CNN decoder. They share the causal latent convention
and 16-D local latent contract.

Phase 1 freezes the physical path. Phase 2 unfreezes it with the same
differential learning rate in all variants. Motion reconstruction after
Phase 2 can therefore expose changes caused by the different semantic
training signals.

## Fixed MSA-VAE Training Protocol

Each architecture runs independently on its own machine. A run may use two
or eight RTX 4090 GPUs, selected through the script argument. GPU count
changes wall-clock time but not the configured total batch size.

Every architecture uses:

- HumanML3D 272-D and the complete HumanML training split;
- `--no_ft_split`;
- offline Sentence-T5-XXL global text features with dimension 768;
- the same sparse local targets and masks;
- seed 123;
- validation seed 123 and validation batch size 32;
- the same fixed Causal TAE checkpoint and its verified SHA-256;
- Transformer settings 768 model width, 8 heads, 6 layers, 2048
  feed-forward width, and 0.1 dropout where a Transformer is present;
- local latent dimension 16;
- Phase 1 full-sequence training for 50,000 iterations;
- Phase 2 mixed full-sequence/window-replay training for 50,000 iterations;
- the root launchers' learning rates, batch-size policy, scheduler, replay
  interval, and remaining optimization settings;
- Phase 1 global/local weights 0.5/0.2;
- Phase 2 global/local weights 0.1/0.001;
- deterministic complete validation at the root launchers' intervals;
- `net_last.pth` as the fixed cross-architecture comparison checkpoint.

Each Phase 2 run must resume its own architecture-matched Phase 1
`net_last.pth`. A checkpoint from another architecture is rejected even if
some tensor shapes happen to match.

## Common Latent Reconstruction Probe

The native semantic decoder cannot define a comparable metric for
decoder-free variants. A separate diagnostic decoder probe therefore
measures how much local-latent information is recoverable from each frozen
768-D `h_cls`.

For every trained MSA-VAE checkpoint:

1. freeze every MSA-VAE parameter;
2. compute deterministic `mu` and `h_cls`;
3. train an identically initialized probe on the HumanML3D training split;
4. select the probe checkpoint by complete validation masked MSE;
5. evaluate that probe on validation and test without updating the MSA-VAE.

The probe uses:

- a linear memory projection from 768 to 256;
- a two-layer Transformer decoder;
- model width 256, 8 heads, feed-forward width 1024, and dropout 0.1;
- positional target queries and an output projection to 16-D latent tokens;
- AdamW with learning rate `1e-4`;
- batch size 64;
- 5,000 optimization steps;
- validation every 1,000 steps;
- seed 123.

The reported probe error is elementwise squared error summed over valid
latent elements and divided by `valid_token_count * 16`. Padding never
contributes. Full AE additionally reports the same masked MSE for its native
semantic decoder. Native decoder error is `N/A`, not zero, for the two
decoder-free variants.

## Screening Evaluation

Architecture selection uses only the complete HumanML3D validation split.
After selection, all three architectures are evaluated on the complete test
split for the final screening table. Test results never influence selection.

### Motion reconstruction

Reuse the standard `msa-vae-standard-v2` evaluation protocol and report:

- FID;
- MPJPE in millimetres;
- P-MPJPE in millimetres;
- acceleration error in millimetres per frame squared;
- skating percentage;
- reconstructed-motion TMR text-to-motion R@1/R@5/MedR;
- reconstructed-motion TMR motion-to-text R@1/R@5/MedR.

### Direct semantic retrieval

For each complete motion, pair its frozen `h_cls` with the first
complete-motion Sentence-T5 embedding from
`humanml3d_272/text_latents_t5/<sample_id>.npy`, matching the existing
complete-motion caption policy.

L2-normalize text and motion embeddings, form the complete square cosine
similarity matrix, use average ranks for exact ties, and report:

- text-to-motion R@1/R@5/MedR;
- motion-to-text R@1/R@5/MedR.

### Complexity

Report total and trainable semantic parameter counts, serialized checkpoint
size, and inference latency for the semantic path. Latency uses a documented
batch shape, fixed warm-up count, fixed timed-iteration count, synchronized
CUDA timing, and the same GPU for all variants. Complexity metrics are
descriptive and do not override the predeclared gate.

## Screening Gate

Full AE always advances to downstream generation as the control.

A simplified candidate passes the reconstruction gate when, on validation,
all of the following hold relative to Full AE:

- MPJPE increases by no more than 1.0 mm;
- FID increases by no more than 0.05;
- common-probe latent MSE increases by no more than 5%.

Among candidates that pass, select the one with the larger average of
text-to-motion and motion-to-text semantic R@1. If their averages differ by
no more than one percentage point, select Direct Pooling because it is the
simpler architecture.

If neither candidate passes, both remain marked as failed. The downstream
diagnostic still selects the candidate with the smaller maximum normalized
gate violation. For a candidate, normalized violation is the maximum of:

```text
max(0, (MPJPE - Full_MPJPE - 1.0) / 1.0)
max(0, (FID - Full_FID - 0.05) / 0.05)
max(0, (probe_MSE / Full_probe_MSE - 1.05) / 0.05)
```

Ties use semantic R@1, then fewer semantic parameters. The collector writes
the decision and every intermediate value to `selected_variants.json`.

The two downstream representatives are therefore always Full AE and exactly
one simplified architecture.

## Downstream Paired RAG Experiment

For each representative checkpoint, extract a matching set of:

- reparameterized motion latents with the reference end latent;
- deterministic `mu` latents;
- 768-D `h_cls` retrieval features.

Extraction uses seed 123 and records the source checkpoint path, hash,
architecture, and model configuration. RAG and No-RAG for one representative
must consume the exact same extracted motion latents. RAG additionally
consumes that representative's own `h_cls` library.

Train four independent Stage-2 runs:

1. Full AE with RAG;
2. Full AE with RAG disabled;
3. selected simplified architecture with RAG;
4. selected simplified architecture with RAG disabled.

The four runs use the same:

- seed 123 and model initialization;
- DDPM generative head;
- 100,000 iterations;
- total batch size 256;
- Sentence-T5 768-D conditioning;
- retrieval top-K 5 where retrieval is enabled;
- packed-cache policy;
- EMA policy and EMA checkpoint selection;
- self-attention, cross-attention, CFG, EOS, and remaining formal launcher
  settings.

The only RAG/No-RAG difference within a representative pair is whether the
retrieval condition is enabled.

Evaluation uses the matching MSA-VAE checkpoint, latent directories,
architecture, DDPM configuration, EMA weights, top-K, and the existing
official generation evaluator. It reports FID, R@1/R@2/R@3, Diversity, and
Matching Distance.

RAG gain uses sign-corrected paired differences:

```text
FID gain = FID(No-RAG) - FID(RAG)
R@1 gain = R@1(RAG) - R@1(No-RAG)
```

Higher gain is better for both values.

## Final Structure Decision

The simplified representative replaces Full AE only when it passed the
screening reconstruction gate and all four downstream conditions hold:

- its RAG FID is at most Full AE RAG FID plus 0.10;
- its RAG R@1 is at least Full AE RAG R@1 minus one percentage point;
- its FID gain is at least Full AE FID gain minus 0.05;
- its R@1 gain is at least Full AE R@1 gain minus one percentage point.

If Encoder-only meets these conditions, remove the Semantic Decoder from the
formal method. If Direct Pooling meets them, remove both semantic Transformer
components and use direct pooling. Otherwise retain Full AE.

Because training uses one seed, the result is a structure gate for this
codebase rather than a statistical claim about training-seed variance. A
near-boundary result must be reported as inconclusive rather than rounded
across a threshold.

## Independent Shell Entrypoints

Every long-running command is a standalone per-machine shell entrypoint.
No entrypoint assumes that another variant is on the same host, assigns
remote hosts, or hard-codes a cluster GPU identifier.

Screening entrypoints:

```text
TRAIN_full_ae.sh
TRAIN_encoder_only.sh
TRAIN_direct_pool.sh
EVAL_full_ae.sh
EVAL_encoder_only.sh
EVAL_direct_pool.sh
COLLECT_SCREENING.sh
```

Latent extraction entrypoints:

```text
EXTRACT_full_ae.sh
EXTRACT_encoder_only.sh
EXTRACT_direct_pool.sh
```

Downstream entrypoints:

```text
TRAIN_RAG_full_ae.sh
TRAIN_NO_RAG_full_ae.sh
TRAIN_RAG_encoder_only.sh
TRAIN_NO_RAG_encoder_only.sh
TRAIN_RAG_direct_pool.sh
TRAIN_NO_RAG_direct_pool.sh
EVAL_RAG_full_ae.sh
EVAL_NO_RAG_full_ae.sh
EVAL_RAG_encoder_only.sh
EVAL_NO_RAG_encoder_only.sh
EVAL_RAG_direct_pool.sh
EVAL_NO_RAG_direct_pool.sh
COLLECT_DOWNSTREAM.sh
```

The user runs only the downstream scripts for the simplified architecture
named in `selected_variants.json`.

Training scripts accept the GPU count as their first positional argument:

```bash
bash explorations/semantic_architecture_gate/TRAIN_full_ae.sh 8
```

The same script works with `2`. Environment variables override output root,
seed, iteration budget, data/cache roots, process port, and validation
settings without editing the script. Defaults implement the fixed protocol
above.

`DRY_RUN=1` performs preflight and prints the exact command without launching
training or evaluation.

## Artifact and Checkpoint Contract

New MSA-VAE checkpoints record:

- semantic architecture;
- all tensor-shaping model arguments;
- phase and seed;
- training budget and alignment weights;
- fixed TAE path and SHA-256;
- Phase 2 parent checkpoint path, SHA-256, and architecture;
- dataset, normalization, and text-feature identity.

Evaluation and extraction reconstruct the model from checkpoint metadata and
load state dictionaries strictly. Existing checkpoints without the new field
remain interpretable as `full_ae`; new experiment checkpoints must contain
the explicit field.

Every evaluation manifest records its checkpoint hash, resolved
architecture, split sample hash, evaluator identity, probe identity, seed,
and protocol version. Collectors reject missing variants, mixed splits,
different sample hashes, mismatched TAE identities, mismatched budgets,
non-finite metrics, or architecture/path inconsistencies.

## Failure Handling

Each long-running script:

- verifies the `mgpt` environment and required files/directories;
- rejects an unsupported GPU count or unavailable requested devices;
- validates numeric overrides and the Accelerate rendezvous port;
- rejects a nonempty target output directory for a fresh run;
- stops Phase 2 if Phase 1 fails or lacks a valid final checkpoint;
- writes the exact command, timestamps, exit code, and checkpoint hash;
- never silently resumes, overwrites, or substitutes another architecture's
  checkpoint.

Independent machines may start all variants at the same time. Distinct
experiment directories prevent shared-filesystem collisions. If machines do
not share a filesystem, the checkpoint and manifest directories must be
copied back under the same output layout before collection; bulk model and
cache artifacts remain untracked.

## Verification

Implementation follows test-driven development. Tests cover:

- all three forward contracts and output shapes;
- absence of semantic decoder parameters in Encoder-only;
- absence of both Transformer modules in Direct Pooling;
- Direct Pooling invariance to padded-token values;
- rejection of zero-valid-token samples;
- latent reconstruction weight validation;
- optional native reconstruction output handling;
- strict checkpoint metadata and architecture matching;
- old-checkpoint Full AE compatibility;
- probe masked MSE and padding behavior;
- deterministic screening selection and tie-breaking;
- paired RAG-gain calculations and final decision thresholds;
- per-variant shell dry-run commands and artifact paths.

Local verification uses:

```bash
conda run -n mgpt python -m unittest <targeted-test-modules>
conda run -n mgpt python -m py_compile <changed-python-files>
bash -n <changed-shell-files>
git diff --check
```

A tiny CPU model smoke test exercises all three structures and strict
state-dict round trips. Fixture-based evaluation tests exercise manifest
collection without loading datasets or launching training. This development
session does not start full MSA-VAE, probe, diffusion, or official generation
runs.

## Completion Criteria

The implementation handoff is complete when:

- the branch contains the three architecture implementations and artifact
  contracts;
- every listed per-variant shell entrypoint exists and passes dry-run
  validation;
- probe, screening, selection, extraction, downstream pairing, and final
  collection pipelines are executable;
- targeted unit, smoke, syntax, and diff checks pass;
- the usage guide explains how to distribute the independent commands across
  separate two- or eight-GPU machines;
- no data or generated artifact is committed.

The scientific experiment is complete only after the user runs the three
MSA-VAE jobs, the three screening evaluations, the two selected
representatives' four paired Stage-2 jobs, and their evaluations, then runs
both collectors successfully.
