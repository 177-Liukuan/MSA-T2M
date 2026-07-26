# MSA-VAE Internal Alignment Evaluation Design

**Date:** 2026-07-26
**Status:** Approved

## Objective

Redesign the MSA-VAE ablation evaluation so that the main table directly
measures the alignment objective learned inside MSA-VAE while separately
measuring reconstructed-motion realism.

The resulting protocol must support the following variants:

- no alignment;
- global alignment only;
- local alignment only;
- global and local alignment.

The pilot uses one fixed training seed. Later formal experiments may repeat the
same protocol with multiple training seeds.

## Problem with the Current Protocol

The current standalone evaluator sends only `x_recon` through a frozen external
TMR motion encoder and sends captions through the matching frozen TMR text
encoder. Its T2M/M2T retrieval values therefore measure whether an external
TMR model can still recognize the reconstructed motion.

Training uses a different semantic contract:

- global MSA feature: `global_proj(h_cls)`;
- local MSA feature: `local_proj(mu)`;
- text target: 768-dimensional SentenceT5 embeddings.

The current TMR retrieval protocol neither consumes the MSA semantic features
nor operates in the SentenceT5 space used by the alignment losses. It is a
valid secondary measure of reconstructed-motion semantic preservation, but it
is not an internal MSA-VAE alignment measure.

## Evaluation Decomposition

The evaluator must use three explicit branches:

```text
motion -> deterministic MSA semantic forward
          |-> global_proj(h_cls) <-> global SentenceT5 -> global alignment/retrieval
          |-> local_proj(mu)     <-> local SentenceT5  -> local alignment
          `-> mu -> CNN decoder  -> reconstructed motion -> realism metrics
```

The external frozen TMR motion encoder remains the feature extractor for
reconstruction FID. External TMR text-to-motion and motion-to-text retrieval
move to a supplementary result and must not be labeled as internal alignment.

## Global Alignment Protocol

### Features

For each valid full HumanML3D test motion, run the deterministic semantic path:

```python
semantic = model.forward_semantic(motion, lengths)
motion_feature = normalize(semantic["clip_global_feat"])
```

The paired text features are the exact SentenceT5 rows associated with the
complete-motion captions in `humanml3d_272/text_latents_t5/`. The evaluator
must preserve each caption's source line index so it loads the same cache row
that the training dataset would use.

### Global cosine similarity

Compute cosine similarity between every complete caption and its motion
feature. Average caption similarities within each motion first, then average
over motions. This motion-macro average prevents motions with more captions
from receiving greater weight.

Report:

```text
Global Cosine = mean_motion(mean_caption(cos(g_motion, t_caption)))
```

This is equivalent in direction to `1 - global_alignment_loss`.

### Internal MSA-SentenceT5 retrieval

Use normalized dot products between SentenceT5 caption features and
`global_proj(h_cls)` motion features.

- T2M: every complete caption queries the full candidate motion corpus; the
  source motion is positive.
- M2T: every motion queries the full caption corpus; any complete caption
  belonging to that motion is positive.
- Rank for a query with multiple positives is the best rank among its
  positives.
- Report R@1, R@5, and median rank in both directions.

The candidate corpus and ordering must be fixed across all variants. Retrieval
must not use `x_recon` or the external TMR text encoder.

## Local Alignment Protocol

For every motion with a valid held-out local target:

```python
local_feature = normalize(semantic["clip_local_feat"])
```

Build its target exactly as training does:

1. load the frame-level local SentenceT5 target;
2. map the approximately 20 FPS target to the 30 FPS motion length;
3. crop to the evaluated complete motion;
4. average-pool to the causal latent rate;
5. mask padding and invalid latent tokens.

Compute token cosine similarity, average valid tokens within each motion, then
average over motions:

```text
Local Cosine = mean_motion(mean_valid_token(cos(local_feature, local_target)))
```

Local token retrieval is not a main-table metric. Adjacent tokens frequently
share the same action description, and tokens crossing action boundaries can
have pooled mixture targets, so a unique diagonal retrieval target is not
well-defined.

## Held-Out Local Target Requirement

The current deterministic HumanML3D evaluation set contains 2,480 valid
motions. All 2,480 have global SentenceT5 cache files, but none has a local
file in the current `humanml3d_272/t5_enc_single/` cache.

The repository contains:

- 7,056 local cache files for `train_ft.txt`;
- 444 raw IDs in `val_ft.txt`;
- 1,348 raw IDs in `test_ft.txt`.

The `val_ft` and `test_ft` IDs belong to the corresponding HumanML3D splits
and are disjoint from `train.txt` and `train_ft.txt`. However, their current
`babel_272/texts/` files contain only complete-motion descriptions and no
temporal local annotations. A complete-motion description must not be repeated
across frames and presented as a local target.

The final local protocol therefore uses the first feasible option below:

1. recover the original temporal BABEL annotations and generate exact
   `val_ft` and `test_ft` SentenceT5 targets with the same encoder and temporal
   preprocessing contract as training; use validation for weight selection and
   test for final reporting;
2. if the original targets cannot be recovered, define a fixed held-out subset
   before training, exclude those motions from all optimization, and retrain
   every variant under the same split;
3. for the already-completed pilot only, local cosine on `train_ft` may be
   emitted as an explicitly labeled `in_sample_local_cosine` diagnostic. It is
   not a paper result and must not populate the final `Local Cosine` column.

BABEL-stream validation is not a substitute for this HumanML3D held-out
protocol because it has a different motion domain, normalization contract, and
TAE checkpoint.

## Reconstruction and Realism Protocol

The main table uses deterministic posterior-mean reconstruction:

```python
semantic = model.forward_semantic(motion, lengths)
prediction = model.decode_cnn(semantic["mu"])
```

This isolates checkpoint differences from Monte Carlo noise introduced by a
single posterior sample. Report:

- FID;
- MPJPE in millimeters;
- P-MPJPE in millimeters;
- ACCEL;
- skating percentage.

The existing frozen TMR motion encoder may compute reconstruction FID. Its text
encoder is not used by the main-table protocol.

As a supplementary robustness check, stochastic reconstruction may be repeated
for three to five fixed evaluation seeds. The same sample-ID-derived epsilon
must be used for every variant at a given seed.

## Checkpoint and Training Controls

- Keep the currently approved TAE checkpoint fixed across all variants.
- Keep architecture, data split, training seed, iteration budget, and
  non-alignment loss weights fixed.
- The pilot uses 25k iterations per phase.
- Use the same fixed Phase-2 terminal iteration for the primary comparison.
- Continue saving `best_fid` and `best_mpjpe`, but do not select a different
  best-realism checkpoint per variant for the primary trade-off table.
- A best-checkpoint sensitivity table may be reported separately.

## Main and Supplementary Tables

The main table is:

| Variant | Global Cosine up | Local Cosine up | FID down | MPJPE down | P-MPJPE down | ACCEL down | Skating percent down | MSA-T5 T2M R@1 up | T2M R@5 up | T2M MedR down | MSA-T5 M2T R@1 up | M2T R@5 up | M2T MedR down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Alignment | | | | | | | | | | | | | |
| Global Only | | | | | | | | | | | | | |
| Local Only | | | | | | | | | | | | | |
| Global + Local | | | | | | | | | | | | | |

Every result records:

- evaluated checkpoint path and SHA-256;
- checkpoint metadata and alignment weights;
- dataset split, valid sample count, and sample-ID hash;
- global and local target coverage;
- text-cache identity;
- deterministic decode mode;
- evaluation seed;
- model structural configuration.

The supplementary external-semantic-preservation table retains the current
frozen-TMR retrieval values under explicit names such as
`External-TMR T2M R@1`.

## Trade-Off Analysis

Do not collapse global and local alignment into a single primary score.

Report:

- global alignment versus each realism metric;
- local alignment versus each realism metric;
- absolute and relative deltas from No Alignment;
- Pareto plots for global alignment versus FID/MPJPE;
- Pareto plots for local alignment versus FID/MPJPE.

For the one-seed pilot, describe differences as trends rather than statistical
significance. Formal results should later report mean and standard deviation
over multiple training seeds.

## Required Sanity Checks

The evaluator must fail closed on incompatible checkpoints, missing cache rows,
wrong feature dimensions, non-finite values, duplicate sample IDs, or target
coverage mismatches.

Run the following negative controls:

- shuffle global text ownership and verify retrieval approaches chance;
- verify retrieval results are invariant to evaluation batch size;
- verify deterministic posterior-mean results repeat exactly;
- verify no main-table retrieval path consumes `x_recon`;
- verify local masking and macro averaging reproduce the training loss
  convention on synthetic data;
- verify a missing held-out local cache cannot silently become a zero target.

## Non-Goals

- Do not change MSA-VAE training architecture as part of this evaluator change.
- Do not add a second alignment head.
- Do not evaluate HumanML checkpoints with BABEL joint normalization.
- Do not claim that an internal alignment gain must necessarily damage
  reconstruction; the experiment measures whether and when that trade-off
  occurs.
