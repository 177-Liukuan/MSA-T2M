# MSA-VAE Alignment–Realism Pilot Experiment Design

## Objective

Produce a single-seed pilot table for the reviewer-requested MSA-VAE
alignment–realism trade-off. The experiment compares no semantic alignment,
global-only alignment, local-only alignment, and combined global/local
alignment while holding the model, data, TAE initialization, optimization
budget, random seed, and evaluation protocol fixed.

The requested output columns are:

| Variant | FID↓ | MPJPE↓ | P-MPJPE↓ | ACCEL↓ | Skating%↓ | T2M R@1↑ | T2M R@5↑ | T2M MedR↓ | M2T R@1↑ | M2T R@5↑ | M2T MedR↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

This is explicitly a rapid pilot. Every variant uses only seed 123. The
result must not be presented as a multi-seed mean or with a standard
deviation. Seeds 456 and 789 can be added later using the same experiment
identity.

## Scientific Comparison

The four variants are:

| Variant | Phase 1 global/local | Phase 2 global/local |
|---|---:|---:|
| No Alignment | 0 / 0 | 0 / 0 |
| Global Only | 0.2 / 0 | 0.05 / 0 |
| Local Only | 0 / 0.2 | 0 / 0.05 |
| Global + Local | 0.2 / 0.2 | 0.05 / 0.05 |

Global supervision covers every HumanML3D training sample. Local supervision
covers 7,056 of 23,384 train IDs (approximately 30.2%) and is masked where
unavailable. The current loss computes a mean over valid local samples and
tokens, so the pilot uses equal active coefficients instead of multiplying
the local weight by the inverse coverage.

The coefficients are deliberately symmetric. Global-only and local-only
therefore compare alignment type under the same nominal coefficient, while
Global + Local applies both constraints and has twice the total nominal
alignment pressure. No Alignment anchors the motion-realism cost of adding
semantic supervision.

The older Phase-1 0.5/0.2 and Phase-2 0.1/0.001 values are not official
weights and are not reused. In particular, the old Phase-2 local coefficient
would make the local-only comparison too weak to interpret.

## Fixed Training Contract

All four variants use:

- HumanML3D 272-D representation and the complete HumanML training split;
- `--no_ft_split`;
- T5 text embeddings with dimension 768;
- seed 123;
- validation seed 123 and validation batch size 32;
- fixed TAE checkpoint
  `Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth`;
- fixed TAE SHA-256
  `7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8`;
- exactly two GPUs per training process group;
- Phase 1 full-sequence training followed by Phase 2 mixed
  full-sequence/window-replay training;
- 25,000 iterations per phase;
- the existing launcher learning rates, batch sizes, model architecture,
  replay interval, and scheduler settings;
- deterministic complete `val.txt` validation every 5,000 iterations;
- `net_best_fid.pth`, `net_best_mpjpe.pth`, and `net_last.pth` publication;
- no automatic GIF or TensorBoard video rendering.

Every Phase 2 run must resume its own Phase 1 `net_last.pth`. No existing
MSA-VAE checkpoint may be reused.

## GPU and Screen Layout

Training uses all eight RTX 4090 GPUs as four independent two-GPU groups:

| Variant | CUDA devices | Screen session |
|---|---|---|
| No Alignment | 0,1 | `msa_pilot_no_align_s123` |
| Global Only | 2,3 | `msa_pilot_global_s123` |
| Local Only | 4,5 | `msa_pilot_local_s123` |
| Global + Local | 6,7 | `msa_pilot_both_s123` |

Each screen session runs one durable orchestration command:

1. launch Phase 1 on the assigned two visible GPUs;
2. stop immediately if Phase 1 exits nonzero or lacks a valid
   `net_last.pth`;
3. launch Phase 2 on the same GPUs using that exact Phase 1 directory;
4. write a terminal status marker and preserve stdout/stderr logs.

The launch process must validate that all eight GPUs are idle before starting.
It must also reject pre-existing nonempty target experiment directories so a
fresh pilot cannot silently resume or overwrite an earlier run.

## Experiment Identity and Artifacts

Use a dedicated output root:

`Experiments/msa_vae_alignment_realism_pilot_s123_20260726/`

Each variant has separate Phase 1 and Phase 2 experiment names containing:

- the variant slug;
- seed 123;
- phase number;
- the 25k pilot budget;
- the global/local weights.

A machine-readable run manifest records the git commit, TAE path and hash,
GPU pair, seed, training budget, validation identity, weights, exact launcher
commands, experiment directories, timestamps, process exit codes, and final
checkpoint SHA-256 values. Runtime logs and screen logs remain local
experiment artifacts and are not committed.

## Evaluation Protocol

Only Phase-2 `net_last.pth` is used for the pilot table. Best checkpoints are
retained for diagnostics but never substituted into the formal comparison.

After all four Phase 2 runs complete:

1. verify checkpoint metadata, phase lineage, model configuration, seed,
   weights, TAE identity, and checkpoint hashes;
2. run `eval_msa_vae_metrics.py` on the complete HumanML3D `test.txt` split;
3. use evaluation seed 123, batch size 32, and the standard
   `msa-vae-standard-v2` evaluator protocol;
4. run the four standalone evaluations concurrently on GPUs 0, 2, 4, and 6;
5. write one `metrics.json` per variant with no GIF/video output;
6. assemble the requested Markdown and CSV table directly from the four
   single-seed manifests.

The existing three-seed formal aggregator is intentionally not used for this
pilot because it correctly rejects fewer than three independently trained
seeds.

## Monitoring and Failure Handling

Monitoring checks:

- screen session presence;
- training process presence;
- GPU memory/utilization;
- last reported iteration and loss values;
- complete-validation sample count, FID, and MPJPE;
- NaN/Inf, traceback, CUDA OOM, NCCL error, or nonzero exit;
- Phase 1 to Phase 2 transition;
- expected checkpoint files and available disk space.

No failed variant is silently omitted. If a run fails, the other independent
screen sessions continue. The failed variant is diagnosed and restarted only
after resolving the cause; it keeps the same scientific configuration and
gets a fresh output directory or an explicitly documented recovery action.

## Analysis Plan

The final summary reports the four raw single-seed rows and labels them as a
pilot. Analysis separates:

- realism/reconstruction: FID, MPJPE, P-MPJPE, ACCEL, and Skating%;
- semantic retrieval: bidirectional R@1, R@5, and MedR;
- global-vs-local differences under equal nominal coefficients;
- whether combined alignment improves retrieval while worsening motion
  realism relative to No Alignment;
- whether the evidence supports the motivation that stronger semantic
  alignment introduces a measurable realism cost.

Claims remain descriptive because one seed does not establish statistical
significance. If the trend is useful for the reviewer response, the same four
variants can be rerun at seeds 456 and 789 and aggregated as mean ± standard
deviation.

## Completion Criteria

The pilot is complete only when:

- all four Phase 1 and Phase 2 runs exit successfully;
- every training group used exactly two GPUs;
- every Phase 2 checkpoint has valid Phase-1 lineage and the fixed TAE
  identity;
- four complete-test `metrics.json` files exist and share the same dataset,
  evaluator, seed, batch size, and protocol identities;
- the requested Markdown/CSV table contains all eleven metrics for all four
  variants;
- a concise, single-seed-qualified alignment–realism analysis is delivered.
