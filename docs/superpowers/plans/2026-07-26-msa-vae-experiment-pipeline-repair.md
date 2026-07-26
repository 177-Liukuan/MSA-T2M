# MSA-VAE Experiment Pipeline Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fresh two-GPU MSA-VAE Phase-1/Phase-2 training and the requested 12-column, three-seed alignment--realism evaluation pipeline runnable and reproducible.

**Architecture:** Keep the two-stage model and legacy entrypoints intact. Repair the full-split dataset boundary, adapt only the standalone standard evaluator to dynamic TMR masks, extend its additive metric schema to R@5, carry immutable training identity through checkpoints, and add a focused three-seed aggregator plus environment-configurable official launchers.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1+cu118, Accelerate 1.0.1, NumPy, SciPy, `unittest`, Bash, existing `mgpt` Conda environment.

## Global Constraints

- Use `conda activate mgpt` or `conda run -n mgpt`; do not replace or upgrade the environment.
- Formal training uses HumanML3D-272 with `--no_ft_split`, full global supervision, and sparse local supervision.
- Every training job uses exactly two RTX 4090 GPUs; four jobs may occupy GPU pairs 0--1, 2--3, 4--5, and 6--7.
- All fresh runs initialize from `Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth`.
- Do not reuse existing MSA-VAE checkpoints for paper results.
- Preserve model tensor keys, causal latent convention, 272-D representation, normalization, and frozen TMR evaluator weights.
- Do not edit `explorations/` or any third-party submodule.
- Do not start formal 50k training during repair validation.
- Preserve unrelated untracked files and user changes.

---

## File Map

- Modify `humanml3d_272/dataset_msa_vae.py`: deterministic sub-unit-motion filtering and summary count.
- Modify `tests/test_dataset_msa_vae_full_sequence.py`: real regression contract for filtering.
- Modify `eval_msa_vae_metrics.py`: dynamic TMR mask construction, v2 R@5 artifact schema, and checkpoint training identity.
- Modify `utils/msa_vae_metrics.py`: additive R@5 calculation.
- Modify `utils/msa_vae_eval_config.py`: expose trusted checkpoint metadata in checkpoint identity.
- Modify `utils/msa_vae_training.py`: record architecture, seed, variant, and alignment weights in new checkpoints.
- Modify `tests/test_msa_vae_metrics.py`: R@5 ranking behavior.
- Modify `tests/test_eval_msa_vae_metrics.py`: dynamic real-encoder contract and v2 artifacts.
- Modify `tests/test_msa_vae_eval_config.py`: checkpoint metadata propagation.
- Modify `tests/test_msa_vae_training_entrypoint.py`: fresh training metadata contract.
- Create `aggregate_msa_vae_metrics.py`: validate and aggregate exactly three evaluation manifests.
- Create `AGGREGATE_msa_vae_metrics.sh`: invoke the aggregator in `mgpt`.
- Create `tests/test_aggregate_msa_vae_metrics.py`: aggregation and fail-closed compatibility tests.
- Modify `TRAIN_msa_vae_phase1.sh`: environment-configurable identity, weights, seed, budget, and output.
- Modify `TRAIN_msa_vae_phase2.sh`: matching Phase-2 overrides and Phase-1 binding.
- Modify `tests/test_msa_vae_full_sequence_launchers.py`: override forwarding and rejection tests.

---

### Task 1: Filter Motions That Cannot Produce a Latent Unit

**Files:**
- Modify: `humanml3d_272/dataset_msa_vae.py:84-180`
- Modify: `tests/test_dataset_msa_vae_full_sequence.py:84-157`

**Interfaces:**
- Consumes: `MSAVAEDataset(..., unit_length: int, sequence_mode: str)`.
- Produces: `dataset.skipped_subunit_count: int`; source motions with `frames < unit_length` are omitted without aborting construction.

- [ ] **Step 1: Replace the fatal fixture expectation with a mixed valid/invalid failing test**

```python
def test_full_dataset_skips_motion_shorter_than_one_latent_unit(self):
    # Build `valid.npy` with shape (20, 272) and `too_short.npy` with
    # shape (3, 272), put both IDs in train.txt, and provide full captions.
    dataset = MSAVAEDataset(
        "t2m_272",
        window_size=64,
        unit_length=4,
        use_ft_split=False,
        text_encoder_type="clip",
        text_embed_dim=2,
        sequence_mode="full",
    )
    self.assertEqual(len(dataset), 1)
    self.assertEqual(dataset.data[0]["name"], "valid")
    self.assertEqual(dataset.skipped_subunit_count, 1)
```

- [ ] **Step 2: Run the focused test and verify the current fatal behavior**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence.MSAVAEDatasetFullSequenceTest.test_full_dataset_skips_motion_shorter_than_one_latent_unit
```

Expected: FAIL because `MotionSequenceTooShortError` is raised.

- [ ] **Step 3: Implement one deterministic skip counter**

In `MSAVAEDataset.__init__`, initialize:

```python
self.skipped_subunit_count = 0
```

Replace the fatal branch with:

```python
if motion.shape[0] < self.unit_length:
    self.skipped_subunit_count += 1
    continue
```

Include `skipped_subunit_count` in the final dataset summary. Remove
`MotionSequenceTooShortError` only if no public test or caller still imports
it; otherwise retain the class unused for compatibility.

- [ ] **Step 4: Run dataset and full-sequence tests**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence \
  tests.test_msa_vae_full_sequence_smoke
```

Expected: PASS.

- [ ] **Step 5: Commit the isolated dataset repair**

```bash
git add humanml3d_272/dataset_msa_vae.py \
  tests/test_dataset_msa_vae_full_sequence.py
git commit -m "fix: skip subunit HumanML motions in MSA-VAE data"
```

---

### Task 2: Make Standalone TMR Evaluation Compatible with Dynamic Padding

**Files:**
- Modify: `eval_msa_vae_metrics.py:405-445`
- Modify: `tests/test_eval_msa_vae_metrics.py`

**Interfaces:**
- Consumes: `load_frozen_humanml_evaluator(...) -> list[nn.Module]`.
- Produces: a frozen `ActorAgnosticEncoder` constructed with `max_len=-1`.

- [ ] **Step 1: Add a failing assertion for dynamic motion-encoder masks**

Add a test that patches the encoder classes with lightweight doubles and
captures the constructor arguments:

```python
def test_frozen_motion_encoder_uses_dynamic_max_length(self):
    # Patch ActorAgnosticEncoder and checkpoint loading as existing loader
    # tests do, then call load_frozen_humanml_evaluator(...).
    self.assertEqual(motion_encoder_constructor.call_args.kwargs["max_len"], -1)
```

Also add a small real-encoder equivalence test using a seeded
`ActorAgnosticEncoder(nfeats=272, vae=True, latent_dim=8, num_layers=1,
max_len=-1)`:

```python
dynamic = encoder(features[:, :64], torch.tensor([60, 64])).loc
encoder.max_len = 80
fixed = encoder(F.pad(features[:, :64], (0, 0, 0, 16)),
                torch.tensor([60, 64])).loc
torch.testing.assert_close(dynamic, fixed)
```

- [ ] **Step 2: Run the new tests and verify the constructor mismatch**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_vae_metrics
```

Expected: FAIL because the production constructor passes `max_len=300`.

- [ ] **Step 3: Change only the standalone evaluator constructor**

In `load_frozen_humanml_evaluator`, use:

```python
motion_encoder = ActorAgnosticEncoder(
    nfeats=272,
    vae=True,
    num_layers=4,
    latent_dim=256,
    max_len=-1,
)
```

Do not edit `Evaluator_272/` and do not change the training-time legacy
evaluator.

- [ ] **Step 4: Run evaluator-focused tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_vae_metrics \
  tests.test_msa_vae_metrics_dataset \
  tests.test_msa_vae_metrics_launcher
```

Expected: PASS.

- [ ] **Step 5: Commit the mask repair**

```bash
git add eval_msa_vae_metrics.py tests/test_eval_msa_vae_metrics.py
git commit -m "fix: use dynamic TMR masks for MSA-VAE metrics"
```

---

### Task 3: Add Bidirectional R@5 and Version the Evaluation Schema

**Files:**
- Modify: `utils/msa_vae_metrics.py:246-290`
- Modify: `eval_msa_vae_metrics.py:31-48,472-590`
- Modify: `tests/test_msa_vae_metrics.py:130-195`
- Modify: `tests/test_eval_msa_vae_metrics.py:195-310`

**Interfaces:**
- Consumes: `retrieval_metrics_from_similarity(similarity: np.ndarray)`.
- Produces: `t2m_r5_percent: float`, `m2t_r5_percent: float`, protocol `msa-vae-standard-v2`.

- [ ] **Step 1: Add failing rank-boundary tests**

Use a 6-by-6 matrix whose diagonal positive has exactly four better
candidates for one query and five for another:

```python
metrics = retrieval_metrics_from_similarity(similarity)
self.assertEqual(metrics["t2m_r5_percent"], 500.0 / 6.0)
self.assertIn("m2t_r5_percent", metrics)
```

Extend artifact fixtures with both R@5 keys and assert the CSV, JSON units,
and report contain `R@5`.

- [ ] **Step 2: Run metric and artifact tests to verify missing keys**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_metrics \
  tests.test_eval_msa_vae_metrics
```

Expected: FAIL with missing `*_r5_percent`.

- [ ] **Step 3: Implement additive R@5**

Extend `_rank_metrics`:

```python
return {
    f"{prefix}_r1_percent": float(np.mean(ranks < 1) * 100.0),
    f"{prefix}_r2_percent": float(np.mean(ranks < 2) * 100.0),
    f"{prefix}_r3_percent": float(np.mean(ranks < 3) * 100.0),
    f"{prefix}_r5_percent": float(np.mean(ranks < 5) * 100.0),
    f"{prefix}_medr": float(np.median(ranks) + 1.0),
}
```

Add both keys to `METRIC_KEYS`, units, final report, and artifact tests. Change
only the standard protocol version string to `msa-vae-standard-v2`.

- [ ] **Step 4: Run the focused suite**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_metrics \
  tests.test_eval_msa_vae_metrics \
  tests.test_msa_vae_metrics_launcher
```

Expected: PASS.

- [ ] **Step 5: Commit the schema change**

```bash
git add utils/msa_vae_metrics.py eval_msa_vae_metrics.py \
  tests/test_msa_vae_metrics.py tests/test_eval_msa_vae_metrics.py
git commit -m "feat: report bidirectional R@5 for MSA-VAE"
```

---

### Task 4: Carry Fresh Training Identity Through Checkpoints and Evaluation

**Files:**
- Modify: `utils/msa_vae_training.py:315-375`
- Modify: `utils/msa_vae_eval_config.py:184-290`
- Modify: `tests/test_msa_vae_training_entrypoint.py:120-190`
- Modify: `tests/test_msa_vae_eval_config.py`
- Modify: `eval_msa_vae_metrics.py:472-530`
- Modify: `tests/test_eval_msa_vae_metrics.py`

**Interfaces:**
- Consumes: parsed MSA-VAE training arguments.
- Produces: `metadata["training_args"]` with architecture fields plus
  `seed`, `global_align_weight`, `local_align_weight`, `exp_name`,
  `msa_data_mode`, and `resume_cnn_pth`; evaluation manifest copies this
  metadata under `checkpoint.metadata`.

- [ ] **Step 1: Extend metadata fixtures and write failing assertions**

Extend the test `_args()` with every field consumed below and assert:

```python
metadata = build_msa_checkpoint_metadata(args)
self.assertEqual(metadata["training_args"]["seed"], 123)
self.assertEqual(metadata["training_args"]["global_align_weight"], 0.25)
self.assertEqual(metadata["training_args"]["trans_enc_layers"], 6)
```

Add an eval-config test:

```python
model, resolved, identity = build_and_load_msa_vae(...)
self.assertEqual(identity["metadata"], checkpoint_metadata)
```

- [ ] **Step 2: Run tests and confirm metadata is absent**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_eval_config \
  tests.test_eval_msa_vae_metrics
```

Expected: FAIL on `training_args` or `checkpoint.metadata`.

- [ ] **Step 3: Add a JSON-safe explicit training-identity allowlist**

Define a tuple containing all `CONFIG_FIELDS` plus:

```python
TRAINING_IDENTITY_FIELDS = (
    "seed",
    "global_align_weight",
    "local_align_weight",
    "latent_recon_weight",
    "root_loss",
    "exp_name",
    "msa_data_mode",
    "text_encoder_type",
    "text_embed_dim",
    "resume_cnn_pth",
)
```

Store values under `metadata["training_args"]`. Keep existing top-level
structural fields for Phase handoff compatibility. In
`build_and_load_msa_vae`, attach a defensive dictionary copy of payload
metadata to the returned checkpoint identity.

- [ ] **Step 4: Run metadata and evaluator tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_eval_config \
  tests.test_eval_msa_vae_metrics
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint provenance**

```bash
git add utils/msa_vae_training.py utils/msa_vae_eval_config.py \
  eval_msa_vae_metrics.py tests/test_msa_vae_training_entrypoint.py \
  tests/test_msa_vae_eval_config.py tests/test_eval_msa_vae_metrics.py
git commit -m "feat: record MSA-VAE training identity in metrics"
```

---

### Task 5: Aggregate Exactly Three Compatible Seeds

**Files:**
- Create: `aggregate_msa_vae_metrics.py`
- Create: `AGGREGATE_msa_vae_metrics.sh`
- Create: `tests/test_aggregate_msa_vae_metrics.py`

**Interfaces:**
- Consumes: `load_manifest(path: Path) -> dict`;
  `aggregate_variant(variant: str, manifests: Sequence[dict]) -> dict`.
- Produces: `aggregate.json`, `aggregate.csv`, and `table.md`.

- [ ] **Step 1: Write failing aggregation tests**

Build three in-memory v2 manifests with metrics `1.0`, `2.0`, and `3.0` and
training seeds 123, 456, 789. Assert:

```python
result = aggregate_variant("Global + Local", manifests)
self.assertEqual(result["variant"], "Global + Local")
self.assertEqual(result["metrics"]["fid"]["mean"], 2.0)
self.assertEqual(result["metrics"]["fid"]["std"], 1.0)
```

Add subtests that reject:

- two or four inputs;
- v1 protocol;
- duplicate seed;
- duplicate checkpoint SHA-256;
- mismatched evaluator SHA-256;
- mismatched sample hash/count;
- mismatched skating configuration;
- missing/non-finite target metrics.

Assert artifact CSV has `fid_mean` and `fid_std`, while Markdown uses
`mean ± std` and only the requested table metrics.

- [ ] **Step 2: Run the new test and verify import failure**

```bash
conda run -n mgpt python -m unittest tests.test_aggregate_msa_vae_metrics
```

Expected: FAIL because `aggregate_msa_vae_metrics` does not exist.

- [ ] **Step 3: Implement the focused aggregator**

Define:

```python
TARGET_METRICS = (
    "fid", "mpjpe_mm", "p_mpjpe_mm", "accel_mm_per_frame2",
    "skating_percent", "t2m_r1_percent", "t2m_r5_percent",
    "t2m_medr", "m2t_r1_percent", "m2t_r5_percent", "m2t_medr",
)

def aggregate_variant(variant, manifests):
    validate_compatible_manifests(manifests)
    return {
        "variant": variant,
        "seed_count": 3,
        "metrics": {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for key in TARGET_METRICS
        },
        "sources": source_identities,
    }
```

The CLI requires `--variant`, `--output-dir`, and exactly three positional
`metrics.json` paths. The shell launcher follows the same active-`mgpt`
pattern as `EVAL_msa_vae_metrics.sh`.

- [ ] **Step 4: Run aggregator and launcher tests**

```bash
conda run -n mgpt python -m unittest tests.test_aggregate_msa_vae_metrics
bash -n AGGREGATE_msa_vae_metrics.sh
```

Expected: PASS.

- [ ] **Step 5: Commit aggregation support**

```bash
git add aggregate_msa_vae_metrics.py AGGREGATE_msa_vae_metrics.sh \
  tests/test_aggregate_msa_vae_metrics.py
git commit -m "feat: aggregate three-seed MSA-VAE metrics"
```

---

### Task 6: Parameterize Official Two-Stage Launchers Safely

**Files:**
- Modify: `TRAIN_msa_vae_phase1.sh`
- Modify: `TRAIN_msa_vae_phase2.sh`
- Modify: `tests/test_msa_vae_full_sequence_launchers.py`

**Interfaces:**
- Consumes environment variables `EXP_NAME`, `GLOBAL_ALIGN_WEIGHT`,
  `LOCAL_ALIGN_WEIGHT`, `SEED`, `TOTAL_ITER`, `EVAL_ITER`, `OUT_DIR`, and
  existing `PHASE1_DIR`.
- Produces unique, logged arguments to `accelerate launch`.

- [ ] **Step 1: Add failing forwarding tests**

For each launcher, invoke the existing stub with:

```python
{
    "EXP_NAME": "ablation_global_seed456",
    "GLOBAL_ALIGN_WEIGHT": "0.25",
    "LOCAL_ALIGN_WEIGHT": "0.0",
    "SEED": "456",
    "TOTAL_ITER": "1234",
    "EVAL_ITER": "321",
    "OUT_DIR": "Experiments/test-output",
}
```

Assert the corresponding CLI values exactly match. Add rejection tests for
`EXP_NAME=""`, `GLOBAL_ALIGN_WEIGHT="-0.1"`, and
`LOCAL_ALIGN_WEIGHT="-0.1"`.

- [ ] **Step 2: Run launcher tests and verify hard-coded values**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_full_sequence_launchers
```

Expected: FAIL because overrides are not forwarded and invalid weights are
not rejected.

- [ ] **Step 3: Add portable environment defaults and validation**

Use parameter expansion without assigning GPU IDs:

```bash
OUT_DIR=${OUT_DIR:-Experiments}
GLOBAL_ALIGN_WEIGHT=${GLOBAL_ALIGN_WEIGHT:-0.5}
LOCAL_ALIGN_WEIGHT=${LOCAL_ALIGN_WEIGHT:-0.2}
SEED=${SEED:-123}
TOTAL_ITER=${TOTAL_ITER:-50000}
EVAL_ITER=${EVAL_ITER:-2500}
EXP_NAME=${EXP_NAME:-MSA_VAEv7_phase1_fullseq_${dataset_name}_${TEXT_ENCODER_TYPE}_fulldb}
```

Phase 2 keeps its own existing weight and evaluation defaults. Validate the
experiment name and non-negative numeric weights with a small shell function
using `awk`, then pass every resolved value to `train_msa_vae.py`. Print all
resolved values before launch.

- [ ] **Step 4: Run launcher tests and shell syntax checks**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_full_sequence_launchers
bash -n TRAIN_msa_vae_phase1.sh TRAIN_msa_vae_phase2.sh
```

Expected: PASS.

- [ ] **Step 5: Commit launcher parameterization**

```bash
git add TRAIN_msa_vae_phase1.sh TRAIN_msa_vae_phase2.sh \
  tests/test_msa_vae_full_sequence_launchers.py
git commit -m "feat: parameterize MSA-VAE ablation launchers"
```

---

### Task 7: Verify the Complete Fresh Two-Stage Pipeline

**Files:**
- Verify only; do not change production files unless a new failing test first
  reproduces an additional root cause.

**Interfaces:**
- Consumes: all outputs from Tasks 1--6.
- Produces: evidence that Phase 1, Phase 2, v2 evaluation, and three-seed
  aggregation entrypoints run under the declared contracts.

- [ ] **Step 1: Run the complete focused test suite**

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence \
  tests.test_msa_vae_full_sequence_smoke \
  tests.test_msa_vae_training \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_eval_config \
  tests.test_msa_vae_metrics \
  tests.test_msa_vae_metrics_dataset \
  tests.test_eval_msa_vae_metrics \
  tests.test_msa_vae_metrics_launcher \
  tests.test_aggregate_msa_vae_metrics \
  tests.test_msa_vae_full_sequence_launchers
```

Expected: PASS.

- [ ] **Step 2: Run static validation**

```bash
conda run -n mgpt python -m py_compile \
  train_msa_vae.py eval_msa_vae_metrics.py \
  aggregate_msa_vae_metrics.py models/msa_vae.py \
  options/option_msa_vae.py humanml3d_272/dataset_msa_vae.py \
  humanml3d_272/dataset_eval_msa_vae_metrics.py \
  utils/msa_vae_training.py utils/msa_vae_metrics.py \
  utils/msa_vae_eval_config.py
bash -n TRAIN_msa_vae_phase1.sh TRAIN_msa_vae_phase2.sh \
  EVAL_msa_vae_metrics.sh AGGREGATE_msa_vae_metrics.sh
git diff --check
```

Expected: all exit 0.

- [ ] **Step 3: Run fresh two-GPU one-step Phase 1**

Use a new temporary directory and GPUs 0--1:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OUT_DIR=/tmp/msa-vae-repair-smoke \
EXP_NAME=phase1_seed123 \
GLOBAL_ALIGN_WEIGHT=0.5 \
LOCAL_ALIGN_WEIGHT=0.2 \
SEED=123 TOTAL_ITER=1 EVAL_ITER=1000000 \
CNN_CKPT=Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth \
bash TRAIN_msa_vae_phase1.sh 2 t2m_272
```

Expected: exit 0 and
`/tmp/msa-vae-repair-smoke/phase1_seed123/net_last.pth` exists.

- [ ] **Step 4: Run fresh two-GPU one-step Phase 2**

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OUT_DIR=/tmp/msa-vae-repair-smoke \
EXP_NAME=phase2_seed123 \
PHASE1_DIR=/tmp/msa-vae-repair-smoke/phase1_seed123 \
GLOBAL_ALIGN_WEIGHT=0.1 \
LOCAL_ALIGN_WEIGHT=0.01 \
SEED=123 TOTAL_ITER=1 EVAL_ITER=1000000 \
bash TRAIN_msa_vae_phase2.sh 2 t2m_272
```

Expected: exit 0 and
`/tmp/msa-vae-repair-smoke/phase2_seed123/net_last.pth` exists.

- [ ] **Step 5: Run the unmodified v2 evaluator entrypoint**

```bash
CUDA_VISIBLE_DEVICES=0 \
bash EVAL_msa_vae_metrics.sh \
  /tmp/msa-vae-repair-smoke/phase2_seed123/net_last.pth \
  --batch-size 32 \
  --num-workers 8 \
  --output-dir /tmp/msa-vae-repair-smoke/eval_seed123
```

Expected: exit 0; JSON/CSV/log contain finite FID, MPJPE, P-MPJPE, ACCEL,
Skating, T2M R@1/R@5/MedR, and M2T R@1/R@5/MedR, with protocol v2.

- [ ] **Step 6: Smoke the aggregator with three copied manifests whose training identities are test fixtures**

Use the aggregator unit-test fixtures or three explicit synthetic v2 manifests
in a temporary directory; do not misrepresent one trained checkpoint as three
formal seeds. Run:

```bash
bash AGGREGATE_msa_vae_metrics.sh \
  --variant smoke \
  --output-dir /tmp/msa-vae-repair-smoke/aggregate \
  /tmp/msa-vae-repair-smoke/fixture-seed123/metrics.json \
  /tmp/msa-vae-repair-smoke/fixture-seed456/metrics.json \
  /tmp/msa-vae-repair-smoke/fixture-seed789/metrics.json
```

Expected: exit 0 and all three aggregate artifacts exist.

- [ ] **Step 7: Inspect repository state and commit final validation-only adjustments if any**

```bash
git status --short
git log -7 --oneline
```

Expected: only the user's pre-existing untracked paths remain; implementation
files are committed in the task commits above.

