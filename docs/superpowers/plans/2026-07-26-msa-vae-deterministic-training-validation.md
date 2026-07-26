# MSA-VAE Deterministic Training Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HumanML3D MSA-VAE training-time validation with the deterministic complete-motion validation protocol while preserving both best checkpoints and isolating training RNG state.

**Architecture:** Add a focused validation utility that wraps the existing standard evaluator core with RNG/model-mode isolation and checkpoint selection. Wire `train_msa_vae.py` to a deterministic, unsharded complete-motion `val.txt` loader on the main rank, use dynamic TMR masks, and retain `net_best_fid.pth`, `net_best_mpjpe.pth`, and `net_last.pth`. Keep the BABEL validation workflow and final standalone test evaluator unchanged.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1+cu118, Accelerate 1.0.1, NumPy, unittest, Bash.

## Global Constraints

- Use the existing `mgpt` conda environment; do not upgrade dependencies.
- HumanML3D uses the 272-D representation and the existing Mean/Std files.
- The fixed TAE remains `Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth`.
- One formal training experiment uses exactly two GPUs; never launch eight-card training.
- The BABEL sparse-global validation path remains unchanged.
- Both phases continue saving `net_best_fid.pth`, `net_best_mpjpe.pth`, and `net_last.pth`.
- Formal paper aggregation continues accepting only the fixed-iteration Phase-2 `net_last.pth`.
- Training-time numeric validation renders no GIFs or TensorBoard videos.
- Do not start a full training run. The only GPU execution is one deterministic validation diagnostic on an existing smoke checkpoint.
- Preserve unrelated worktree changes and do not edit third-party evaluator source.

---

## File Map

- Create `utils/msa_vae_validation.py`: RNG isolation, deterministic metric execution, logging, TensorBoard scalar publication, and best/last checkpoint selection.
- Modify `options/option_msa_vae.py`: add explicit internal validation seed and batch-size arguments.
- Modify `utils/msa_vae_training.py`: record internal validation identity in checkpoint metadata.
- Reuse `humanml3d_272/dataset_eval_msa_vae_metrics.py` unchanged for its deterministic dataset and loader.
- Modify `train_msa_vae.py`: construct the complete-motion val loader, use dynamic TMR masks, call the new validation utility only on the main rank, and remove the legacy visualization-capable validation call.
- Modify `TRAIN_msa_vae_phase1.sh` and `TRAIN_msa_vae_phase2.sh`: pass and print the fixed internal validation contract.
- Modify `aggregate_msa_vae_metrics.py`: require valid Phase-1 and Phase-2 internal validation identity.
- Create `tests/test_msa_vae_deterministic_validation.py`: utility behavior and checkpoint-selection tests.
- Modify `tests/test_msa_vae_training_entrypoint.py`: checkpoint metadata tests.
- Modify `tests/test_msa_vae_full_sequence_launchers.py`: launcher contract tests.
- Modify `tests/test_aggregate_msa_vae_metrics.py`: fail-closed aggregation tests.
- Modify `tests/test_eval_msa_vae_metrics.py`: reuse guarantees for the standard metric core where needed.

---

### Task 1: Record the Internal Validation Contract

**Files:**
- Modify: `options/option_msa_vae.py`
- Modify: `utils/msa_vae_training.py`
- Modify: `TRAIN_msa_vae_phase1.sh`
- Modify: `TRAIN_msa_vae_phase2.sh`
- Test: `tests/test_msa_vae_training_entrypoint.py`
- Test: `tests/test_msa_vae_full_sequence_launchers.py`

**Interfaces:**
- Produces CLI fields `args.validation_seed: int` and `args.validation_batch_size: int`.
- Produces checkpoint fields `training_args.validation_seed` and `training_args.validation_batch_size`.
- Launcher defaults are `VALIDATION_SEED=123` and `VALIDATION_BATCH_SIZE=32`.

- [ ] **Step 1: Write failing option, metadata, and launcher tests**

Add parser assertions:

```python
self.assertEqual(args.validation_seed, 123)
self.assertEqual(args.validation_batch_size, 32)
```

Extend `CheckpointMetadataTest._args()`:

```python
"validation_seed": 123,
"validation_batch_size": 32,
```

Assert the fields are persisted:

```python
self.assertEqual(metadata["training_args"]["validation_seed"], 123)
self.assertEqual(metadata["training_args"]["validation_batch_size"], 32)
```

Extend both launcher tests to assert:

```python
self.assertEqual(
    self._value_after(arguments, "--validation-seed"),
    "123",
)
self.assertEqual(
    self._value_after(arguments, "--validation-batch-size"),
    "32",
)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_full_sequence_launchers
```

Expected: FAIL because the parser, metadata, and launchers do not expose the two fields.

- [ ] **Step 3: Add the parser and metadata fields**

In `options/option_msa_vae.py`, add:

```python
parser.add_argument(
    "--validation-seed",
    type=int,
    default=123,
    help="fixed RNG seed for deterministic internal validation",
)
parser.add_argument(
    "--validation-batch-size",
    type=int,
    default=32,
    help="batch size for deterministic complete-motion validation",
)
```

Reject invalid values after parsing:

```python
if args.validation_batch_size < 1:
    parser.error("--validation-batch-size must be positive")
```

Add both names to `TRAINING_IDENTITY_FIELDS` in
`utils/msa_vae_training.py`.

- [ ] **Step 4: Pass explicit values from both launchers**

In each launcher, resolve and print:

```bash
VALIDATION_SEED=${VALIDATION_SEED:-123}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
```

Pass:

```bash
--validation-seed "$VALIDATION_SEED" \
--validation-batch-size "$VALIDATION_BATCH_SIZE" \
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add options/option_msa_vae.py utils/msa_vae_training.py \
  TRAIN_msa_vae_phase1.sh TRAIN_msa_vae_phase2.sh \
  tests/test_msa_vae_training_entrypoint.py \
  tests/test_msa_vae_full_sequence_launchers.py
git commit -m "feat: record deterministic MSA-VAE validation identity"
```

---

### Task 2: Add RNG- and Model-Mode-Isolated Validation

**Files:**
- Create: `utils/msa_vae_validation.py`
- Create: `tests/test_msa_vae_deterministic_validation.py`

**Interfaces:**
- Produces `isolated_validation_rng(seed: int)` as a context manager.
- Produces `run_deterministic_msa_validation(model, evaluator, loader, device, seed, skating_config=None) -> dict`.
- Consumes `eval_msa_vae_metrics.evaluate_msa_vae_metrics`.

- [ ] **Step 1: Write failing RNG restoration tests**

Create `tests/test_msa_vae_deterministic_validation.py` with probes for Python,
NumPy, and PyTorch:

```python
def _probe_rng():
    return (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

def test_rng_guard_restores_states_after_success(self):
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    expected = _probe_rng()

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    with isolated_validation_rng(123):
        _probe_rng()
        _probe_rng()
    actual = _probe_rng()

    self.assertEqual(actual[0], expected[0])
    self.assertEqual(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])
```

Add an exception-path version:

```python
with self.assertRaisesRegex(RuntimeError, "diagnostic"):
    with isolated_validation_rng(123):
        _probe_rng()
        raise RuntimeError("diagnostic")
```

When CUDA is available, compare every tensor returned by
`torch.cuda.get_rng_state_all()` before and after both paths. Also assert
`torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark`
return to their original values.

- [ ] **Step 2: Run the RNG tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_deterministic_validation
```

Expected: ERROR importing the missing validation utility.

- [ ] **Step 3: Implement the RNG context manager**

Create `utils/msa_vae_validation.py`:

```python
import contextlib
import random

import numpy as np
import torch


@contextlib.contextmanager
def isolated_validation_rng(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
```

- [ ] **Step 4: Add a failing deterministic wrapper test**

Use the existing tiny dataset/model/evaluator fixtures or local equivalents.
Make the fake model consume `torch.randn_like` and record its initial training
mode. Assert:

```python
first = run_deterministic_msa_validation(
    model, evaluator, loader, torch.device("cpu"), seed=123
)
second = run_deterministic_msa_validation(
    model, evaluator, loader, torch.device("cpu"), seed=123
)
self.assertEqual(first, second)
self.assertTrue(model.training)
```

Add an evaluator failure and assert the model returns to its original training
mode and the caller RNG probe remains unchanged.

- [ ] **Step 5: Run the wrapper test and verify RED**

Run the Step 2 command.

Expected: FAIL because `run_deterministic_msa_validation` is missing.

- [ ] **Step 6: Implement the deterministic wrapper**

Add:

```python
def run_deterministic_msa_validation(
    model,
    evaluator,
    loader,
    device,
    seed,
    skating_config=None,
):
    from eval_msa_vae_metrics import evaluate_msa_vae_metrics
    from utils.msa_vae_metrics import SkatingConfig

    was_training = model.training
    try:
        with isolated_validation_rng(seed):
            return evaluate_msa_vae_metrics(
                model,
                evaluator,
                loader,
                device,
                skating_config or SkatingConfig(),
            )
    finally:
        model.train(was_training)
```

- [ ] **Step 7: Run the utility tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass on CPU; CUDA assertions run when available.

- [ ] **Step 8: Commit**

```bash
git add utils/msa_vae_validation.py \
  tests/test_msa_vae_deterministic_validation.py
git commit -m "feat: isolate deterministic MSA-VAE validation"
```

---

### Task 3: Preserve Best and Last Checkpoints Without Visualization

**Files:**
- Modify: `utils/msa_vae_validation.py`
- Modify: `tests/test_msa_vae_deterministic_validation.py`

**Interfaces:**
- Produces immutable `MSAValidationState(best_fid: float, best_mpjpe: float)`.
- Produces `publish_msa_validation(result, iteration, out_dir, model, metadata, state, logger, writer, validation_seed, validation_batch_size) -> MSAValidationState`.
- Consumes `utils.msa_vae_training.save_msa_checkpoint`.

- [ ] **Step 1: Write failing checkpoint publication tests**

Use a temporary directory, `torch.nn.Linear`, in-memory logger, and a writer
stub whose `add_scalar` records calls.

First result:

```python
result = {
    "sample_count": 802,
    "fid": 0.95,
    "mpjpe_mm": 21.6,
    "p_mpjpe_mm": 15.9,
    "accel_mm_per_frame2": 4.6,
    "skating_percent": 8.0,
    "t2m_r1_percent": 13.0,
    "t2m_r2_percent": 26.0,
    "t2m_r3_percent": 34.0,
    "t2m_r5_percent": 47.0,
    "t2m_medr": 6.0,
    "m2t_r1_percent": 19.0,
    "m2t_r2_percent": 27.0,
    "m2t_r3_percent": 38.0,
    "m2t_r5_percent": 48.0,
    "m2t_medr": 6.0,
}
```

For both Phase-1 and Phase-2 metadata, assert all three files exist after
iteration 0 and contain the supplied metadata. Mutate the model, publish a
worse result, and assert best-file SHA-256 values do not change while
`net_last.pth` does. Publish a strict improvement in only one metric and assert
only its best file changes.

Assert scalar tags start with `CompleteVal/`. Patch
`utils.eval_trans.tensorborad_add_video_xyz` and assert numeric validation
never calls it.

- [ ] **Step 2: Run the publication tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_deterministic_validation
```

Expected: FAIL because state and publication interfaces are missing.

- [ ] **Step 3: Implement result validation and state**

Add:

```python
from dataclasses import dataclass
import math
import os

from utils.msa_vae_training import save_msa_checkpoint


@dataclass(frozen=True)
class MSAValidationState:
    best_fid: float = float("inf")
    best_mpjpe: float = float("inf")
```

Require every numeric value to be finite and require `sample_count > 0` before
writing any checkpoint.

- [ ] **Step 4: Implement checkpoint publication**

Implement strict `<` comparisons, log the complete validation protocol, write
all numeric values to TensorBoard, then save:

```python
logger.info(
    "Complete deterministic val: "
    f"iteration={iteration} samples={result['sample_count']} "
    f"seed={validation_seed} batch_size={validation_batch_size} "
    f"FID={result['fid']:.6f} "
    f"MPJPE={result['mpjpe_mm']:.3f}mm"
)
if result["fid"] < state.best_fid:
    save_msa_checkpoint(
        os.path.join(out_dir, "net_best_fid.pth"),
        model,
        metadata,
    )
if result["mpjpe_mm"] < state.best_mpjpe:
    save_msa_checkpoint(
        os.path.join(out_dir, "net_best_mpjpe.pth"),
        model,
        metadata,
    )
save_msa_checkpoint(
    os.path.join(out_dir, "net_last.pth"),
    model,
    metadata,
)
```

Return the updated immutable state.

- [ ] **Step 5: Run publication tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass and the visualization rendering helper is never
called.

- [ ] **Step 6: Commit**

```bash
git add utils/msa_vae_validation.py \
  tests/test_msa_vae_deterministic_validation.py
git commit -m "feat: publish deterministic MSA-VAE validation checkpoints"
```

---

### Task 4: Wire Complete-Motion Validation Into Training

**Files:**
- Modify: `train_msa_vae.py`
- Modify: `tests/test_msa_vae_training_entrypoint.py`
- Modify: `tests/test_exploration_layout.py`

**Interfaces:**
- Consumes `MSAVAEMetricsDataset`, `make_msa_vae_metrics_loader`.
- Consumes `run_deterministic_msa_validation`, `publish_msa_validation`, and `MSAValidationState`.
- Removes HumanML training use of `dataset_eval_t2m.DATALoader` and `evaluation_msa_vae_multi`.

- [ ] **Step 1: Write failing source-contract tests**

Add AST/text contract assertions that HumanML training:

```python
self.assertIn("MSAVAEMetricsDataset", source)
self.assertIn("make_msa_vae_metrics_loader", source)
self.assertIn("run_deterministic_msa_validation", source)
self.assertIn("publish_msa_validation", source)
self.assertNotIn("dataset_eval_t2m.DATALoader(", source)
self.assertNotIn("evaluation_msa_vae_multi(", source)
self.assertNotIn("tensorborad_add_video_xyz(", training_source)
```

Also assert the training TMR encoder is constructed with `max_len=-1`.

- [ ] **Step 2: Run the source-contract tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint \
  tests.test_exploration_layout
```

Expected: FAIL because training still uses the legacy loader and evaluator.

- [ ] **Step 3: Construct deterministic HumanML validation data**

Replace the legacy loader with:

```python
if accelerator.is_main_process:
    validation_dataset = MSAVAEMetricsDataset(
        "humanml3d_272",
        split_file="humanml3d_272/split/val.txt",
        unit_length=unit_length,
    )
    validation_loader = make_msa_vae_metrics_loader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        num_workers=8,
        pin_memory=comp_device.type == "cuda",
    )
else:
    validation_dataset = None
    validation_loader = None
```

Log:

```python
logger.info(
    "Deterministic complete validation: split=val "
    f"samples={len(validation_dataset)} "
    f"sample_hash={validation_dataset.sample_hash} "
    f"seed={args.validation_seed} "
    f"batch_size={args.validation_batch_size}"
)
```

Do not pass this loader through `accelerator.prepare`.

- [ ] **Step 4: Use dynamic TMR masks**

Construct the training-time frozen TMR motion encoder with:

```python
max_len=-1
```

Keep the evaluator checkpoint, text encoder, weights, and normalization
unchanged.

- [ ] **Step 5: Replace both HumanML validation call sites**

Initialize:

```python
validation_state = MSAValidationState()
```

At iteration 0 and every `args.eval_iter`, synchronize all ranks. On the main
rank only:

```python
eval_model = accelerator.unwrap_model(net)
result = run_deterministic_msa_validation(
    eval_model,
    evaluator,
    validation_loader,
    comp_device,
    seed=args.validation_seed,
)
validation_state = publish_msa_validation(
    result=result,
    iteration=nb_iter,
    out_dir=args.out_dir,
    model=eval_model,
    metadata=checkpoint_metadata,
    state=validation_state,
    logger=logger,
    writer=writer,
    validation_seed=args.validation_seed,
    validation_batch_size=args.validation_batch_size,
)
```

Synchronize again after publication. Leave BABEL calls untouched.

Remove `EvalCompat`, `net_eval`, the HumanML call to
`evaluation_msa_vae_multi`, and all numeric-validation visualization flags.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 2 command plus:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_deterministic_validation \
  tests.test_eval_msa_vae_metrics
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add train_msa_vae.py \
  tests/test_msa_vae_training_entrypoint.py \
  tests/test_exploration_layout.py
git commit -m "fix: use deterministic complete MSA-VAE validation"
```

---

### Task 5: Fail Closed on Validation-Protocol Mismatch

**Files:**
- Modify: `aggregate_msa_vae_metrics.py`
- Modify: `tests/test_aggregate_msa_vae_metrics.py`

**Interfaces:**
- Requires positive integer `validation_batch_size`.
- Requires integer `validation_seed`.
- Requires both values in Phase-1 and Phase-2 checkpoint metadata.
- Requires the same validation seed and batch size across all three training seeds through canonical configuration comparison.
- Requires every formal manifest checkpoint path to end in `net_last.pth`.

- [ ] **Step 1: Write failing aggregation tests**

Add both fields to the valid Phase-1 and Phase-2 fixtures:

```python
"validation_seed": 123,
"validation_batch_size": 32,
```

For each phase, delete each field and expect
`official two-stage protocol`. Also test invalid values:

```python
("validation_seed", True),
("validation_seed", 1.5),
("validation_batch_size", 0),
("validation_batch_size", True),
("validation_batch_size", 1.5),
```

Mutate one manifest to a different valid seed or batch size and expect
`training configuration mismatch`.

Change a manifest checkpoint path to `net_best_fid.pth` or
`net_best_mpjpe.pth` and expect `net_last.pth`.

- [ ] **Step 2: Run aggregation tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_aggregate_msa_vae_metrics
```

Expected: FAIL because old/incomplete checkpoint metadata is still accepted.

- [ ] **Step 3: Require the fields in both phases**

Extend `_validate_official_two_stage_protocol` with a focused helper:

```python
def _valid_validation_identity(training_args):
    validation_seed = training_args.get("validation_seed")
    validation_batch_size = training_args.get("validation_batch_size")
    return (
        isinstance(validation_seed, int)
        and not isinstance(validation_seed, bool)
        and isinstance(validation_batch_size, int)
        and not isinstance(validation_batch_size, bool)
        and validation_batch_size > 0
    )
```

Require it for current Phase 2 and parent Phase 1 metadata. Canonical training
configuration comparison already rejects cross-seed mismatches.

Before aggregating metrics, require:

```python
checkpoint_path = _nested(
    manifest,
    ("checkpoint", "path"),
    "checkpoint",
)
if Path(checkpoint_path).name != "net_last.pth":
    raise ValueError("formal aggregation requires Phase 2 net_last.pth")
```

- [ ] **Step 4: Run aggregation tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add aggregate_msa_vae_metrics.py \
  tests/test_aggregate_msa_vae_metrics.py
git commit -m "fix: enforce MSA-VAE validation protocol identity"
```

---

### Task 6: Full Verification and One-GPU Diagnostic

**Files:**
- Verify all changed Python and shell files.
- Diagnostic output only: `/tmp/msa-vae-pipeline-smoke.9oL8iR/deterministic_internal_val`

**Interfaces:**
- Consumes the existing Phase-1 smoke checkpoint:
  `/tmp/msa-vae-pipeline-smoke.9oL8iR/phase1_smoke/net_last.pth`.
- Uses one GPU for evaluation only; no training is launched.

- [ ] **Step 1: Run the complete unit suite**

```bash
conda run -n mgpt python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 2: Run required static checks**

```bash
conda run -n mgpt python -m py_compile \
  aggregate_msa_vae_metrics.py \
  eval_msa_vae_metrics.py \
  train_msa_vae.py \
  options/option_msa_vae.py \
  utils/msa_vae_training.py \
  utils/msa_vae_validation.py \
  tests/test_msa_vae_deterministic_validation.py \
  tests/test_msa_vae_training_entrypoint.py \
  tests/test_msa_vae_full_sequence_launchers.py \
  tests/test_aggregate_msa_vae_metrics.py
bash -n AGGREGATE_msa_vae_metrics.sh \
  TRAIN_msa_vae_phase1.sh \
  TRAIN_msa_vae_phase2.sh
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Run a one-GPU deterministic validation diagnostic**

Use GPU 0 only:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mgpt python \
  eval_msa_vae_metrics.py \
  /tmp/msa-vae-pipeline-smoke.9oL8iR/phase1_smoke/net_last.pth \
  --data-root \
  /share/home/tm878032203900000/a878044490/MotionStreamer/humanml3d_272 \
  --split-file \
  /share/home/tm878032203900000/a878044490/MotionStreamer/humanml3d_272/split/val.txt \
  --evaluator-root \
  /share/home/tm878032203900000/a878044490/MotionStreamer/Evaluator_272 \
  --output-dir \
  /tmp/msa-vae-pipeline-smoke.9oL8iR/deterministic_internal_val \
  --device cuda \
  --batch-size 32 \
  --num-workers 8 \
  --seed 123
```

Expected:

- 802 samples;
- FID in the previously observed deterministic validation range near 0.95;
- MPJPE near 21--22 mm;
- no GIF files;
- JSON records seed 123 and batch size 32.

- [ ] **Step 4: Inspect repository and artifact state**

```bash
git status --short
find /tmp/msa-vae-pipeline-smoke.9oL8iR/deterministic_internal_val \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: the worktree is clean after commits; diagnostic artifacts contain
only numeric evaluation outputs and no GIF/video files.

- [ ] **Step 5: Request final code review**

Review the complete range from `8a48b56` through the final implementation
commit for correctness, deterministic RNG restoration, distributed safety,
checkpoint selection, and BABEL non-regression. Resolve all Important
findings before completion.
