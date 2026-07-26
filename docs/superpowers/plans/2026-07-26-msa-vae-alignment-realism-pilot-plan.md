# MSA-VAE Alignment–Realism Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run four fresh, two-GPU, single-seed MSA-VAE variants through Phase 1, Phase 2, complete-test evaluation, and validated table collection for the reviewer-requested alignment–realism pilot.

**Architecture:** Keep the ablation under a focused `explorations/msa_vae_alignment_realism/` directory. A Python contract/collector is the single source of truth for variants and metric validation; small shell entrypoints perform durable per-variant training/evaluation inside named GNU Screen sessions. Existing official Phase-1/Phase-2 launchers and the standard MSA-VAE evaluator remain the execution cores.

**Tech Stack:** Bash, GNU Screen 4.09, Python 3.8.11, PyTorch 2.4.1+cu118, Accelerate 1.0.1, unittest, JSON/CSV/Markdown.

## Global Constraints

- Use the existing `mgpt` conda environment; do not upgrade dependencies.
- Use exactly two GPUs for each training variant and run four variants concurrently.
- GPU mapping is fixed: No Alignment=0,1; Global Only=2,3; Local Only=4,5; Global + Local=6,7.
- Use seed 123 only and label all results as a single-seed pilot.
- Train fresh Phase 1 and Phase 2 MSA-VAE checkpoints; never reuse an existing MSA-VAE checkpoint.
- Fix the TAE checkpoint to `Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth`.
- Require TAE SHA-256 `7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8`.
- Use Phase 1/2 budgets of 25,000 iterations and complete validation every 5,000 iterations.
- Use Phase-2 `net_last.pth` for the final table while retaining both best checkpoints.
- Disable automatic GIF/video rendering.
- Preserve the BABEL workflow and third-party evaluator source unchanged.
- Keep checkpoints, logs, Screen output, evaluation manifests, and table artifacts local and uncommitted.
- Preserve unrelated user changes and untracked exploration directories.

---

## File Map

- Create `explorations/msa_vae_alignment_realism/pilot.py`: variant contract, checkpoint/manifest validation, and single-seed CSV/Markdown table generation.
- Create `explorations/msa_vae_alignment_realism/run_variant.sh`: durable Phase 1→Phase 2 execution for one variant.
- Create `explorations/msa_vae_alignment_realism/eval_variant.sh`: durable standard evaluation for one completed variant.
- Create `explorations/msa_vae_alignment_realism/RUN_PILOT.sh`: preflight and four-session Screen launcher.
- Create `explorations/msa_vae_alignment_realism/STATUS_PILOT.sh`: non-mutating Screen/GPU/log/checkpoint status report.
- Create `explorations/msa_vae_alignment_realism/EVAL_PILOT.sh`: preflight and four-session evaluation launcher.
- Create `explorations/msa_vae_alignment_realism/README.md`: exact experiment, monitoring, evaluation, and collection commands.
- Create `tests/test_msa_vae_alignment_pilot.py`: contract, runner, preflight, and collector regression tests.
- Modify `tests/test_exploration_layout.py`: register the new intentional exploration directory and entrypoints.

### Task 1: Implement the Pilot Contract and Single-Seed Collector

**Files:**
- Create: `explorations/msa_vae_alignment_realism/pilot.py`
- Create: `tests/test_msa_vae_alignment_pilot.py`
- Modify: `tests/test_exploration_layout.py`

**Interfaces:**
- Produces immutable `PilotVariant` records with `label`, `slug`, `training_gpus`, `evaluation_gpu`, `screen_session`, Phase-1 weights, and Phase-2 weights.
- Produces `emit_contract(output_root: Path, format_name: str)`.
- Produces `validate_pilot_manifests(output_root: Path) -> list[dict]`.
- Produces `write_pilot_table(output_root: Path) -> dict[str, Path]`.
- CLI commands are `contract`, `verify`, and `collect`.

- [ ] **Step 1: Write failing contract tests**

Assert the exact matrix:

```python
expected = {
    "no_align": ((0.0, 0.0), (0.0, 0.0), "0,1", "0"),
    "global_only": ((0.2, 0.0), (0.05, 0.0), "2,3", "2"),
    "local_only": ((0.0, 0.2), (0.0, 0.05), "4,5", "4"),
    "global_local": ((0.2, 0.2), (0.05, 0.05), "6,7", "6"),
}
```

Also require:

```python
self.assertEqual(PILOT_SEED, 123)
self.assertEqual(PHASE_ITERATIONS, 25000)
self.assertEqual(EVAL_INTERVAL, 5000)
self.assertEqual(VALIDATION_SEED, 123)
self.assertEqual(VALIDATION_BATCH_SIZE, 32)
```

- [ ] **Step 2: Write failing collector tests**

Build four realistic `msa-vae-standard-v2` fixture manifests. Assert:

- all eleven requested metrics are written in the exact requested order;
- Markdown contains four raw rows and no `±`;
- CSV contains `seed=123` and raw numeric values;
- all manifests must share evaluator SHA, test sample hash/count, skating config, model structure, evaluation seed 123, and evaluation batch size 32;
- every checkpoint basename is `net_last.pth`;
- every checkpoint is Phase 2 with a Phase-1 parent;
- Phase 1/2 weights exactly match the variant contract;
- Phase 1/2 `total_iter=25000`, `eval_iter=5000`, `num_gpus=2`, `seed=123`, `use_ft_split=False`, and `msa_data_mode=humanml_full`;
- the fixed TAE path/hash appears in both-phase lineage;
- missing metrics, NaN/Inf, duplicate slugs, best-checkpoint paths, wrong weights, or identity mismatches fail closed.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_pilot \
  tests.test_exploration_layout
```

Expected: import/file failures because the pilot module and registered layout do not exist.

- [ ] **Step 4: Implement the contract and collector**

Use:

```python
TARGET_METRICS = (
    "fid",
    "mpjpe_mm",
    "p_mpjpe_mm",
    "accel_mm_per_frame2",
    "skating_percent",
    "t2m_r1_percent",
    "t2m_r5_percent",
    "t2m_medr",
    "m2t_r1_percent",
    "m2t_r5_percent",
    "m2t_medr",
)
```

The `contract --format tsv` output must have one tab-separated line per
variant:

```text
slug label training_gpus evaluation_gpu screen_session p1_global p1_local p2_global p2_local
```

The `contract --format json` output records the fixed TAE, git commit, output
root, seed, budgets, validation identity, and all four variants.

`collect` reads:

```text
<output_root>/evaluation/<slug>/metrics.json
```

and writes:

```text
<output_root>/summary/pilot_table.json
<output_root>/summary/pilot_table.csv
<output_root>/summary/pilot_table.md
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 3 command.

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add \
  explorations/msa_vae_alignment_realism/pilot.py \
  tests/test_msa_vae_alignment_pilot.py \
  tests/test_exploration_layout.py
git commit -m "feat: define MSA-VAE alignment pilot contract"
```

### Task 2: Add the Durable Per-Variant Training Runner

**Files:**
- Create: `explorations/msa_vae_alignment_realism/run_variant.sh`
- Modify: `tests/test_msa_vae_alignment_pilot.py`

**Interfaces:**
- Consumes positional fields emitted by `pilot.py contract --format tsv`.
- Consumes `PILOT_ROOT`, `TAE_CHECKPOINT`, `TAE_SHA256`, and optional test-only `PHASE1_LAUNCHER`/`PHASE2_LAUNCHER`.
- Produces Phase-1 and Phase-2 experiment directories and status files under `<output_root>/status/`.

- [ ] **Step 1: Write failing runner tests**

Invoke the runner with temporary fake Phase-1/Phase-2 launchers that record
their environment and create fake `net_last.pth` files. Assert:

```text
CUDA_VISIBLE_DEVICES=<assigned pair>
NUM_GPUS positional argument=2
SEED=123
TOTAL_ITER=25000
EVAL_ITER=5000
VALIDATION_SEED=123
VALIDATION_BATCH_SIZE=32
CNN_CKPT=<fixed path>
CNN_CKPT_SHA256=<fixed hash>
GLOBAL_ALIGN_WEIGHT/LOCAL_ALIGN_WEIGHT=<phase-specific contract>
PHASE1_DIR=<this variant's fresh Phase-1 directory>
```

Assert Phase 2 starts only after Phase 1 exits zero and creates
`net_last.pth`. A Phase-1 failure or missing checkpoint must produce a failed
status and must not invoke Phase 2.

- [ ] **Step 2: Run the runner tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_pilot.PilotRunnerTest
```

Expected: FAIL because `run_variant.sh` is absent.

- [ ] **Step 3: Implement the runner**

Use strict Bash mode and fixed names:

```text
<slug>_s123_phase1_25k_g<weight>_l<weight>
<slug>_s123_phase2_25k_g<weight>_l<weight>
```

Write status transitions atomically:

```text
phase1_running
phase1_complete
phase2_running
complete
failed
```

Use `trap` to record a nonzero exit. After each phase, require a nonempty
`net_last.pth` before continuing. Invoke the official launchers as:

```bash
CUDA_VISIBLE_DEVICES="$gpu_pair" \
OUT_DIR="$PILOT_ROOT" \
EXP_NAME="$phase1_name" \
SEED=123 \
TOTAL_ITER=25000 \
EVAL_ITER=5000 \
VALIDATION_SEED=123 \
VALIDATION_BATCH_SIZE=32 \
GLOBAL_ALIGN_WEIGHT="$phase1_global" \
LOCAL_ALIGN_WEIGHT="$phase1_local" \
CNN_CKPT="$TAE_CHECKPOINT" \
CNN_CKPT_SHA256="$TAE_SHA256" \
bash "$PHASE1_LAUNCHER" 2 t2m_272
```

Phase 2 uses the same visible pair and exact Phase-1 directory.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command.

Expected: all runner tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  explorations/msa_vae_alignment_realism/run_variant.sh \
  tests/test_msa_vae_alignment_pilot.py
git commit -m "feat: run durable MSA-VAE pilot variants"
```

### Task 3: Add Screen Launch, Status, and Evaluation Entrypoints

**Files:**
- Create: `explorations/msa_vae_alignment_realism/RUN_PILOT.sh`
- Create: `explorations/msa_vae_alignment_realism/STATUS_PILOT.sh`
- Create: `explorations/msa_vae_alignment_realism/EVAL_PILOT.sh`
- Create: `explorations/msa_vae_alignment_realism/eval_variant.sh`
- Create: `explorations/msa_vae_alignment_realism/README.md`
- Modify: `tests/test_msa_vae_alignment_pilot.py`
- Modify: `tests/test_exploration_layout.py`

**Interfaces:**
- `RUN_PILOT.sh` starts four detached named Screen sessions after complete preflight.
- `STATUS_PILOT.sh` is read-only and reports Screen, status, checkpoints, log tail, GPU utilization, and disk space.
- `EVAL_PILOT.sh` starts four detached one-GPU evaluation sessions only after all training variants are complete.
- `eval_variant.sh` evaluates one Phase-2 `net_last.pth` and writes one metrics manifest/status.

- [ ] **Step 1: Write failing launch-preflight tests**

With fake `screen` and `nvidia-smi` binaries, assert `RUN_PILOT.sh`:

- starts exactly four sessions with the approved names;
- passes GPU pairs `0,1`, `2,3`, `4,5`, and `6,7`;
- uses `screen -dmS`, logging, and the per-variant runner;
- rejects missing Screen, active named sessions, any GPU compute process,
  non-idle memory, missing/wrong-hash TAE, or pre-existing target phase
  directories;
- supports `PILOT_DRY_RUN=1`, which prints commands without starting sessions
  or creating experiment directories.

- [ ] **Step 2: Write failing evaluation tests**

Assert `EVAL_PILOT.sh` rejects incomplete training and non-`net_last.pth`
checkpoints. With fake completed checkpoints, assert it starts exactly four
Screen sessions on evaluation GPUs 0, 2, 4, and 6.

Assert `eval_variant.sh` invokes:

```bash
conda run -n mgpt python eval_msa_vae_metrics.py \
  <phase2/net_last.pth> \
  --data-root humanml3d_272 \
  --split-file humanml3d_272/split/test.txt \
  --evaluator-root Evaluator_272 \
  --output-dir <pilot_root>/evaluation/<slug> \
  --device cuda \
  --batch-size 32 \
  --num-workers 8 \
  --seed 123
```

- [ ] **Step 3: Run entrypoint tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_pilot \
  tests.test_exploration_layout
```

Expected: FAIL because Screen/status/evaluation entrypoints are absent.

- [ ] **Step 4: Implement launch and status entrypoints**

`RUN_PILOT.sh` must:

1. resolve the repository root and absolute pilot output root;
2. validate the fixed TAE path/hash;
3. validate Screen availability and session-name uniqueness;
4. require no GPU compute processes and at most 100 MiB used per GPU;
5. reject existing target phase directories;
6. write `contract.json`;
7. create logs/status directories;
8. start each session with an absolute runner path and absolute Screen log;
9. print exact attach, detach, and status commands.

`STATUS_PILOT.sh` must never write files or signal processes.

- [ ] **Step 5: Implement evaluation entrypoints and README**

`EVAL_PILOT.sh` verifies all `complete` training markers and checkpoint
metadata through `pilot.py verify` before starting evaluation sessions.

`eval_variant.sh` records `evaluation_running`, `evaluation_complete`, or
`evaluation_failed` and requires the resulting `metrics.json`.

Document:

```bash
bash explorations/msa_vae_alignment_realism/RUN_PILOT.sh
bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
screen -r msa_pilot_<variant>_s123
bash explorations/msa_vae_alignment_realism/EVAL_PILOT.sh
conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect
```

- [ ] **Step 6: Run focused tests and static checks**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_pilot \
  tests.test_exploration_layout
conda run -n mgpt python -m py_compile \
  explorations/msa_vae_alignment_realism/pilot.py \
  tests/test_msa_vae_alignment_pilot.py
bash -n \
  explorations/msa_vae_alignment_realism/RUN_PILOT.sh \
  explorations/msa_vae_alignment_realism/STATUS_PILOT.sh \
  explorations/msa_vae_alignment_realism/EVAL_PILOT.sh \
  explorations/msa_vae_alignment_realism/run_variant.sh \
  explorations/msa_vae_alignment_realism/eval_variant.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 7: Commit**

```bash
git add explorations/msa_vae_alignment_realism \
  tests/test_msa_vae_alignment_pilot.py \
  tests/test_exploration_layout.py
git commit -m "feat: orchestrate MSA-VAE alignment pilot"
```

### Task 4: Preflight and Launch Four Training Sessions

**Files:**
- Runtime output only:
  `Experiments/msa_vae_alignment_realism_pilot_s123_20260726/`

**Interfaces:**
- Consumes the committed pilot scripts and fixed local data/TAE assets.
- Produces four detached Screen sessions and local experiment artifacts.

- [ ] **Step 1: Run the full source verification suite**

```bash
conda run -n mgpt python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests pass before GPU execution.

- [ ] **Step 2: Verify assets and GPU state**

```bash
sha256sum \
  Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader
screen -ls
df -h Experiments
```

Expected: exact TAE hash, eight idle GPUs, no conflicting sessions, and
sufficient space.

- [ ] **Step 3: Run a no-write dry run**

```bash
PILOT_DRY_RUN=1 \
bash explorations/msa_vae_alignment_realism/RUN_PILOT.sh
```

Inspect all four variant names, weight pairs, GPU pairs, output directories,
Screen names, and exact Phase-1/Phase-2 commands.

- [ ] **Step 4: Start all four sessions**

```bash
bash explorations/msa_vae_alignment_realism/RUN_PILOT.sh
```

Immediately verify:

```bash
screen -ls
bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
nvidia-smi
```

Expected: four detached sessions and two training processes per GPU pair.

- [ ] **Step 5: Monitor until all Phase 1 and Phase 2 runs complete**

Poll status without blocking longer than 60 seconds per check:

```bash
bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
```

At every transition, verify no NaN/Inf, traceback, OOM, or NCCL error. Confirm
each Phase 1 produces `net_last.pth` before its paired Phase 2 begins.

### Task 5: Evaluate, Collect, and Analyze the Pilot Table

**Files:**
- Runtime output only:
  `Experiments/msa_vae_alignment_realism_pilot_s123_20260726/evaluation/`
  and `summary/`.

**Interfaces:**
- Consumes four completed Phase-2 `net_last.pth` checkpoints.
- Produces four standard `metrics.json` manifests and the requested raw
  single-seed table.

- [ ] **Step 1: Verify training completion**

Run:

```bash
conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py verify
bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
```

Expected: all four variants pass lineage, weight, TAE, seed, budget, and
checkpoint checks.

- [ ] **Step 2: Start four durable evaluation sessions**

```bash
bash explorations/msa_vae_alignment_realism/EVAL_PILOT.sh
```

Monitor with `screen -ls` and `STATUS_PILOT.sh` until four
`evaluation_complete` markers exist.

- [ ] **Step 3: Collect the table**

```bash
conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect
```

Expected artifacts:

```text
summary/pilot_table.json
summary/pilot_table.csv
summary/pilot_table.md
```

- [ ] **Step 4: Perform final protocol checks**

Verify:

- exactly 2,480 complete-test samples per variant;
- identical test sample hash, evaluator hash, model structure, skating
  thresholds, evaluation seed, and batch size;
- no GIF/video artifacts;
- only Phase-2 `net_last.pth` sources;
- eleven finite metrics in all four rows.

- [ ] **Step 5: Analyze the alignment–realism trade-off**

Report:

1. the four-row Markdown table;
2. deltas from No Alignment for every realism and retrieval metric;
3. Global Only versus Local Only under equal nominal weights;
4. whether Global + Local improves bidirectional retrieval while degrading
   FID, reconstruction, acceleration, or skating;
5. a concise reviewer-facing interpretation explicitly qualified as
   single-seed pilot evidence;
6. whether seeds 456/789 are warranted before manuscript submission.
