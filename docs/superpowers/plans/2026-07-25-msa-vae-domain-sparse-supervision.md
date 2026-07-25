# MSA-VAE Domain-Specific Sparse Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible BABEL sparse-global MSA-VAE reconstruction experiment while preserving the existing HumanML3D full-training behavior and checkpoint structure.

**Architecture:** Keep HumanML3D loading unchanged and add an isolated BABEL loader that combines a dual-supervised HumanML3D/BABEL bridge subset with local-only BABEL-stream samples. Build BABEL local SentenceT5 targets offline, normalize semantic losses by global valid counts under DDP, select Phase-1 checkpoints by semantic validation loss, and select Phase-2 checkpoints by reconstruction MPJPE.

**Tech Stack:** Python 3.8, PyTorch 2.4.1, Accelerate 1.0.1, NumPy, SentenceTransformers, `unittest`, Bash.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-25-msa-vae-domain-sparse-supervision-design.md`.
- Preserve the authoritative HumanML3D phase-1/phase-2 behavior and all existing model state-dict keys.
- Use `humanml3d_272/mean_std/` and the HumanML3D Causal TAE for `humanml_full`.
- Use `babel_272/t2m_babel_mean_std/` and `Experiments/causal_TAE_t2m_babel_272_h100_20260205/net_best_mpjpe.pth` for `babel_sparse_global`.
- Never mix HumanML3D and BABEL MSA-VAE checkpoints or normalization paths.
- Treat absent BABEL-only global supervision as valid; reject absent bridge global/local supervision and absent BABEL-only local supervision.
- Keep datasets, generated SentenceT5 caches, cache manifests, checkpoints, logs, and validation artifacts out of Git.
- Do not run full training, TMR evaluation, SentenceT5-XXL preprocessing, or GPU benchmarks during implementation verification.
- Preserve unrelated working-tree changes, including `open_flamingo` relocation and `paper writing/Research-Paper-Writing-Skills`.
- Use the existing `mgpt` environment and do not upgrade dependencies.
- Treat commit `747a904` as the implementation-review base.

---

### Task 1: Add the Offline BABEL-Stream SentenceT5 Cache Builder

**Files:**

- Create: `humanml3d_272/babel_stream_t5_cache.py`
- Create: `scripts/prepare_babel_stream_t5.py`
- Create: `tests/test_babel_stream_t5_cache.py`
- Modify: `.gitignore`

**Interfaces:**

- Produces: `CacheBuildError(rejections: tuple)`.
- Produces: `BabelStreamRecord(first_text: str, second_text: str, boundary: int)`.
- Produces: `parse_babel_stream_text(line: str) -> BabelStreamRecord`.
- Produces: `expand_segment_embeddings(record, motion_frames, embedding_by_text) -> numpy.ndarray`.
- Produces: `build_cache(split, motion_dir, text_dir, output_dir, encoder, model_signature, overwrite=False) -> dict`.
- Produces: `validate_cache_manifest(manifest_path, expected) -> dict`.
- CLI consumes: `--split`, `--motion-dir`, `--text-dir`, `--output-dir`, `--t5-model-path`, `--batch-size`, `--device`, and `--overwrite`.

- [ ] **Step 1: Write parsing and expansion tests**

Create `tests/test_babel_stream_t5_cache.py` with synthetic two-segment records:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanml3d_272.babel_stream_t5_cache import (
    BabelStreamRecord,
    expand_segment_embeddings,
    parse_babel_stream_text,
)


class BabelStreamT5CacheTest(unittest.TestCase):
    def test_parse_two_segment_record(self):
        line = (
            "throw#throw/VERB#0.0#0.0*"
            "catch#catch/VERB#0.0#0.0#3"
        )
        self.assertEqual(
            parse_babel_stream_text(line),
            BabelStreamRecord("throw", "catch", 3),
        )

    def test_expand_uses_boundary_and_exact_motion_length(self):
        record = BabelStreamRecord("walk", "sit", 2)
        embeddings = {
            "walk": np.array([1.0, 0.0], dtype=np.float32),
            "sit": np.array([0.0, 1.0], dtype=np.float32),
        }
        result = expand_segment_embeddings(record, 5, embeddings)
        np.testing.assert_array_equal(
            result,
            np.array(
                [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
                dtype=np.float32,
            ),
        )

    def test_parse_rejects_missing_segment_and_invalid_boundary(self):
        with self.assertRaisesRegex(ValueError, "two segments"):
            parse_babel_stream_text("walk#walk/VERB#0.0#0.0")
        with self.assertRaisesRegex(ValueError, "boundary"):
            parse_babel_stream_text(
                "walk#walk/VERB#0.0#0.0*sit#sit/VERB#0.0#0.0#bad"
            )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_babel_stream_t5_cache -v
```

Expected: import failure because `babel_stream_t5_cache.py` does not exist.

- [ ] **Step 3: Implement strict parsing and frame expansion**

Create immutable records and reject:

- blank action text;
- anything other than exactly two `*`-separated segments;
- missing or non-integer boundary;
- `boundary <= 0`;
- `boundary >= motion_frames`.

The expansion implementation must allocate float32 output and assign:

```python
output[:record.boundary] = embedding_by_text[record.first_text]
output[record.boundary:] = embedding_by_text[record.second_text]
```

- [ ] **Step 4: Add failing cache-build tests with an injected fake encoder**

Add a fake encoder that records its inputs:

```python
class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array(
            [[float(index), float(index + 1)] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )
```

Build two synthetic motions sharing an action string and assert:

- unique text is encoded once;
- each output length equals its motion length;
- manifest reports `valid_samples=2`, `rejected_samples=0`, dimension 2;
- rerunning without `overwrite` validates and skips existing files;
- malformed input returns a deterministic rejection report and raises before
  publishing the manifest.

- [ ] **Step 5: Implement atomic cache publication and manifest validation**

`build_cache` must:

1. discover exact matching `.npy`/`.txt` stems;
2. parse every text file before encoding;
3. load motion arrays with `mmap_mode="r"` and require `(T, 272)`;
4. encode sorted unique descriptions in caller-controlled batches;
5. write each array to a temporary file in the output directory and replace
   its final path only after shape validation;
6. write `manifest.json` last via temporary-path replacement.

The manifest includes:

```json
{
  "version": 1,
  "split": "train",
  "model_signature": "path-or-model-signature",
  "embedding_dim": 768,
  "motion_dir": "...",
  "text_dir": "...",
  "valid_samples": 2,
  "rejected_samples": 0,
  "records": {
    "seq_1": {"frames": 64, "text_sha256": "...", "motion_size": 69632}
  }
}
```

`validate_cache_manifest` checks version, split, model signature, embedding
dimension, source roots, exact record membership, each array shape, and input
signatures.

- [ ] **Step 6: Implement the thin SentenceT5 CLI**

`scripts/prepare_babel_stream_t5.py` imports the cache module, constructs one
`SentenceTransformer`, and passes a small adapter exposing `encode(texts)`.
Defaults are:

```text
train motion: babel_272_stream/train_stream
train text:   babel_272_stream/train_stream_text
train output: babel_272_stream/t5_enc_single/train
val motion:   babel_272_stream/val_stream
val text:     babel_272_stream/val_stream_text
val output:   babel_272_stream/t5_enc_single/val
model:        sentencet5-xxl/
```

Do not import SentenceTransformers from the testable core module.

- [ ] **Step 7: Ignore generated cache content**

Append exact rules:

```gitignore
babel_272_stream/t5_enc_single/
babel_272_stream/t5_enc_single/**
```

- [ ] **Step 8: Run focused tests and static checks**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_babel_stream_t5_cache -v
conda run -n mgpt python -m py_compile \
  humanml3d_272/babel_stream_t5_cache.py \
  scripts/prepare_babel_stream_t5.py \
  tests/test_babel_stream_t5_cache.py
git diff --check
```

Expected: tests pass without loading SentenceT5 or touching real data.

- [ ] **Step 9: Commit**

```bash
git add .gitignore \
  humanml3d_272/babel_stream_t5_cache.py \
  scripts/prepare_babel_stream_t5.py \
  tests/test_babel_stream_t5_cache.py
git commit -m "feat: add BABEL local target cache"
```

---

### Task 2: Add the BABEL Sparse-Global Dataset Contract

**Files:**

- Create: `humanml3d_272/dataset_msa_vae_babel.py`
- Create: `tests/test_dataset_msa_vae_babel.py`
- Modify: `options/option_msa_vae.py:11-45`

**Interfaces:**

- Consumes: validated train/val manifests from Task 1.
- Produces: `BabelSparseGlobalMSAVAEDataset(bridge_split_file: str, bridge_motion_dir: str, bridge_text_dir: str, bridge_global_embed_dir: str, bridge_local_embed_dir: str, babel_motion_dir: str, babel_cache_dir: str, babel_cache_manifest: str, mean_path: str, std_path: str, window_size: int, unit_length: int, text_embed_dim: int)`.
- Produces: `BabelSparseGlobalMSAVAEValidationDataset(babel_motion_dir: str, babel_cache_dir: str, babel_cache_manifest: str, mean_path: str, std_path: str, window_size: int, unit_length: int, text_embed_dim: int)`.
- Produces: `DATALoader(batch_size: int, num_workers: int, **dataset_kwargs) -> torch.utils.data.DataLoader`.
- Produces: `ValidationDATALoader(batch_size: int, num_workers: int, **dataset_kwargs) -> torch.utils.data.DataLoader`.
- Returns the existing MSA-VAE batch tuple:
  `(motion, caption, global_embed, has_global, local_embed, has_local, total_frames, local_pooled)`.

- [ ] **Step 1: Write synthetic source-contract tests**

Build temporary HumanML and BABEL roots with:

- one bridge motion, text file, global T5 array, and 20 FPS local T5 array;
- one BABEL motion, two-segment text file, exact-length 30 FPS local cache;
- 272-D Mean/Std arrays;
- a valid cache manifest.

Assert:

```python
bridge = dataset[dataset.index_for("hml:000001")]
self.assertTrue(bridge[3])   # has_global
self.assertTrue(bridge[5])   # has_local

babel = dataset[dataset.index_for("babel:seq_1")]
self.assertFalse(babel[3])   # has_global
self.assertTrue(babel[5])    # has_local
self.assertEqual(babel[0].shape, (64, 272))
self.assertEqual(babel[4].shape, (16, 768))
```

Also assert the bridge local target is upsampled from 20 FPS to motion length
before window slicing, while BABEL cache length must already match exactly.

- [ ] **Step 2: Run dataset tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_babel -v
```

Expected: import failure because the BABEL dataset module is absent.

- [ ] **Step 3: Implement separate bridge and BABEL source discovery**

Use source-prefixed internal names such as `hml:000001` and `babel:seq_1` to
prevent collisions. Require:

- bridge IDs from `train_ft.txt`;
- bridge motion length at least `window_size`;
- at least one full-motion HumanML caption;
- valid global and local arrays with configured dimension;
- BABEL IDs exactly equal to validated cache-manifest records;
- BABEL motion `(T, 272)` and cache `(T, D)`.

Do not wrap loading in a broad `except: pass`. Aggregate validation failures
and raise one `RuntimeError` listing counts and the first bounded set of IDs.

- [ ] **Step 4: Implement window sampling and the existing tuple contract**

For bridge samples:

- choose a HumanML global caption and matching global embedding row;
- upsample the 20 FPS local array to full motion length using the current
  `linspace` behavior;
- slice one shared motion/local window.

For BABEL samples:

- use caption `"None"` and a zero global vector;
- slice the same indices from motion and exact-length local cache.

Pool the local window to `window_size // unit_length` with the same
average-pooling convention as `dataset_msa_vae.py`.

The validation dataset is deterministic: enumerate non-overlapping
`window_size` windows and add one end-aligned tail window only when the final
frames were not covered. Never sample validation windows randomly.

- [ ] **Step 5: Add failure-boundary tests**

Assert construction fails for:

- missing bridge global target;
- missing bridge local target;
- missing BABEL local cache;
- BABEL cache/motion length mismatch;
- embedding dimension mismatch;
- cache manifest source-root mismatch;
- empty resulting dataset.

- [ ] **Step 6: Add explicit MSA data-mode and path options**

Add:

```python
parser.add_argument(
    "--msa_data_mode",
    choices=["humanml_full", "babel_sparse_global"],
    default="humanml_full",
)
parser.add_argument("--bridge_split_file", default="./humanml3d_272/split/train_ft.txt")
parser.add_argument("--bridge_motion_dir", default="./humanml3d_272/motion_data")
parser.add_argument("--bridge_text_dir", default="./humanml3d_272/texts")
parser.add_argument("--bridge_global_embed_dir", default="./humanml3d_272/text_latents_t5")
parser.add_argument("--bridge_local_embed_dir", default="./humanml3d_272/t5_enc_single")
parser.add_argument("--babel_train_motion_dir", default="./babel_272_stream/train_stream")
parser.add_argument("--babel_train_text_dir", default="./babel_272_stream/train_stream_text")
parser.add_argument("--babel_train_t5_cache_dir", default="./babel_272_stream/t5_enc_single/train")
parser.add_argument("--babel_train_cache_manifest", default="./babel_272_stream/t5_enc_single/train/manifest.json")
parser.add_argument("--babel_val_motion_dir", default="./babel_272_stream/val_stream")
parser.add_argument("--babel_val_text_dir", default="./babel_272_stream/val_stream_text")
parser.add_argument("--babel_val_t5_cache_dir", default="./babel_272_stream/t5_enc_single/val")
parser.add_argument("--babel_val_cache_manifest", default="./babel_272_stream/t5_enc_single/val/manifest.json")
parser.add_argument("--msa_mean_path", default="")
parser.add_argument("--msa_std_path", default="")
```

Empty mean/std options resolve from `msa_data_mode`; explicit overrides are
logged and stored in checkpoint metadata.

`babel_sparse_global` requires `text_encoder_type=t5` and
`text_embed_dim=768`; reject other values before dataset construction.
Training samples uniformly from the concatenated sequence entries through the
normal shuffled DataLoader, with no hidden source reweighting. Log exact
bridge and BABEL-only counts.

- [ ] **Step 7: Run dataset, parser, and existing regression tests**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_babel \
  tests.test_archive_exploration_results \
  tests.test_exploration_launchers -v
conda run -n mgpt python -m py_compile \
  humanml3d_272/dataset_msa_vae_babel.py \
  options/option_msa_vae.py \
  tests/test_dataset_msa_vae_babel.py
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add humanml3d_272/dataset_msa_vae_babel.py \
  options/option_msa_vae.py \
  tests/test_dataset_msa_vae_babel.py
git commit -m "feat: add BABEL sparse-global dataset"
```

---

### Task 3: Make Masked Alignment Globally Stable Under DDP

**Files:**

- Create: `utils/msa_vae_alignment.py`
- Create: `tests/test_msa_vae_alignment.py`
- Modify: `train_msa_vae.py:88-130`
- Modify: `train_msa_vae.py:410-509`

**Interfaces:**

- Produces: `AlignmentLossResult(backward_loss: torch.Tensor, global_mean: torch.Tensor, valid_count: torch.Tensor)`.
- Produces: `masked_cosine_sum_and_count(feat_a: torch.Tensor, feat_b: torch.Tensor, sample_mask: torch.Tensor) -> tuple`.
- Produces: `distributed_masked_cosine_alignment(feat_a: torch.Tensor, feat_b: torch.Tensor, sample_mask: torch.Tensor, accelerator) -> AlignmentLossResult`.
- Removes the training script's private `CLIPAlignmentLoss` implementation.

- [ ] **Step 1: Write single-rank masked-loss tests**

Cover `(B, D)` and `(B, T, D)` features. Compute expected values with:

```python
expected = (1.0 - torch.nn.functional.cosine_similarity(a_valid, b_valid)).mean()
```

Assert:

- invalid samples contribute no value or gradient;
- token count equals `valid_samples * T`;
- an empty mask returns differentiable zero and count zero.

- [ ] **Step 2: Write simulated DDP normalization tests**

Use a fake accelerator exposing:

```python
class FakeAccelerator:
    num_processes = 2

    def __init__(self, global_count, global_sum):
        self.global_count = torch.tensor(float(global_count))
        self.global_sum = torch.tensor(float(global_sum))

    def reduce(self, value, reduction="sum"):
        return self.global_count if value.ndim == 0 and value.dtype == torch.long else self.global_sum
```

Test two ranks where one has zero global samples and the other has two.
The average of rank-local backward-loss gradients must equal the gradient of
the concatenated global masked mean.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment -v
```

Expected: import failure because the alignment module does not exist.

- [ ] **Step 4: Implement sum/count and DDP scaling**

For each rank:

```python
local_sum, local_count = masked_cosine_sum_and_count(...)
global_count = accelerator.reduce(local_count.detach(), reduction="sum")
```

When `global_count > 0`, use:

```python
backward_loss = local_sum * accelerator.num_processes / global_count
global_sum = accelerator.reduce(local_sum.detach(), reduction="sum")
global_mean = global_sum / global_count
```

The `num_processes` factor compensates for DDP's gradient averaging. When the
count is zero, derive the zero from `feat_a.sum() * 0.0` to keep autograd
connected.

- [ ] **Step 5: Integrate the helper without changing objective weights**

In `compute_losses`:

- call the helper for global and local alignment;
- use `backward_loss` in `total_loss`;
- log `global_mean`;
- expose `global_valid_count`, `local_valid_count`,
  `global_valid_ratio`, and `local_valid_ratio`;
- keep Spotlight mixing and `has_local` alpha masking unchanged;
- keep HumanML full semantics unchanged.

Do not set `args.global_align_weight` or `args.local_align_weight` from batch
contents.

- [ ] **Step 6: Run focused and existing training tests**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment \
  tests.test_rag_training \
  tests.test_rag_training_ddp -v
conda run -n mgpt python -m py_compile \
  utils/msa_vae_alignment.py \
  train_msa_vae.py \
  tests/test_msa_vae_alignment.py
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add utils/msa_vae_alignment.py \
  train_msa_vae.py \
  tests/test_msa_vae_alignment.py
git commit -m "fix: normalize sparse alignment across ranks"
```

---

### Task 4: Add BABEL Reconstruction and Local-Alignment Validation

**Files:**

- Create: `utils/eval_msa_vae_babel.py`
- Create: `tests/test_eval_msa_vae_babel.py`
- Modify: `train_msa_vae.py:229-411`
- Modify: `train_msa_vae.py:541-600`
- Modify: `eval_msa_vae.py:22-120`

**Interfaces:**

- Consumes: `BabelSparseGlobalMSAVAEValidationDataset` from Task 2.
- Produces: `evaluate_msa_vae_babel(out_dir: str, val_loader, net, dataset, logger, writer, iteration: int, phase: int, best_semantic: float, best_mpjpe: float, device, accelerator, metadata: dict, save_checkpoints: bool = True) -> BabelEvalResult`.
- Produces: `BabelEvalResult(mpjpe: float, reconstruction: float, kl: float, latent: float, local_cosine: float, local_loss: float, global_coverage: float, local_coverage: float, semantic_objective: float, best_semantic: float, best_mpjpe: float)`.

- [ ] **Step 1: Write metric and checkpoint-selection tests**

Use a deterministic fake network returning:

- exact reconstruction for one test;
- a fixed perturbation for another;
- local features identical to normalized local targets.

Assert:

- exact reconstruction gives zero MPJPE;
- identical local features give cosine 1 and local loss 0;
- invalid local tokens do not affect the mean;
- Phase 1 updates `net_best_semantic.pth` only when
  `latent + local_align_weight * local_loss` improves;
- Phase 2 updates `net_best_mpjpe.pth` only when MPJPE improves;
- saved payload contains `net` plus metadata without changing state-dict keys.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_vae_babel -v
```

Expected: import failure because the evaluator does not exist.

- [ ] **Step 3: Implement reconstruction and semantic metrics**

The evaluator must:

- run under `torch.no_grad()`;
- reconstruct normalized motion;
- inverse-transform predictions and targets with the validation dataset's
  joint Mean/Std;
- recover joints with the existing 272-D recovery utility;
- aggregate sums/counts through `accelerator.reduce`;
- compute MPJPE in millimeters;
- compute reconstruction, KL, latent, masked local cosine, and local loss;
- restore the model's original train/eval mode.

Do not import or initialize the HumanML3D TMR evaluator.
Prepare the BABEL validation loader with `accelerator.prepare` so each rank
evaluates a disjoint shard. Only the main rank writes TensorBoard values or
checkpoints after all reductions complete.

- [ ] **Step 4: Save compatible checkpoints with metadata**

Use:

```python
payload = {
    "net": net.state_dict(),
    "metadata": dict(
        metadata,
        supervision_coverage=coverage,
    ),
}
```

Existing code that loads `checkpoint["net"]` remains compatible.
Before evaluation, the training entrypoint constructs `metadata` with exact
`msa_data_mode`, resolved mean/std paths, cache-manifest SHA-256 identity,
global/local loss weights, phase, and serialized training arguments.

- [ ] **Step 5: Branch training setup by data mode**

In `train_msa_vae.py`:

- `humanml_full` uses the current loader and current TMR evaluator path;
- `babel_sparse_global` uses Task 2 train/validation loaders and Task 4
  evaluator;
- resolve and log exact mean/std paths;
- fail before model construction if required cache/checkpoint assets are
  missing;
- do not import/load DistilBERT TMR weights in BABEL mode;
- use the semantic objective for Phase-1 checkpoint selection and BABEL MPJPE
  for Phase-2 checkpoint selection.

In `eval_msa_vae.py`:

- preserve the current HumanML evaluator path for `humanml_full`;
- load the BABEL validation dataset and call `evaluate_msa_vae_babel` with
  `save_checkpoints=False` for `babel_sparse_global`;
- validate checkpoint metadata mode and normalization paths before loading;
- refuse a HumanML checkpoint in BABEL mode and vice versa.

- [ ] **Step 6: Add mode-selection regression tests**

Without executing the script's training loop, test extracted setup helpers or
AST/source contracts to assert:

- HumanML mode still selects `dataset_msa_vae` and
  `evaluation_msa_vae_multi`;
- BABEL mode selects `dataset_msa_vae_babel` and
  `evaluate_msa_vae_babel`;
- BABEL mode does not initialize the TMR evaluator;
- checkpoint metadata records the selected mode and paths.
- standalone BABEL evaluation does not write or replace training checkpoints.

- [ ] **Step 7: Run evaluation, dataset, alignment, and legacy tests**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_vae_babel \
  tests.test_dataset_msa_vae_babel \
  tests.test_msa_vae_alignment \
  tests.test_rag_training \
  tests.test_rag_training_ddp -v
conda run -n mgpt python -m py_compile \
  train_msa_vae.py \
  eval_msa_vae.py \
  utils/eval_msa_vae_babel.py \
  tests/test_eval_msa_vae_babel.py
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add train_msa_vae.py eval_msa_vae.py \
  utils/eval_msa_vae_babel.py \
  tests/test_eval_msa_vae_babel.py
git commit -m "feat: evaluate BABEL MSA-VAE reconstruction"
```

---

### Task 5: Add Authoritative BABEL Launchers and Operational Documentation

**Files:**

- Create: `TRAIN_msa_vae_babel_phase1.sh`
- Create: `TRAIN_msa_vae_babel_phase2.sh`
- Create: `EVAL_msa_vae_babel.sh`
- Create: `tests/test_msa_vae_babel_launchers.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**

- Consumes: data mode, cache, dataset, and evaluator interfaces from Tasks
  1-4.
- Produces: reproducible user-facing preprocessing, phase-1, phase-2, and
  evaluation commands.

- [ ] **Step 1: Write launcher contract tests**

Assert:

- all three scripts enter the repository root;
- both training scripts pass `--msa_data_mode babel_sparse_global`;
- both use `--dataname t2m_babel_272`;
- phase 1 defaults to the existing joint Causal TAE checkpoint;
- phase 2 requires a BABEL phase-1 checkpoint;
- both pass joint mean/std and train/val cache manifests;
- BABEL experiment names contain `babel_sparse_global`;
- no BABEL launcher points to a HumanML MSA-VAE checkpoint;
- evaluation uses BABEL mode and BABEL validation paths.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_babel_launchers -v
```

Expected: failure because the launchers do not exist.

- [ ] **Step 3: Implement phase-1 launcher**

Use overridable environment variables:

```bash
NUM_GPUS=${1:-1}
BATCH_SIZE=$((128 / NUM_GPUS))
CNN_CKPT=${CNN_CKPT:-Experiments/causal_TAE_t2m_babel_272_h100_20260205/net_best_mpjpe.pth}
BABEL_T5_ROOT=${BABEL_T5_ROOT:-babel_272_stream/t5_enc_single}
```

Match the approved HumanML phase-1 architecture and loss weights unless a
documented experiment variable overrides them. Pass explicit joint Mean/Std,
cache, bridge, and BABEL-stream paths.

- [ ] **Step 4: Implement phase-2 and evaluation launchers**

Phase 2 requires:

```bash
PHASE1_DIR=${PHASE1_DIR:?"ERROR: set PHASE1_DIR to the BABEL sparse-global phase-1 directory"}
RESUME_PTH="${PHASE1_DIR}/net_best_semantic.pth"
```

Evaluation requires a BABEL MSA-VAE checkpoint and never falls back to a
HumanML checkpoint.

- [ ] **Step 5: Document the experiment**

Update README and AGENTS with:

- existing HumanML full sparse-local behavior;
- BABEL sparse-global purpose and dataset composition;
- offline cache command for train and val;
- exact training/evaluation order;
- normalization/checkpoint incompatibility warning;
- statement that BABEL results validate reconstruction/local alignment, not
  text-to-motion generation;
- local artifact policy for generated caches.

- [ ] **Step 6: Run launcher, syntax, and full unit tests**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_babel_launchers -v
bash -n TRAIN_msa_vae_babel_phase1.sh
bash -n TRAIN_msa_vae_babel_phase2.sh
bash -n EVAL_msa_vae_babel.sh
conda run -n mgpt python -m unittest discover -s tests -v
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add AGENTS.md README.md \
  TRAIN_msa_vae_babel_phase1.sh \
  TRAIN_msa_vae_babel_phase2.sh \
  EVAL_msa_vae_babel.sh \
  tests/test_msa_vae_babel_launchers.py
git commit -m "docs: add BABEL sparse-global workflow"
```

---

### Task 6: Run Final Cross-Mode Verification

**Files:**

- No new production files expected.
- Modify only scoped files if verification exposes a defect, using a failing
  regression test before each fix.

**Interfaces:**

- Consumes: all deliverables from Tasks 1-5.
- Produces: evidence that HumanML behavior is preserved and the new BABEL path
  is runnable up to the expensive-job boundary.

- [ ] **Step 1: Run the complete unit suite**

```bash
conda run -n mgpt python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Compile every changed Python file**

```bash
git diff --name-only 747a904..HEAD |
  rg '\.py$' |
  xargs conda run -n mgpt python -m py_compile
```

Expected: exit 0.

- [ ] **Step 3: Validate every changed Shell file**

```bash
while IFS= read -r file; do
  bash -n "$file"
done < <(git diff --name-only 747a904..HEAD | rg '\.sh$')
```

Expected: exit 0.

- [ ] **Step 4: Run a synthetic end-to-end CPU smoke check**

Using temporary data and a tiny fake text encoder:

1. build a two-sample BABEL cache;
2. construct one bridge plus one BABEL-only dataset;
3. collate one mixed batch;
4. run masked global/local alignment on synthetic features;
5. run the fake-network BABEL evaluator;
6. assert supervision flags, finite losses, and checkpoint metadata.

Do not load SentenceT5, CUDA, real checkpoints, or full datasets.

- [ ] **Step 5: Verify artifact and Git boundaries**

Run:

```bash
git diff --check
git status --short
git check-ignore -v babel_272_stream/t5_enc_single/train/example.npy
git diff --submodule=short
```

Confirm:

- no cache, checkpoint, log, event, dataset, or validation output is tracked;
- no `open_flamingo` or third-party submodule change is staged or committed;
- only intended source, test, launcher, and documentation files changed.

- [ ] **Step 6: Review the full implementation range**

Review against the approved spec:

```bash
git log --oneline --decorate 747a904..HEAD
git diff --stat 747a904..HEAD
git diff --check 747a904..HEAD
```

Check every global constraint and both data-mode contracts line by line. If a
check fails, stop final verification and return to the owning task with a
failing regression test; do not create an unreviewed catch-all commit.
