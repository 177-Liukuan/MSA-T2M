# MSA-T2M Mainline Training Pipeline Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the official global-RAG DDPM training path onto a packed, precomputed input pipeline and a correct single-forward DDP boundary without changing retrieval semantics, the loss, checkpoint fields, or evaluation.

**Architecture:** A standalone cache module packs the static motion, caption, and `[CLS]` inputs and stores per-caption Top-K retrieval outputs behind a validated manifest. The dataset retains an explicit reference mode and adds a packed mode. A focused `nn.Module` owns the unchanged RAG model and computes the complete two-forward loss so Accelerate/DDP observes exactly one gradient-bearing module forward per step.

**Tech Stack:** Python 3.8, NumPy, PyTorch 2.4, `torch.distributed`/Gloo, Hugging Face Accelerate 1.0, Bash, `unittest`

## Global Constraints

- The official method remains global `[CLS]` single-token RAG with the DDPM autoregressive head.
- Minor floating-point rounding and sample-order differences are allowed; retrieval semantics, training objectives, checkpoint structure, and final evaluation metrics must remain equivalent within normal statistical variation.
- Do not change MSA-VAE, CFG dropout, EMA equations, caption sampling probabilities, Top-K fusion, attention internals, inference, or evaluation.
- Keep `trans`, `rag`, `trans_ema`, and `rag_ema` state-dict parameter names and tensor shapes compatible with existing checkpoints.
- Packed mode must reject stale or incompatible caches and must never silently fall back to online retrieval.
- Do not launch a full training run or final TMR evaluation during code verification.
- Preserve the unrelated modification in `paper writing/Research-Paper-Writing-Skills`.

## File Structure

- `humanml3d_272/msa_rag_cache.py`: source discovery, manifest generation and validation, atomic cache building, and memory-mapped cache access.
- `build_msa_rag_cache.py`: user-facing cache build/validation CLI.
- `humanml3d_272/dataset_msa_rag.py`: reference and packed dataset modes plus unchanged collation.
- `models/rag_training.py`: DDP-visible two-forward loss module and model-unwrapping helper.
- `train_t2m_rag.py`: argument plumbing, packed loader selection, DDP-safe wrapper use, nonblocking transfer, EMA/checkpoint access, and single loss synchronization.
- `TRAIN_t2m_rag.sh`: cache preflight and overridable official launcher configuration.
- `.gitignore`: generated MSA-RAG cache directory.
- `README.md`: official cache build/use and reference rollback instructions.
- `tests/msa_rag_fixtures.py`: deterministic small source-feature fixture shared by cache and dataset tests.
- `tests/test_msa_rag_cache.py`: cache values, validation, and invalidation.
- `tests/test_dataset_msa_rag.py`: reference/packed behavioral equivalence.
- `tests/test_rag_training.py`: two-forward loss/gradient and checkpoint-name equivalence.
- `tests/test_rag_training_ddp.py`: two-process DDP gradient synchronization.
- `tests/test_train_t2m_rag_launcher.py`: launcher preflight ordering and reference-mode behavior.

---

### Task 1: Build and Validate a Packed Global-RAG Cache

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/msa_rag_fixtures.py`
- Create: `tests/test_msa_rag_cache.py`
- Create: `humanml3d_272/msa_rag_cache.py`
- Create: `build_msa_rag_cache.py`

**Interfaces:**
- Produces: `CacheValidationError(ValueError)`.
- Produces: `build_cache(dataset_name: str, motion_latent_dir: str, text_latent_dir: str, hcls_dir: str, cache_dir: str, topk: int, text_embed_dim: int, exclude_self: bool = True, force: bool = False, retrieval_batch_size: int = 256) -> dict`.
- Produces: `validate_cache(cache_dir: str, dataset_name: str, motion_latent_dir: str, text_latent_dir: str, hcls_dir: str, topk: int, text_embed_dim: int, exclude_self: bool = True) -> dict`.
- Produces: `PackedMSARAGCache(cache_dir: str, requested_topk: int)` with `sample_ids`, `__len__()`, and `get(sample_idx: int, caption_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`.
- Produces CLI: `python build_msa_rag_cache.py --dataset-name ... --motion-latent-dir ... --text-latent-dir ... --hcls-dir ... --cache-dir ... --topk ... --text-embed-dim ... [--retrieval-batch-size 256] [--force] [--validate-only]`.

- [ ] **Step 1: Add a deterministic three-sample feature fixture**

Create `tests/msa_rag_fixtures.py` with `create_rag_fixture(root: pathlib.Path) -> dict`. It writes:

```python
sample_ids = ["000001", "000002", "000003"]
texts = {
    "000001": np.array([[1., 0.], [0.8, 0.2]], np.float32),
    "000002": np.array([[0., 1.]], np.float32),
    "000003": np.array([[0.6, 0.8], [-1., 0.]], np.float32),
}
hcls = {
    "000001": np.array([1., 0.], np.float32),
    "000002": np.array([0., 1.], np.float32),
    "000003": np.array([0.6, 0.8], np.float32),
}
motions = {
    "000001": np.arange(8, dtype=np.float32).reshape(4, 2),
    "000002": np.arange(6, dtype=np.float32).reshape(3, 2),
    "000003": np.arange(10, dtype=np.float32).reshape(5, 2),
}
```

It creates `humanml3d_272/split/train.txt` and separate motion, text, and h_cls directories, then returns their paths. Literal vectors make expected self-excluded neighbors and scores hand-checkable.

- [ ] **Step 2: Write failing cache-value and shape tests**

Create `tests/test_msa_rag_cache.py` using `tempfile.TemporaryDirectory` and the real fixture. Add tests which call the missing `build_cache` and `PackedMSARAGCache`, then assert:

```python
self.assertEqual(cache.sample_ids, ["000001", "000002", "000003"])
text, top_hcls, scores, motion = cache.get(sample_idx=0, caption_idx=0)
np.testing.assert_array_equal(text, np.array([1., 0.], np.float32))
np.testing.assert_array_equal(top_hcls[0], np.array([0.6, 0.8], np.float32))
np.testing.assert_allclose(scores, np.array([0.6, 0.0], np.float32), rtol=1e-6, atol=1e-6)
np.testing.assert_array_equal(motion, np.arange(8, dtype=np.float32).reshape(4, 2))
```

Also assert that packed arrays have numeric dtypes rather than `object`, and that requesting `topk=1` from a cache built at `topk=2` returns the first neighbor only.

- [ ] **Step 3: Run the cache test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_rag_cache -v
```

Expected: `ERROR` because `humanml3d_272.msa_rag_cache` does not exist.

- [ ] **Step 4: Implement the minimal packed cache**

Implement `humanml3d_272/msa_rag_cache.py` with schema version `1` and these files:

```text
manifest.json
sample_ids.txt
motion_values.npy
motion_offsets.npy
motion_lengths.npy
text_values.npy
text_offsets.npy
text_counts.npy
hcls_values.npy
retrieval_indices.npy
retrieval_scores.npy
```

Discover samples from `humanml3d_272/split/train.txt` and retain the same ordered intersection rule as the reference dataset. Convert all feature inputs to C-contiguous float32 arrays. Match the reference handling of `[CLS]` inputs by averaging axis 0 when an input is two-dimensional and flattening it otherwise. Flatten only the leading motion-sequence/caption axis; preserve the final feature dimension.

Compute retrieval in bounded caption chunks with the same float32 reference
operations:

```python
query = torch.from_numpy(caption_chunk).float()
query = query / (query.norm(dim=-1, keepdim=True) + 1e-6)
similarity = torch.matmul(query, library_hcls_norm.t())
similarity[torch.arange(chunk_size), source_sample_indices] = -1e6
scores, indices = torch.topk(similarity, k=topk, dim=1)
```

The default chunk contains at most 256 captions so the full similarity matrix
is never materialized. The test fixture additionally compares chunk sizes 1
and 3, requiring identical Top-K indices and float32-close scores. Store both
indices and scores. For a target such as
`humanml3d_272/msa_rag_cache/experiment-top5`, construct into a unique sibling
such as `humanml3d_272/msa_rag_cache/.experiment-top5.tmp-550e8400`, write the
manifest last, validate all shapes, then atomically publish it. With
`force=True`, rename an existing cache to a unique sibling such as
`.experiment-top5.stale-550e8400`,
publish the new cache, and remove only that exact stale generated directory
after successful validation.

`PackedMSARAGCache` loads numeric arrays with `np.load(..., mmap_mode="r")`; `get` slices motion/text by offsets, retrieves `[CLS]` vectors by stored indices, and returns float32 arrays.

- [ ] **Step 5: Run the cache-value tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_rag_cache -v
```

Expected: all cache-value and shape tests pass.

- [ ] **Step 6: Write failing cache validation tests**

Add tests asserting `CacheValidationError` for:

- requested Top-K greater than cached Top-K;
- changed `exclude_self`;
- changed text dimension;
- a missing packed array;
- modification of a source caption file after cache creation;
- a cache directory containing arrays but no completed manifest.

The source modification case must rewrite `000001.npy`, advance its mtime with
`os.utime`, and call `validate_cache` with the original source directories.

- [ ] **Step 7: Run validation tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_rag_cache -v
```

Expected: validation cases fail because manifest/source validation is not yet implemented.

- [ ] **Step 8: Implement manifest and source validation**

Record canonical paths and, for each accepted source file, relative name,
`st_size`, and `st_mtime_ns`. Validate schema, configuration, sample ordering,
source metadata, required array presence, dtypes, and shapes. Error messages
must name the mismatched field or file and recommend rebuilding with
`build_msa_rag_cache.py --force`.

Implement the CLI so `--validate-only` never writes and returns a non-zero exit
code on invalid data.

- [ ] **Step 9: Run Task 1 tests and static checks**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_rag_cache -v
conda run -n mgpt python -m py_compile humanml3d_272/msa_rag_cache.py build_msa_rag_cache.py tests/msa_rag_fixtures.py tests/test_msa_rag_cache.py
git diff --check
```

Expected: tests pass, compilation exits zero, and `git diff --check` is silent.

- [ ] **Step 10: Commit Task 1**

```bash
git add humanml3d_272/msa_rag_cache.py build_msa_rag_cache.py tests/__init__.py tests/msa_rag_fixtures.py tests/test_msa_rag_cache.py
git commit -m "feat: add validated global RAG training cache"
```

---

### Task 2: Add Reference/Packed Dataset Equivalence

**Files:**
- Modify: `humanml3d_272/dataset_msa_rag.py:23-231`
- Create: `tests/test_dataset_msa_rag.py`

**Interfaces:**
- Consumes: `validate_cache` and `PackedMSARAGCache` from Task 1.
- Produces: `Text2MotionMSARAGDataset(..., cache_mode: str = "reference", cache_dir: Optional[str] = None)`.
- Produces: `Text2MotionMSARAGDataset.get_item(idx: int, caption_idx: Optional[int] = None)`; `__getitem__` delegates with `caption_idx=None`, while tests and diagnostics can select a caption deterministically.
- Produces: `DATALoader(..., cache_mode: str = "reference", cache_dir: Optional[str] = None)` with the existing four-tensor batch contract.

- [ ] **Step 1: Write failing reference/packed equivalence tests**

Create `tests/test_dataset_msa_rag.py`. Build the Task 1 fixture/cache, change
the temporary working directory so the dataset sees its split, instantiate one
dataset in each mode, and compare every sample/caption:

```python
reference_item = reference.get_item(sample_idx, caption_idx)
packed_item = packed.get_item(sample_idx, caption_idx)
for actual, expected in zip(packed_item, reference_item):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
```

Also compare `collate_fn` results for motions of lengths 3 and 5, asserting
identical shapes, zero padding, float32 dtype, and values. Add errors for an
unknown mode, missing `cache_dir` in packed mode, and an invalid cache.

- [ ] **Step 2: Run dataset tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_dataset_msa_rag -v
```

Expected: failures because `cache_mode`, `cache_dir`, and `get_item` do not exist.

- [ ] **Step 3: Implement the two explicit dataset modes**

Preserve the existing constructor and reference arrays. Add explicit mode
validation:

```python
if cache_mode not in {"reference", "packed"}:
    raise ValueError(...)
if cache_mode == "packed" and not cache_dir:
    raise ValueError(...)
```

Reference `get_item` retains the current `np.load`, normalization, self
exclusion, and `torch.topk`. Packed `get_item` delegates to
`PackedMSARAGCache.get`. When `caption_idx is None`, choose it with the existing
uniform `random.randint` rule. Do not change `collate_fn`, padding, shuffle,
worker initialization, `drop_last`, pinned memory, or persistent workers.

- [ ] **Step 4: Run Task 2 tests and regression tests**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_rag_cache tests.test_dataset_msa_rag -v
conda run -n mgpt python -m py_compile humanml3d_272/dataset_msa_rag.py tests/test_dataset_msa_rag.py
git diff --check
```

Expected: all tests pass and static checks exit zero.

- [ ] **Step 5: Commit Task 2**

```bash
git add humanml3d_272/dataset_msa_rag.py tests/test_dataset_msa_rag.py
git commit -m "feat: load global RAG training data from packed cache"
```

---

### Task 3: Put the Complete Two-forward Loss Behind One Module Forward

**Files:**
- Create: `models/rag_training.py`
- Create: `tests/test_rag_training.py`

**Interfaces:**
- Produces: `lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor`.
- Produces: `estimate_lengths_from_padded_latents(m_tokens: torch.Tensor) -> torch.Tensor`.
- Produces: `replace_with_pred(latents: torch.Tensor, pred_xstart: torch.Tensor, step: int, total_steps: int) -> torch.Tensor`.
- Produces: `RAGTwoForwardLoss(rag_model: nn.Module, diffmlps_batch_mul: int = 4)`.
- Produces: `RAGTwoForwardLoss.forward(latents, m_lens, text_emb, top3_h_cls, top3_sim_scores, step, total_steps, cfg_drop_mask, empty_text_emb) -> torch.Tensor`.
- Produces: `get_rag_model(model: nn.Module) -> nn.Module`, recursively unwrapping `.module` and then returning `.rag_model`.

- [ ] **Step 1: Write a deterministic fake RAG model and failing loss-equivalence test**

In `tests/test_rag_training.py`, define a small real `nn.Module` whose
`forward` concatenates two condition positions with a learned projection of
motion latents, whose `motion_condition_slice` follows the production offset,
and whose `base_model.diff_loss` is a learned linear projection returning MSE
and `pred_xstart`.

Implement a test-only copy of the current reference
`forward_loss_withmask_2_forward`. Seed PyTorch, create two identical fake
models, restore the same RNG state before each call, and assert:

```python
torch.testing.assert_close(actual_loss, reference_loss)
torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)
```

For every trainable parameter, require matching gradients or matching `None`.
Also record the first-pass output in the fake model and assert it has
`requires_grad=False`.

- [ ] **Step 2: Run the loss test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training.RAGTrainingTest.test_loss_and_gradients_match_reference -v
```

Expected: `ERROR` because `models.rag_training` does not exist.

- [ ] **Step 3: Implement `RAGTwoForwardLoss` minimally**

Move the pure two-forward helpers from `train_t2m_rag.py` into
`models/rag_training.py`. Keep the operation and random-draw order unchanged:
mask creation, first pass under `torch.no_grad`, `pred_xstart.detach`, scheduled
replacement, second pass, repeated diffusion-head batch, mask indexing, then
the final diffusion loss.

The module owns the existing model only as:

```python
self.rag_model = rag_model
```

Do not add trainable parameters. Its `forward` returns only the scalar loss.

- [ ] **Step 4: Run the loss and gradient test and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training.RAGTrainingTest.test_loss_and_gradients_match_reference -v
```

Expected: pass.

- [ ] **Step 5: Write and run failing state-dict compatibility tests**

Add a test with the fake model asserting:

```python
self.assertEqual(
    list(get_rag_model(training_module).state_dict()),
    list(original_rag_model.state_dict()),
)
```

Wrap the training module in a test object exposing `.module`, repeat the
assertion, and verify there are no `rag_model.` prefixes in the returned
research-model state dict.

Run the test and expect failure until recursive unwrapping is implemented:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training -v
```

- [ ] **Step 6: Implement unwrapping and run Task 3 checks**

Implement `get_rag_model`. Leave `train_t2m_rag.py` unchanged in this task so
the existing launcher remains runnable until Task 4 switches the whole training
loop to the new module in one atomic change.

Run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training -v
conda run -n mgpt python -m py_compile models/rag_training.py tests/test_rag_training.py
git diff --check
```

Expected: tests pass and static checks exit zero.

- [ ] **Step 7: Commit Task 3**

```bash
git add models/rag_training.py tests/test_rag_training.py
git commit -m "fix: define a DDP-safe RAG two-forward loss boundary"
```

---

### Task 4: Integrate the DDP-safe Module and Preserve Checkpoints

**Files:**
- Modify: `train_t2m_rag.py:69-468`
- Modify: `tests/test_rag_training.py`
- Create: `tests/test_rag_training_ddp.py`

**Interfaces:**
- Consumes: `RAGTwoForwardLoss` and `get_rag_model` from Task 3.
- Produces: `build_checkpoint_payload(training_model, accelerator, optimizer, scheduler, iteration, generative_head_type, ema_enabled, ema_decay, ema_base_state=None, ema_rag_state=None) -> dict`.
- Preserves existing checkpoint top-level keys and unprefixed `trans`/`rag` state-dict keys.

- [ ] **Step 1: Write a failing checkpoint compatibility test**

Add a test that builds the fake RAG model and training wrapper, calls the
missing `build_checkpoint_payload` through a minimal accelerator double whose
`unwrap_model` returns its argument, and asserts these exact required keys:

```python
{
    "trans", "rag", "scheduler", "optimizer", "iter",
    "generative_head_type", "use_ema", "ema_decay",
}
```

With EMA enabled, additionally require `trans_ema` and `rag_ema`. Assert
`payload["rag"]` has exactly the same key order and tensor shapes as the
original research model state dict and no `rag_model.` prefix.

- [ ] **Step 2: Run the checkpoint test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training.RAGTrainingTest.test_checkpoint_payload_preserves_research_model_keys -v
```

Expected: failure because `build_checkpoint_payload` does not exist.

- [ ] **Step 3: Implement checkpoint serialization and training integration**

In `train_t2m_rag.py`:

1. build/load `rag_model` exactly as now;
2. wrap it with `RAGTwoForwardLoss`;
3. build the optimizer over the training wrapper;
4. pass the training wrapper, optimizer, and loader to `accelerator.prepare`;
5. call the prepared training wrapper once per step;
6. obtain the research model for EMA and saving through
   `get_rag_model(accelerator.unwrap_model(training_model))`;
7. delete the `train_forward_model = rag_model.module ...` path and its legacy
   log message;
8. remove the superseded two-forward helper implementations from
   `train_t2m_rag.py`;
9. build checkpoints through `build_checkpoint_payload`.

Keep checkpoint loading before the training wrapper is constructed, so old
`trans` and `rag` dictionaries load exactly as before.

- [ ] **Step 4: Implement nonblocking transfer and one loss synchronization**

Change only DataLoader-origin tensors:

```python
text_emb = text_emb.to(comp_device, non_blocking=True)
top3_h_cls = top3_h_cls.to(comp_device, non_blocking=True)
top3_sim_scores = top3_sim_scores.to(comp_device, non_blocking=True)
m_tokens = m_tokens.to(comp_device, non_blocking=True)
```

After the optimizer/EMA operations, call:

```python
loss_value = loss.item()
avg_loss += loss_value
```

Reuse `loss_value` for the active DDPM/RF accumulator. Do not change the
dropout-ratio logging cadence or optimizer/scheduler ordering.

- [ ] **Step 5: Run the checkpoint test and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training -v
```

Expected: all loss, gradient, and checkpoint tests pass.

- [ ] **Step 6: Write the failing two-process DDP regression test**

Create `tests/test_rag_training_ddp.py`. Use `torch.multiprocessing.spawn`,
`torch.distributed.init_process_group(backend="gloo", init_method="file://...")`,
and `DistributedDataParallel(RAGTwoForwardLoss(fake_model))`.

Each rank uses a different literal input batch, performs one wrapped forward,
backward, and SGD step, then writes its final flattened parameter vector to a
rank-specific file. The parent test loads both vectors and asserts exact
closeness. The test must call the DDP object itself; it must not access
`.module(...)` for the forward.

- [ ] **Step 7: Verify the DDP test catches the legacy bypass**

First temporarily call `ddp.module(...)` inside the test worker and run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training_ddp -v
```

Expected: fail because rank-local updates diverge. Restore the intended
`ddp(...)` call and rerun.

- [ ] **Step 8: Run the DDP test and full Task 4 checks**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_rag_training tests.test_rag_training_ddp -v
conda run -n mgpt python -m py_compile train_t2m_rag.py models/rag_training.py tests/test_rag_training.py tests/test_rag_training_ddp.py
git diff --check
```

Expected: both ranks finish, all tests pass, and static checks exit zero.

- [ ] **Step 9: Commit Task 4**

```bash
git add train_t2m_rag.py tests/test_rag_training.py tests/test_rag_training_ddp.py
git commit -m "fix: synchronize mainline RAG gradients through DDP"
```

---

### Task 5: Make Packed Mode the Safe Official Launcher Default

**Files:**
- Create: `tests/test_train_t2m_rag_launcher.py`
- Modify: `TRAIN_t2m_rag.sh:9-75`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: `build_msa_rag_cache.py` from Task 1.
- Produces launcher variables:
  - `RAG_CACHE_MODE=packed|reference` (default `packed`);
  - `RAG_CACHE_DIR` (default is
    `humanml3d_272/msa_rag_cache/` plus the basename of
    `MOTION_LATENT_DIR`, followed by `-top${RETRIEVAL_TOPK}`);
  - `REBUILD_RAG_CACHE=false|true`;
  - `NUM_WORKERS` (default `4`);
  - `RETRIEVAL_TOPK` (default `5`);
  - `PYTHON_BIN` (default `python`);
  - `ACCELERATE_BIN` (default `accelerate`).

- [ ] **Step 1: Write a failing launcher behavior test**

Create `tests/test_train_t2m_rag_launcher.py`. In a temporary directory create
the three required source directories and executable stub programs:

- the Python stub appends the literal prefix `cache:` followed by its received
  command-line arguments to a log and creates the requested cache directory;
- the Accelerate stub appends the literal prefix `accelerate:` followed by its
  received command-line arguments to the same log.

Run the real launcher with controlled environment variables and assert:

1. packed mode invokes cache building/validation before Accelerate;
2. Accelerate receives `--cache_mode packed`, `--cache_dir`, `--num_workers`,
   and the selected Top-K;
3. reference mode skips the builder and invokes Accelerate with
   `--cache_mode reference`;
4. a builder failure prevents Accelerate from running.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_train_t2m_rag_launcher -v
```

Expected: failures because the launcher does not expose or enforce the cache
workflow.

- [ ] **Step 3: Implement launcher cache preflight**

Move `#!/bin/bash` to the first line, add `set -euo pipefail` immediately
after it, and retain the existing commented preprocessing example below that
preamble. Derive the default cache name from
`basename "${MOTION_LATENT_DIR%/}"` and `RETRIEVAL_TOPK`.

For packed mode, invoke:

```bash
"$PYTHON_BIN" build_msa_rag_cache.py \
  --dataset-name "$DATASET" \
  --motion-latent-dir "$MOTION_LATENT_DIR" \
  --text-latent-dir "$TEXT_LATENT_DIR" \
  --hcls-dir "$HCLS_DIR" \
  --cache-dir "$RAG_CACHE_DIR" \
  --topk "$RETRIEVAL_TOPK" \
  --text-embed-dim "$TEXT_EMBED_DIM"
```

Append `--force` only when `REBUILD_RAG_CACHE=true`. For reference mode, skip
the builder. Reject other modes. Pass cache mode/directory, workers, and Top-K
to `train_t2m_rag.py`. Keep BF16 and the current total global batch size.

- [ ] **Step 4: Wire dataset arguments through the trainer**

Add `--cache_mode` and `--cache_dir` to `parse_args`, pass them to
`dataset_msa_rag.DATALoader`, and log both values. Packed mode requires a cache
directory; reference mode accepts none.

- [ ] **Step 5: Ignore generated caches and document both workflows**

Add:

```gitignore
humanml3d_272/msa_rag_cache/
```

Update the README official RAG training section to explain:

- the first packed launch builds and validates the cache before Accelerate;
- later launches reuse it;
- changed MSA-VAE/text sources require `REBUILD_RAG_CACHE=true`;
- `RAG_CACHE_MODE=reference` is the slow equivalence/rollback path;
- cache files are generated artifacts and must not be committed.

- [ ] **Step 6: Run launcher and complete regression checks**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_rag_cache \
  tests.test_dataset_msa_rag \
  tests.test_rag_training \
  tests.test_rag_training_ddp \
  tests.test_train_t2m_rag_launcher -v
conda run -n mgpt python -m py_compile \
  build_msa_rag_cache.py \
  humanml3d_272/msa_rag_cache.py \
  humanml3d_272/dataset_msa_rag.py \
  models/rag_training.py \
  train_t2m_rag.py
bash -n TRAIN_t2m_rag.sh
git diff --check
```

Expected: all tests pass, Python compilation and Bash parsing exit zero, and
`git diff --check` is silent.

- [ ] **Step 7: Commit Task 5**

```bash
git add .gitignore README.md TRAIN_t2m_rag.sh train_t2m_rag.py tests/test_train_t2m_rag_launcher.py
git commit -m "feat: use packed inputs for official RAG training"
```

---

### Task 6: Verify on Real MSA-T2M Features and Record Evidence

**Files:**
- Modify only if verification exposes a defect: files from Tasks 1-5.
- Do not commit cache artifacts or benchmark output.

**Interfaces:**
- Consumes all prior tasks.
- Produces no new production interface.

- [ ] **Step 1: Build the real Top-5 packed cache**

Run:

```bash
conda run -n mgpt python build_msa_rag_cache.py \
  --dataset-name t2m_272 \
  --motion-latent-dir humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right \
  --text-latent-dir humanml3d_272/text_latents_t5 \
  --hcls-dir humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right \
  --cache-dir humanml3d_272/msa_rag_cache/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right-top5 \
  --topk 5 \
  --text-embed-dim 768
```

Expected: cache validation succeeds and reports accepted sample/caption counts,
array shapes, and elapsed build time.

- [ ] **Step 2: Compare reference and packed retrieval on real samples**

Run a read-only Python probe that instantiates both dataset modes, checks the
first, middle, and last dataset items and every caption for those items, then
samples 100 additional `(sample_idx, caption_idx)` pairs from
`random.Random(1234)`. Require exact retrieved vectors and Top-K ordering, with
`np.testing.assert_allclose(..., rtol=1e-5, atol=1e-6)` for scores.

Expected: all comparisons pass.

- [ ] **Step 3: Measure reference versus packed item throughput**

With both datasets already initialized, time 2,000 deterministic `get_item`
calls in each mode using the same repeated index/caption list and
`time.perf_counter`. Report initialization time, total item time, and items/s.
Do not assert a hard speed ratio in tests because shared-filesystem load varies;
require only that the packed path performs no source `.npy` reads or online
full-library matmul according to its selected mode/log.

- [ ] **Step 4: Check an existing checkpoint contract**

Locate one existing official global-RAG checkpoint under `Experiments/` without
modifying it. Load it on CPU, instantiate the current research model, and
confirm the existing `trans` and `rag` keys load with the same missing/unexpected
key sets observed before this refactor. If no checkpoint is available, report
that the synthetic state-dict compatibility test is the available evidence and
do not claim a real-checkpoint validation.

- [ ] **Step 5: Run the full final verification suite**

Run fresh:

```bash
conda run -n mgpt python -m unittest discover -s tests -p 'test_*.py' -v
conda run -n mgpt python -m py_compile \
  build_msa_rag_cache.py \
  humanml3d_272/msa_rag_cache.py \
  humanml3d_272/dataset_msa_rag.py \
  models/rag_training.py \
  train_t2m_rag.py
bash -n TRAIN_t2m_rag.sh
git diff --check
git status --short --branch
```

Expected: all targeted tests pass, static checks exit zero, generated cache
artifacts remain ignored, and the only unrelated dirty path is the pre-existing
`paper writing/Research-Paper-Writing-Skills` submodule.
