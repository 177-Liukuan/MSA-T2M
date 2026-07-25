# MSA-VAE Full-Sequence Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the MSA-VAE Semantic Transformer on complete motions and fine-tune the full model with deterministic 64-frame replay while keeping model tensor names and legacy window-mode invocation compatible.

**Architecture:** Extend the existing HumanML MSA-VAE dataset with full/window views over shared records and a variable-length collator. Add reusable masked-loss and schedule helpers, add a semantic-only forward to the existing model, and route the two authoritative phases through explicit full/mixed modes. Keep evaluation architecture unchanged, add checkpoint metadata, and hand Phase 2 the final Phase-1 checkpoint.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1+cu118, Accelerate 1.0.1, NumPy, `unittest`, Bash.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-26-msa-vae-full-sequence-training-design.md`.
- Use the existing `mgpt` conda environment; do not install or upgrade core dependencies.
- Preserve the 272-D motion representation, causal latent convention, architecture dimensions, and all existing model state-dict keys.
- Keep direct `train_msa_vae.py` invocation backward compatible with fixed-window mode; authoritative phase launchers opt into the new modes explicitly.
- Do not modify `explorations/`, datasets, checkpoints, latent caches, logs, generated media, or third-party submodules.
- Do not run full training, TMR evaluation, SentenceT5 preprocessing, or GPU benchmarks.
- Use test-first development and observe every new behavior fail before implementation.
- Preserve unrelated changes in the primary checkout; all work occurs in `.worktrees/msa-vae-full-sequence`.

---

### Task 1: Add Full-Sequence Dataset Views and Variable-Length Collation

**Files:**

- Modify: `humanml3d_272/dataset_msa_vae.py`
- Create: `tests/test_dataset_msa_vae_full_sequence.py`

**Interfaces:**

- Produces: `MSAVAEDataset.get_item(item: int, sequence_mode: str) -> tuple`.
- Produces: `MSAVAESequenceView(dataset, sequence_mode)`.
- Produces: `collate_fn(batch) -> tuple` with valid motion lengths appended.
- Produces: `LengthBucketBatchSampler(lengths, batch_size, bucket_size, drop_last, seed)`.
- Produces: `make_loader(dataset, sequence_mode, batch_size, num_workers, bucket_size=0)`.
- Preserves: `DATALoader(...)` as the legacy constructor, extended with `sequence_mode` and `bucket_size`.

- [ ] **Step 1: Write failing full/window item tests**

Construct an `MSAVAEDataset` with `__new__`, inject one synthetic record with
70 motion frames, identity normalization, one global embedding, and a
frame-level local embedding file. Assert:

```python
full = dataset.get_item(0, "full")
window = dataset.get_item(0, "window")

self.assertEqual(full[0].shape, (68, 272))
self.assertEqual(full[4].shape[0], 17)
self.assertEqual(full[-1], 68)
self.assertEqual(window[0].shape, (64, 272))
self.assertEqual(window[4].shape[0], 16)
self.assertEqual(window[-1], 64)
```

Patch `random.randint` to make the window crop deterministic and assert its
motion and local targets use the same offset. Assert invalid sequence modes
raise `ValueError`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence.FullSequenceDatasetTest.test_full_and_window_views_share_aligned_record -v
```

Expected: failure because `get_item` does not exist.

- [ ] **Step 3: Implement mode-aware item construction**

Refactor the existing `__getitem__` body into:

```python
def __getitem__(self, item):
    return self.get_item(item, self.sequence_mode)

def get_item(self, item, sequence_mode):
    if sequence_mode not in ("window", "full"):
        raise ValueError(...)
```

For `full`, compute:

```python
valid_length = (len(motion) // self.unit_length) * self.unit_length
start = 0
motion_view = motion[:valid_length]
latent_len = valid_length // self.unit_length
```

For `window`, preserve the existing random 64-frame crop. Upsample local text
once to the source motion length, then slice it with the same `start` and
`valid_length`. Append `valid_length` to the existing return tuple. Reject a
record whose valid length is below one temporal unit.

- [ ] **Step 4: Run item tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence.FullSequenceDatasetTest -v
```

Expected: all item tests pass.

- [ ] **Step 5: Write failing collation and shared-view tests**

Add two items of lengths 64 and 68. Assert:

```python
batch = collate_fn([item64, item68])
self.assertEqual(batch[0].shape, (2, 68, 272))
self.assertEqual(batch[4].shape[:2], (2, 17))
torch.testing.assert_close(batch[-1], torch.tensor([64, 68]))
self.assertTrue(torch.all(batch[0][0, 64:] == 0))
self.assertTrue(torch.all(batch[4][0, 16:] == 0))
```

Construct `full_view = MSAVAESequenceView(dataset, "full")` and
`window_view = MSAVAESequenceView(dataset, "window")`; assert both expose the
same `dataset` object and report the underlying source lengths without copying
records.

- [ ] **Step 6: Run collation tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence.FullSequenceCollateTest -v
```

Expected: failure because variable arrays cannot be stacked and the view class
does not exist.

- [ ] **Step 7: Implement padded collation and dataset views**

Allocate zero-filled tensors using the maximum motion and local lengths, copy
valid prefixes, and return the existing fields plus `motion_lengths`.
`MSAVAESequenceView` delegates `__len__`, calls
`dataset.get_item(index, sequence_mode)`, and exposes:

```python
@property
def source_lengths(self):
    if self.sequence_mode == "window":
        return [self.dataset.window_size] * len(self.dataset)
    return [
        (len(entry["motion"]) // self.dataset.unit_length)
        * self.dataset.unit_length
        for entry in self.dataset.data
    ]
```

- [ ] **Step 8: Write and verify RED for length-bucketed batches**

For lengths `[64, 68, 192, 196, 72, 200]`, batch size 2, and bucket size 4,
assert every index occurs once when `drop_last=False`, every batch has at most
two indices, `set_epoch(1)` is deterministic across two sampler instances,
and batches do not mix the shortest and longest bucket.

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence.LengthBucketBatchSamplerTest -v
```

Expected: import failure for `LengthBucketBatchSampler`.

- [ ] **Step 9: Implement bucket sampler and loader factory**

Sort indices by length, partition them into buckets of
`max(bucket_size, batch_size)`, shuffle within each bucket and shuffle the
resulting batches using `random.Random(seed + epoch)`. Implement `__len__` and
`set_epoch`. `make_loader` uses the sampler only for full mode with a positive
bucket size; otherwise it uses normal shuffled batches.

- [ ] **Step 10: Run dataset tests and commit**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_dataset_msa_vae_full_sequence -v
conda run -n mgpt python -m py_compile \
  humanml3d_272/dataset_msa_vae.py \
  tests/test_dataset_msa_vae_full_sequence.py
git diff --check
```

Commit:

```bash
git add humanml3d_272/dataset_msa_vae.py \
  tests/test_dataset_msa_vae_full_sequence.py
git commit -m "feat: load full MSA-VAE motion sequences"
```

---

### Task 2: Add Length-Invariant Masked MSA-VAE Objectives

**Files:**

- Create: `utils/msa_vae_training.py`
- Create: `tests/test_msa_vae_training.py`

**Interfaces:**

- Produces: `valid_mask_from_lengths(lengths, max_len) -> BoolTensor`.
- Produces: `latent_lengths_from_frames(lengths, stride_t, down_t) -> LongTensor`.
- Produces: `masked_mse(pred, target, valid_mask) -> Tensor`.
- Produces: `masked_kl(mu, logvar, valid_mask) -> Tensor`.
- Produces: `masked_optimal_sigma_nll(pred, target, valid_mask, feature_slice) -> Tensor`.
- Produces: `masked_cosine_alignment(pred, target, valid_mask) -> Tensor`.
- Produces: `is_window_replay_step(step, interval) -> bool`.
- Produces: `validate_sequence_training_config(phase, mode, full_batch_size, replay_interval)`.
- Produces: `MSAVAELossWeights`.
- Produces: `compute_msa_vae_objective(outputs, targets, phase, batch_kind, weights)`.

- [ ] **Step 1: Write failing mask and floor-length tests**

Assert:

```python
lengths = torch.tensor([64, 68, 70])
latent = latent_lengths_from_frames(lengths, stride_t=2, down_t=2)
torch.testing.assert_close(latent, torch.tensor([16, 17, 17]))

mask = valid_mask_from_lengths(torch.tensor([2, 3]), 4)
torch.testing.assert_close(
    mask,
    torch.tensor([[True, True, False, False],
                  [True, True, True, False]]),
)
```

Run and verify import failure:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training.MaskHelperTest -v
```

- [ ] **Step 2: Implement length and mask helpers**

Use repeated integer floor division:

```python
result = lengths.long()
for _ in range(down_t):
    result = torch.div(result, stride_t, rounding_mode="floor")
```

Validate positive stride/downsampling inputs and nonnegative lengths.

- [ ] **Step 3: Write failing padding-invariance tests for dense losses**

Create two samples with valid lengths 2 and 3, then change only padded values
to large constants. Assert `masked_mse`, `masked_kl`, reconstruction NLL, and
root NLL remain unchanged. Duplicate valid frames within a sample and assert
the per-sample normalized loss does not double.

For `masked_optimal_sigma_nll`, use `slice(None)` for reconstruction and
`slice(0, 8)` for root features.

- [ ] **Step 4: Implement masked dense losses and verify GREEN**

Implement a private per-sample masked mean that expands `(B, T)` masks across
feature dimensions, divides by each sample's valid element count, and averages
samples with a nonzero count.

For optimal-sigma NLL:

```python
squared_error = (target_selected - pred_selected).pow(2)
sigma = masked_global_mean(squared_error).sqrt().clamp_min(math.exp(-6))
log_sigma = sigma.log()
nll = 0.5 * ((target_selected - pred_selected) / sigma).pow(2)
nll = nll + log_sigma + 0.5 * math.log(2 * math.pi)
return masked_per_sample_mean(nll, valid_mask)
```

Return `pred.sum() * 0.0` for an empty valid mask.

- [ ] **Step 5: Write failing token/sample alignment tests**

Assert `masked_cosine_alignment` accepts both `(B, D)` with `(B,)` mask and
`(B, T, D)` with `(B, T)` mask, ignores padded/adversarial target values, and
returns differentiable zero when the mask is empty.

- [ ] **Step 6: Implement cosine alignment and verify GREEN**

Normalize both features, compute `1 - cosine_similarity`, select valid rows or
tokens, and return their mean.

- [ ] **Step 7: Write failing objective and replay schedule tests**

Build synthetic output/target dictionaries. Assert:

- Phase 1 contains only latent/global/local terms;
- Phase-2 full contains all six terms;
- Phase-2 replay excludes latent/global numerically;
- calling `backward()` on replay produces zero (not absent) gradients for
  `mu_recon` and `clip_global_feat`;
- steps 4 and 8 replay for interval 4, while steps 1-3 and 5-7 do not;
- invalid phase/mode combinations raise descriptive `ValueError`.

- [ ] **Step 8: Implement objective composition and config validation**

Define:

```python
class MSAVAELossWeights(NamedTuple):
    root: float
    latent: float
    global_align: float
    local_align: float
```

`targets` contains `motion`, `motion_lengths`, `global_text`, `has_global`,
`local_text`, and `has_local`. Combine `has_local[:, None]` with the latent
valid mask. On replay, add:

```python
zero_semantic = (
    outputs["mu_recon"].sum() + outputs["clip_global_feat"].sum()
) * 0.0
```

to the total objective.

- [ ] **Step 9: Run masked-loss tests and commit**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_training -v
conda run -n mgpt python -m py_compile \
  utils/msa_vae_training.py tests/test_msa_vae_training.py
git diff --check
```

Commit:

```bash
git add utils/msa_vae_training.py tests/test_msa_vae_training.py
git commit -m "feat: mask variable-length MSA-VAE losses"
```

---

### Task 3: Add the Semantic-Only Model Forward

**Files:**

- Modify: `models/causal_cnn.py`
- Modify: `models/msa_vae.py`
- Create: `tests/test_msa_vae_full_sequence_model.py`

**Interfaces:**

- Produces: `CausalEncoder.encode_stats(x) -> (mu, logvar)`.
- Produces: `MSA_VAE.forward_semantic(x, lengths=None) -> dict`.
- Extends: `MSA_HumanVAE.forward(x, lengths=None, semantic_only=False)`.
- Preserves: existing `forward`, `encode`, `forward_decoder`, and state-dict
  key names.

- [ ] **Step 1: Write failing posterior-statistics equivalence test**

Instantiate a tiny `CausalEncoder`, fix its RNG seed, and assert the `mu` and
`logvar` returned by `forward` equal those from `encode_stats` for the same
input in evaluation mode.

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_full_sequence_model.CausalEncoderStatsTest -v
```

Expected: `AttributeError` for `encode_stats`.

- [ ] **Step 2: Implement `encode_stats` and verify GREEN**

Move the deterministic model/projection/chunk/clamp operations into
`encode_stats`; keep `forward` as:

```python
mu, logvar = self.encode_stats(x)
z = self.reparameterize(mu, logvar)
return z, mu, logvar
```

- [ ] **Step 3: Write failing semantic-only forward tests**

Instantiate a tiny `MSA_HumanVAE` with downsampling 2, latent dimension 4,
Transformer dimension 16, one encoder/decoder layer, and four attention heads.
Replace `cnn_decoder` with a module whose `forward` raises. Call:

```python
out = model(x, lengths=torch.tensor([64, 68]), semantic_only=True)
```

Assert it succeeds, returns `mu`, `mu_recon`, `h_cls`,
`clip_global_feat`, and `clip_local_feat`, omits `x_recon` and `z_local`, and
masks the seventeenth latent token of the 64-frame sample.

- [ ] **Step 4: Implement semantic-only forward and floor masks**

Extract one private helper:

```python
def _latent_padding_mask(self, lengths, max_len):
    latent_lengths = lengths
    for _ in range(self.down_t):
        latent_lengths = torch.div(
            latent_lengths, self.stride_t, rounding_mode="floor"
        )
    return self.lengths_to_mask(latent_lengths, max_len)
```

Use it from both standard and semantic-only forwards. Semantic-only computes
CNN statistics, Transformer encode/decode, and both projections without
sampling or CNN decoding.

- [ ] **Step 5: Write and pass state-dict compatibility test**

In the test, capture the exact state-dict key set from an unmodified
construction contract:

```python
keys = set(model.state_dict())
self.assertFalse(any("encode_stats" in key for key in keys))
self.assertFalse(any("semantic" in key for key in keys))
```

Save and strict-load the state dict into a second identical model. Also assert
the legacy `model(x)` output retains `x_recon`.

- [ ] **Step 6: Run model tests and commit**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_full_sequence_model -v
conda run -n mgpt python -m py_compile \
  models/causal_cnn.py models/msa_vae.py \
  tests/test_msa_vae_full_sequence_model.py
git diff --check
```

Commit:

```bash
git add models/causal_cnn.py models/msa_vae.py \
  tests/test_msa_vae_full_sequence_model.py
git commit -m "feat: add semantic-only MSA-VAE forward"
```

---

### Task 4: Route Phase Training Through Full and Replay Batches

**Files:**

- Modify: `train_msa_vae.py`
- Modify: `utils/eval_trans.py`
- Create: `tests/test_msa_vae_training_entrypoint.py`

**Interfaces:**

- Consumes: dataset views/loaders from Task 1.
- Consumes: objectives and schedule helpers from Task 2.
- Consumes: semantic-only model forward from Task 3.
- Produces: explicit full/mixed training routing.
- Produces: `select_training_batch(step, mode, full_iter, window_iter, replay_interval)`.
- Produces: `build_global_alignment_target(...)`.
- Produces: `save_msa_checkpoint(path, model, metadata)`.
- Produces: `build_msa_checkpoint_metadata(args) -> dict`.
- Produces: `validate_msa_checkpoint_metadata(metadata, args)`.
- Extends: `evaluation_msa_vae_multi(..., checkpoint_metadata=None)`.

- [ ] **Step 1: Write failing routing and metadata behavior tests**

Add importable helpers to `utils/msa_vae_training.py`. Using real finite
iterators containing literal sentinel batches, assert:

- full mode always consumes the full iterator;
- mixed mode consumes the window iterator only at replay steps;
- window mode always consumes the window iterator;
- an invalid mode raises;
- full/mixed global targets equal normalized global text even when pooled local
  text is adversarial;
- legacy window targets retain Spotlight interpolation.

Using a tiny real `nn.Linear`, call `save_msa_checkpoint`, load the file, and
assert it contains the exact model state and metadata. Assert required
metadata fields and incompatible `down_t`, `stride_t`, or `latent_dim` values
raise.

- [ ] **Step 2: Run entrypoint tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint -v
```

Expected: missing routing strings/helpers and metadata assertions fail.

- [ ] **Step 3: Implement checkpoint metadata helpers**

Return:

```python
{
    "format_version": 1,
    "phase": args.phase,
    "sequence_mode": args.sequence_mode,
    "window_size": args.window_size,
    "full_seq_batch_size": args.full_seq_batch_size,
    "window_replay_interval": args.window_replay_interval,
    "down_t": args.down_t,
    "stride_t": args.stride_t,
    "latent_dim": args.latent_dim,
    "normalized_loss_version": 1,
}
```

Legacy checkpoints without metadata remain accepted. New metadata rejects
different `down_t`, `stride_t`, or `latent_dim`; phase and sequence mode are
logged but do not block Phase-1 to Phase-2 handoff.

- [ ] **Step 4: Build phase-aware loaders**

Construct one `MSAVAEDataset`, then:

- window mode: one window loader using `args.batch_size`;
- full mode: one full loader using `args.full_seq_batch_size`;
- mixed mode: one full loader plus one window loader sharing that dataset.

Prepare every used loader with Accelerate. Maintain independent infinite
iterators. Select `batch_kind="window"` only on deterministic replay steps.

- [ ] **Step 5: Replace inline unmasked losses**

Unpack the appended `motion_lengths`, pass them to the model, construct the
target dictionary, and call `compute_msa_vae_objective`.

For full/mixed global batches:

```python
global_target = F.normalize(global_text_gt, dim=-1)
```

For legacy window mode, preserve Spotlight mixing. Phase 1 passes
`semantic_only=True`; all training calls use the Accelerate-wrapped `net`
instead of bypassing the wrapper.

- [ ] **Step 6: Add metadata to evaluation saves and final save**

Extend `evaluation_msa_vae_multi` payload creation to:

```python
payload = {"net": net.state_dict()}
if checkpoint_metadata is not None:
    payload["metadata"] = dict(checkpoint_metadata)
```

Use `save_msa_checkpoint` for best-FID, best-MPJPE, and last checkpoints.
After the final optimizer step, synchronize ranks and have the main process
save an unwrapped `net_last.pth` payload so Phase 1 always hands off its final
state. The helper behavior test proves the payload contract; a focused CPU
smoke invocation of the extracted batch router and objective proves the
training route without importing the top-level executable.

- [ ] **Step 7: Run entrypoint and regression tests**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_training_entrypoint \
  tests.test_msa_vae_training \
  tests.test_dataset_msa_vae_full_sequence \
  tests.test_msa_vae_full_sequence_model -v
conda run -n mgpt python -m py_compile \
  train_msa_vae.py utils/eval_trans.py \
  utils/msa_vae_training.py \
  tests/test_msa_vae_training_entrypoint.py
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add train_msa_vae.py utils/eval_trans.py \
  utils/msa_vae_training.py \
  tests/test_msa_vae_training_entrypoint.py
git commit -m "feat: train MSA-VAE on full and replay batches"
```

---

### Task 5: Add Options and Authoritative Launcher Contracts

**Files:**

- Modify: `options/option_msa_vae.py`
- Modify: `TRAIN_msa_vae_phase1.sh`
- Modify: `TRAIN_msa_vae_phase2.sh`
- Create: `tests/test_msa_vae_full_sequence_launchers.py`

**Interfaces:**

- Produces CLI options `sequence_mode`, `full_seq_batch_size`,
  `window_replay_interval`, and `length_bucket_size`.
- Phase 1 explicitly selects full mode.
- Phase 2 explicitly selects mixed mode and loads Phase-1 `net_last.pth`.

- [ ] **Step 1: Write failing parser and launcher tests**

Load parser defaults by patching `sys.argv` and assert:

```python
self.assertEqual(args.sequence_mode, "window")
self.assertEqual(args.full_seq_batch_size, 32)
self.assertEqual(args.window_replay_interval, 4)
self.assertEqual(args.length_bucket_size, 256)
```

Create a temporary executable named `accelerate` that writes its received
arguments to a file. Run each real launcher with the temporary directory first
on `PATH` and controlled environment variables. Assert the captured arguments
show:

- Phase 1 passes `--sequence_mode full`;
- Phase 2 passes `--sequence_mode mixed`;
- both pass `--full-seq-batch-size`;
- Phase 2 passes `--window-replay-interval`;
- Phase 2 resolves `${PHASE1_DIR}/net_last.pth`;
- experiment names contain `fullseq` and `fullseq_replay`;
- environment overrides reach the command.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_full_sequence_launchers -v
```

- [ ] **Step 3: Add parser options and validation call**

Add the four arguments under a `sequence training` option section. After
parsing in `train_msa_vae.py`, call
`validate_sequence_training_config(...)` before constructing data or models.

- [ ] **Step 4: Update Phase-1 launcher**

Add:

```bash
FULL_SEQ_BATCH_SIZE=${FULL_SEQ_BATCH_SIZE:-32}
LENGTH_BUCKET_SIZE=${LENGTH_BUCKET_SIZE:-256}
```

Pass full mode, both values, and rename the experiment with `fullseq`.

- [ ] **Step 5: Update Phase-2 launcher**

Add:

```bash
FULL_SEQ_BATCH_SIZE=${FULL_SEQ_BATCH_SIZE:-16}
WINDOW_REPLAY_INTERVAL=${WINDOW_REPLAY_INTERVAL:-4}
LENGTH_BUCKET_SIZE=${LENGTH_BUCKET_SIZE:-256}
RESUME_PTH="${PHASE1_DIR}/net_last.pth"
```

Pass mixed mode, all values, and rename the experiment with
`fullseq_replay${WINDOW_REPLAY_INTERVAL}`.

- [ ] **Step 6: Run parser, launcher, and shell checks**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_full_sequence_launchers -v
bash -n TRAIN_msa_vae_phase1.sh
bash -n TRAIN_msa_vae_phase2.sh
conda run -n mgpt python -m py_compile options/option_msa_vae.py
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add options/option_msa_vae.py \
  TRAIN_msa_vae_phase1.sh TRAIN_msa_vae_phase2.sh \
  tests/test_msa_vae_full_sequence_launchers.py
git commit -m "feat: launch full-sequence MSA-VAE curriculum"
```

---

### Task 6: Document Artifact Regeneration and Run Final Verification

**Files:**

- Modify: `README.md`
- Modify: `get_msa_latent.py`
- Create: `tests/test_msa_vae_artifact_contract.py`

**Interfaces:**

- Documents the new phase order, batch controls, checkpoint handoff, and
  required latent regeneration.
- Logs checkpoint metadata and rejects output directories that already contain
  MSA artifacts from a different checkpoint unless explicitly empty.

- [ ] **Step 1: Write failing artifact-contract tests**

With temporary checkpoint and output files, behavior-test pure manifest
helpers extracted to `utils/msa_vae_training.py`. Assert the generated
`extraction_metadata.json` records checkpoint path, file size/mtime, sequence
mode, downsampling, and latent dimension. A matching rerun succeeds; a
different checkpoint signature raises before any manifest is replaced.

Extract `prepare_extraction_roots(roots, checkpoint_path, checkpoint_metadata,
args)` and call it from `get_msa_latent.py` before creating any `.npy` output.
Test the real helper against three temporary roots rather than checking source
text.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_artifact_contract -v
```

- [ ] **Step 3: Implement extraction manifest helpers**

Add:

```python
def checkpoint_signature(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def validate_or_write_extraction_manifest(root, payload):
    ...
```

Write JSON atomically through a sibling temporary file and `os.replace`.
Empty roots are accepted. A matching existing manifest is accepted. A
different signature raises before any latent array is written.

- [ ] **Step 4: Integrate manifests and update README**

Validate all three output roots before extraction begins. Document:

1. Phase-1 full-sequence training;
2. Phase-2 full-sequence/replay training;
3. extraction from the selected Phase-2 checkpoint into new empty roots;
4. rebuilding retrieval caches;
5. retraining/evaluating RAG.

State that legacy and new MSA artifact roots must not be mixed.

- [ ] **Step 5: Run complete verification**

Run:

```bash
conda run -n mgpt python -m unittest discover -s tests -v
conda run -n mgpt python -m py_compile \
  humanml3d_272/dataset_msa_vae.py \
  models/causal_cnn.py models/msa_vae.py \
  options/option_msa_vae.py \
  utils/msa_vae_training.py utils/eval_trans.py \
  train_msa_vae.py get_msa_latent.py
bash -n TRAIN_msa_vae_phase1.sh
bash -n TRAIN_msa_vae_phase2.sh
git diff --check
git status --short
```

- [ ] **Step 6: Run synthetic CPU end-to-end smoke check**

Using only random tensors:

1. collate 64- and 68-frame samples;
2. run a tiny semantic-only forward and Phase-1 objective;
3. run a tiny standard forward and Phase-2 full objective;
4. run a 64-frame replay objective;
5. call backward for every objective;
6. assert finite losses and gradients.

Do not load real data, checkpoints, SentenceT5, TMR, or CUDA.

- [ ] **Step 7: Commit documentation and artifact safeguards**

```bash
git add README.md get_msa_latent.py \
  utils/msa_vae_training.py \
  tests/test_msa_vae_artifact_contract.py
git commit -m "docs: record full-sequence MSA artifact workflow"
```

- [ ] **Step 8: Review implementation range**

Run:

```bash
git log --oneline 93f45db..HEAD
git diff --stat 93f45db..HEAD
git diff --check 93f45db..HEAD
```

Confirm every design requirement is represented by code or tests and no data,
checkpoint, cache, log, rendered output, exploration file, or submodule
change is included.
