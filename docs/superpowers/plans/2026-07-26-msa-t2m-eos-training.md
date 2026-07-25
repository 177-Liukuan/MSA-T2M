# MSA-T2M Consistent EOS Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Stage-2 sample supervise its EOS token independently of batch composition while excluding padding from loss and pseudo-target replacement.

**Architecture:** The RAG collate boundary returns pre-padding lengths alongside right-padded latents. Training passes the complete latent tensor and exact lengths into the existing target-aligned RAG loss, which validates lengths, masks padded targets, and restricts two-forward replacement to valid positions. The RAG wrapper enforces the combined condition-plus-motion context bound.

**Tech Stack:** Python 3.8, PyTorch 2.4, NumPy, `unittest`, existing `mgpt` conda environment.

## Global Constraints

- Do not alter latent files, model parameters, state-dict names, checkpoint formats, inference interfaces, or retrieval features.
- Lengths include EOS and come from sequence shapes before padding.
- Do not infer valid lengths from latent values, globally remove a batch column, or silently clamp invalid lengths.
- Keep validation CPU-only; do not launch full training or TMR evaluation.

---

### Task 1: Return exact motion lengths and preserve all valid tokens

**Files:**
- Modify: `tests/test_dataset_msa_rag.py`
- Modify: `humanml3d_272/dataset_msa_rag.py:226-239`
- Modify: `train_t2m_rag.py:319-348`

**Interfaces:**
- Consumes: dataset items ending in an unpadded `np.ndarray[T, D]`.
- Produces: `collate_fn(batch) -> (text, top_hcls, top_scores, motions, motion_lengths)`, where `motions` is `[B, max(T), D]` and `motion_lengths` is `torch.long[B]`.

- [ ] **Step 1: Write failing collate regression tests**

Update the reference/packed parity test to treat `batch[-2]` as motions and
`batch[-1]` as lengths. Assert literal lengths `[3, 5]`, motion shape
`(2, 5, 2)`, motion dtype `float32`, and length dtype `long`.

Add a direct collate test whose first sequence ends in a literal all-zero
valid token. Assert the returned lengths are `[2, 1]`, proving length does not
depend on latent values.

- [ ] **Step 2: Run dataset tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v tests.test_dataset_msa_rag
```

Expected: the tests fail because `collate_fn` returns four values and exposes
no explicit length tensor.

- [ ] **Step 3: Implement the collate contract**

Before allocating `motion_batch`, construct:

```python
motion_lengths = torch.tensor(
    [seq.shape[0] for seq in motion_list],
    dtype=torch.long,
)
max_len = int(motion_lengths.max().item())
```

Return `motion_lengths` after `motion_batch`.

- [ ] **Step 4: Consume full motion tensors in training**

Change the loader unpacking to receive `m_tokens_len`, transfer it to the
training device, set `input_latent = m_tokens`, and remove
`estimate_lengths_from_padded_latents`, `[:, :-1]`, and the length clamp.

- [ ] **Step 5: Run dataset tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest -v tests.test_dataset_msa_rag
```

Expected: all dataset tests pass in reference and packed modes.

---

### Task 2: Validate lengths and mask two-forward replacement per sample

**Files:**
- Modify: `tests/test_rag_training.py`
- Modify: `models/rag_training.py:7-125`

**Interfaces:**
- Consumes: `latents[B, T, D]`, authoritative `m_lens[B]`, and
  `valid_mask = lengths_to_mask(m_lens, T)`.
- Produces: a loss over valid targets only and
  `replace_with_pred(latents, pred_xstart, step, total_steps, valid_mask)`
  that never changes padding.

- [ ] **Step 1: Write failing length and replacement tests**

Add tests that:

- pass `m_lens=[4]` with a width-three tensor and expect `ValueError`;
- pass `m_lens=[0]` and expect `ValueError`;
- call `replace_with_pred` at `step=total_steps` with lengths `[3, 1]` and assert all
  four valid positions use `pred_xstart` while both padding positions remain
  unchanged;
- call it at a half-decay step with lengths `[4, 2]` and assert exactly two
  and one positions, respectively, are replaced.

Set the existing reference loss fixture to `step=total_steps` so every valid
position has a deterministic replacement and the reference continues to test
real loss and gradient equivalence.

- [ ] **Step 2: Run RAG training tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v tests.test_rag_training
```

Expected: replacement tests fail because the function has no valid-mask
argument, and invalid lengths are not rejected.

- [ ] **Step 3: Implement authoritative length validation**

In `RAGTwoForwardLoss.forward`, reject:

```python
if m_lens.ndim != 1 or m_lens.shape[0] != batch_size:
    raise ValueError(...)
if torch.any(m_lens <= 0):
    raise ValueError(...)
if torch.any(m_lens > sequence_length):
    raise ValueError(...)
```

Then build one `valid_mask = lengths_to_mask(m_lens, sequence_length)` and
derive the flattened repeated loss mask from it.

- [ ] **Step 4: Implement per-sample valid replacement**

Extend `replace_with_pred` with a required `valid_mask`. Draw independent
random scores for `[B, T]`, rank only valid positions, and replace
`floor(valid_length * decay_factor)` positions per sample:

```python
scores = torch.rand(batch_size, sequence_length, device=latents.device)
scores = scores.masked_fill(~valid_mask, float("inf"))
ranks = scores.argsort(dim=1).argsort(dim=1)
counts = torch.floor(
    valid_mask.sum(dim=1).to(torch.float32) * decay_factor
).to(torch.long)
replace_mask = valid_mask & (ranks < counts.unsqueeze(1))
```

Pass `valid_mask` from `RAGTwoForwardLoss.forward`.

- [ ] **Step 5: Run RAG training tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest -v tests.test_rag_training
```

Expected: all loss, gradient, checkpoint, validation, and replacement tests
pass.

---

### Task 3: Enforce the RAG transformer context boundary

**Files:**
- Modify: `tests/test_rag_training.py`
- Modify: `models/llama_rag_model.py:245-283`

**Interfaces:**
- Consumes: `num_condition_tokens`, `motion_latents.shape[1]`, and
  `base_model.config.block_size`.
- Produces: normal RAG forward output when their sum is within the context,
  otherwise a `ValueError` naming required and configured lengths.

- [ ] **Step 1: Write a failing real-wrapper boundary test**

Construct a tiny base model with a four-token block size, one text token, one
RAG token, and identity/no-op transformer blocks. Assert two motion tokens
produce a four-position output, while three motion tokens raise `ValueError`.

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_rag_training.RAGTrainingTest.test_rag_wrapper_enforces_total_context_length
```

Expected: the overlong call reaches RoPE/model execution instead of raising
the required explicit error.

- [ ] **Step 3: Implement the context guard**

After concatenating condition and motion embeddings, check:

```python
if x.shape[1] > self.base_model.config.block_size:
    raise ValueError(
        "RAG sequence length {} exceeds block_size {} "
        "({} condition + {} motion tokens)".format(...)
    )
```

- [ ] **Step 4: Run the focused and full test suites**

Run:

```bash
conda run -n mgpt python -m unittest -v \
  tests.test_dataset_msa_rag tests.test_rag_training
conda run -n mgpt python -m unittest discover -s tests -v
conda run -n mgpt python -m py_compile \
  humanml3d_272/dataset_msa_rag.py \
  train_t2m_rag.py \
  models/rag_training.py \
  models/llama_rag_model.py \
  tests/test_dataset_msa_rag.py \
  tests/test_rag_training.py
git diff --check
```

Expected: all repository tests pass, compilation succeeds, and no whitespace
errors are reported.

- [ ] **Step 5: Commit and publish**

```bash
git add \
  humanml3d_272/dataset_msa_rag.py \
  train_t2m_rag.py \
  models/rag_training.py \
  models/llama_rag_model.py \
  tests/test_dataset_msa_rag.py \
  tests/test_rag_training.py
git commit -m "fix: make EOS supervision batch independent"
git push origin fix/msa-t2m-eos-decode-main
```

Update draft PR #4 to describe both inference-side EOS exclusion and
training-side consistent EOS supervision.
