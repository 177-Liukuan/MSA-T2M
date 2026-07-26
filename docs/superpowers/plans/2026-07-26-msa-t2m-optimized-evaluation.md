# MSA-T2M Protocol-Equivalent Optimized Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate FP32 single-GPU MSA-T2M evaluation pipeline that preserves the existing metric protocol while batching text encoding, retrieval, autoregressive DDPM sampling, and equal-length MSA-VAE decoding.

**Architecture:** A new utility module owns the active-set generation state machine, equal-length decoder grouping, and protocol-equivalent metric loop. A new entrypoint imports the existing official loader/model/checkpoint helpers, constructs the same DDPM/RAG/MSA-VAE/evaluator stack, and calls the optimized utility without changing any existing evaluation file.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1+cu118, NumPy, SentenceTransformers, existing MSA-T2M models and TEMOS evaluator, Bash, unittest.

## Global Constraints

- Create new evaluation files only; do not modify any existing evaluation source file, launcher, model, dataset, or metric helper.
- Preserve HumanML3D-272 batch size 32, shuffle, length sorting, stochastic caption/crop behavior, and `drop_last=True`.
- Preserve FP32, CFG scale, DDPM 50-step sampling, retrieval Top-K, EOS distance rule, MSA-VAE decoding, evaluator checkpoint, and metric formulas.
- The EOS-triggering candidate is never appended or decoded.
- Preserve the existing one-zero-latent fallback when EOS occurs before any accepted motion token.
- Do not run a full HumanML3D evaluation during code validation.

---

### Task 1: Active-Set Batched Latent Generation

**Files:**
- Create: `utils/eval_msa_t2m_optimized.py`
- Create: `tests/test_eval_msa_t2m_optimized.py`

**Interfaces:**
- Produces: `BatchedLatentResult(latents, stop_steps, empty_fallback_count)`.
- Produces: `generate_latents_active_set(rag_model, text_emb, empty_text_emb, top_hcls, top_scores, max_token_lengths, latent_dim, reference_end_latent, stop_threshold, enable_stopping, cfg_scale, temperature=1.0) -> BatchedLatentResult`.
- `latents` is a list of `[1, T_i, latent_dim]` FP32 tensors in original sample order.
- `stop_steps` is a CPU `LongTensor[B]`; `-1` means length exhaustion and nonnegative values identify the excluded EOS candidate step.

- [ ] **Step 1: Write the failing no-EOS equivalence test**

Add a deterministic fake RAG model whose next candidate encodes the original
sample id and current prefix length:

```python
class DeterministicRAG(torch.nn.Module):
    def sample_next_with_cfg(
        self, motion_prefix, text_emb, empty_text_emb, top3_h_cls=None,
        top3_sim_scores=None, cfg_scale=4.0, temperature=1.0,
    ):
        step = motion_prefix.shape[1]
        return torch.stack((text_emb[:, 0], torch.full_like(text_emb[:, 0], step)), dim=-1)


def test_active_set_matches_serial_without_eos():
    model = DeterministicRAG()
    text = torch.tensor([[10.0], [20.0], [30.0]])
    result = generate_latents_active_set(
        model, text, torch.zeros(1), None, None,
        torch.tensor([1, 3, 2]), latent_dim=2,
        reference_end_latent=None, stop_threshold=0.1,
        enable_stopping=False, cfg_scale=4.0,
    )
    assert [x.squeeze(0).tolist() for x in result.latents] == [
        [[10.0, 0.0]],
        [[20.0, 0.0], [20.0, 1.0], [20.0, 2.0]],
        [[30.0, 0.0], [30.0, 1.0]],
    ]
    assert result.stop_steps.tolist() == [-1, -1, -1]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized.OptimizedGenerationTests.test_active_set_matches_serial_without_eos -v
```

Expected: import failure because `utils.eval_msa_t2m_optimized` does not exist.

- [ ] **Step 3: Implement the minimal active-set state machine**

Create a frozen dataclass and generation function. At each step, select samples
whose ceilings exceed the step, stack their accepted prefixes, index all
condition tensors with the same active indices, sample once, test EOS, and
append only non-EOS candidates. Convert an empty accepted list to
`zeros(1, 1, latent_dim)` after generation.

```python
@dataclass(frozen=True)
class BatchedLatentResult:
    latents: List[torch.Tensor]
    stop_steps: torch.Tensor
    empty_fallback_count: int
```

Reject non-1D ceilings, nonpositive ceilings, dimension mismatches, and a
sample-count mismatch with `ValueError`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Add failing EOS and length tests**

Add independent tests proving:

- a candidate equal to the reference end latent is excluded;
- one sample can stop while remaining samples continue;
- each sample obeys its own maximum token count;
- first-candidate EOS returns one zero latent and increments
  `empty_fallback_count`;
- `stop_steps` reports the excluded candidate index.

Use candidate vectors with a dedicated EOS dimension so the assertions inspect
real returned latent values rather than model call counts.

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_t2m_optimized -v
```

Expected: at least one EOS/active-set assertion fails until per-sample removal
and fallback handling are complete.

- [ ] **Step 7: Complete EOS and validation behavior**

Implement per-sample `finished`, preserve original indices during active-set
shrinkage, exclude EOS before append, and validate the reference latent has
shape `[latent_dim]`.

- [ ] **Step 8: Run the task tests**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_t2m_optimized -v
```

Expected: all Task 1 tests PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add utils/eval_msa_t2m_optimized.py tests/test_eval_msa_t2m_optimized.py
git commit -m "feat: batch MSA-T2M latent generation"
```

---

### Task 2: Batched Conditioning and Equal-Length Decoding

**Files:**
- Modify: `utils/eval_msa_t2m_optimized.py`
- Modify: `tests/test_eval_msa_t2m_optimized.py`

**Interfaces:**
- Produces: `OptimizedRAGEvalSampler.sample_batch_for_eval_CFG(text, lengths, unit_length=4, cfg=4.0) -> BatchedLatentResult`.
- Produces: `decode_equal_length_groups(decoder, latent_sequences, max_motion_length, motion_dim) -> Tuple[Tensor, LongTensor]`.
- The sampler accepts the same text source, RAG retriever, reference latent, and stopping configuration as the existing serial adapter.

- [ ] **Step 1: Write failing batched-conditioning tests**

Use recording text encoders and retrievers to assert one call for a
three-caption batch:

```python
def test_sampler_encodes_and_retrieves_once_per_batch():
    sampler = make_recording_sampler()
    result = sampler.sample_batch_for_eval_CFG(
        ["one", "two", "three"], torch.tensor([4, 8, 12]), unit_length=4,
    )
    self.assertEqual(sampler.text_encoder.calls, [["one", "two", "three"]])
    self.assertEqual(sampler.retriever.calls, 1)
    self.assertEqual(len(result.latents), 3)
```

Add separate coverage for offline `batch_lookup`, no-RAG mode, text embedding
dimension rejection, and `max(1, length // unit_length)` ceilings.

- [ ] **Step 2: Run focused sampler tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized.OptimizedSamplerTests -v
```

Expected: failure because `OptimizedRAGEvalSampler` is absent.

- [ ] **Step 3: Implement `OptimizedRAGEvalSampler`**

For online T5, call `text_encoder.encode(text_list)` once and preserve the
existing NumPy-FP32-to-CUDA conversion. For offline embeddings, call the
existing lookup once. Retrieve Top-K once unless RAG is disabled, then call
`generate_latents_active_set`.

- [ ] **Step 4: Run sampler tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing grouped-decoder tests**

Use a decoder that records group shapes and adds a sample-id-dependent value.
Pass latent lengths `[2, 1, 2, 3, 1]`. Assert:

- decoder calls contain batch/length shapes `(2, 2)`, `(2, 1)`, `(1, 3)`;
- outputs return in original order;
- predicted frame lengths equal decoder output lengths capped at 300;
- the output buffer has `[5, 300, motion_dim]`.

- [ ] **Step 6: Run grouped-decoder test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized.OptimizedDecodeTests -v
```

Expected: failure because `decode_equal_length_groups` is absent.

- [ ] **Step 7: Implement grouped decoding**

Build a stable mapping `latent_length -> original indices`, concatenate only
identical-length tensors, decode each group once, and copy each decoded motion
back to its original position. Keep predicted lengths on CPU to match the
existing evaluator interface.

- [ ] **Step 8: Run all focused tests**

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_t2m_optimized -v
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add utils/eval_msa_t2m_optimized.py tests/test_eval_msa_t2m_optimized.py
git commit -m "feat: batch MSA-T2M conditioning and decoding"
```

---

### Task 3: Protocol-Equivalent Metric Loop

**Files:**
- Modify: `utils/eval_msa_t2m_optimized.py`
- Modify: `tests/test_eval_msa_t2m_optimized.py`

**Interfaces:**
- Produces: `evaluation_transformer_272_optimized(val_loader, net, trans, logger, evaluator, cfg=4.0, device=torch.device("cuda"), unit_length=4)`.
- Returns the same tuple as `evaluation_transformer_272_single`: FID, generated diversity, R@1, R@2, R@3, generated matching score, logger.

- [ ] **Step 1: Write a failing batch-boundary test**

Construct two four-sample batches with deterministic text/motion embeddings.
Compare the optimized loop's accumulated R-Precision and matching score with
the sum of two independent calls to existing
`utils.eval_trans.calculate_R_precision`. Also calculate the intentionally
wrong eight-sample pooled result and assert the fixture distinguishes it.

- [ ] **Step 2: Run the metric-loop test and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized.OptimizedMetricTests -v
```

Expected: failure because `evaluation_transformer_272_optimized` is absent.

- [ ] **Step 3: Implement the optimized evaluation loop**

Mirror `evaluation_transformer_272_single` while replacing only its inner
per-sample generation loop:

```python
batch_result = trans.sample_batch_for_eval_CFG(
    text, m_length, unit_length=unit_length, cfg=cfg,
)
pred_pose_eval, pred_len = decode_equal_length_groups(
    net, batch_result.latents, max_motion_length=pose.shape[1],
    motion_dim=pose.shape[-1],
)
```

Continue using existing `calculate_R_precision`,
`calculate_activation_statistics`, `calculate_diversity`, and
`calculate_frechet_distance`. Calculate R-Precision independently for every
original loader batch before concatenating embeddings for FID.

- [ ] **Step 4: Run metric-loop test and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Add return-contract and ordering tests**

Assert the return tuple has seven elements, the evaluator sees generated
motions in input order, and sample counts used for normalization equal the
sum of original batch sizes.

- [ ] **Step 6: Run all optimized evaluation tests**

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_t2m_optimized -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add utils/eval_msa_t2m_optimized.py tests/test_eval_msa_t2m_optimized.py
git commit -m "feat: preserve metrics in optimized MSA-T2M evaluation"
```

---

### Task 4: Independent Optimized Entrypoint

**Files:**
- Create: `eval_msa_t2m_rag_t5_optimized.py`
- Create: `tests/test_eval_msa_t2m_optimized_entrypoint.py`

**Interfaces:**
- Imports path/checkpoint/retrieval helpers from
  `eval_msa_t2m_rag_t5.py`.
- Constructs `OptimizedRAGEvalSampler` and calls
  `evaluation_transformer_272_optimized`.
- Exposes `main()` and runs it only under `if __name__ == "__main__"`.

- [ ] **Step 1: Write the failing import and parser test**

```python
def test_optimized_entrypoint_import_has_no_side_effects(self):
    module = importlib.import_module("eval_msa_t2m_rag_t5_optimized")
    self.assertTrue(callable(module.main))
    self.assertTrue(callable(module.build_optimized_parser))
```

Test `build_optimized_parser` with representative official arguments and
assert `optimized_pipeline is True`, default `generative_head_type == "ddpm"`,
and default output experiment name ends in `_optimized`.

- [ ] **Step 2: Run entrypoint tests and verify RED**

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_t2m_optimized_entrypoint -v
```

Expected: import failure because the new entrypoint is absent.

- [ ] **Step 3: Implement parser and preflight validation**

Reuse the existing parser and path resolvers without modifying them. Add only
optimized-pipeline metadata. Reject non-DDPM evaluation explicitly because
the official mainline evaluation construction is DDPM and this pipeline must
not silently reinterpret RF checkpoints.

- [ ] **Step 4: Implement official-equivalent model loading**

Construct:

- the same `MSA_HumanVAE` arguments and strict VAE checkpoint load;
- `LLaMAHFConfig.from_name("Normal_size")` with block size 78;
- default DDPM `LLaMAHF` and `LLaMARAGWrapper`;
- existing EMA key selection;
- online/offline text source, RAG retriever, empty embedding, and reference EOS
  latent;
- the same DistilBERT text evaluator, 272-D motion evaluator, and
  `epoch=99.ckpt`.

Log `optimized_pipeline=true` and pass the constructed components into the
optimized utility loop.

- [ ] **Step 5: Run entrypoint tests**

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_t2m_optimized_entrypoint -v
```

Expected: PASS without loading checkpoints or CUDA during import.

- [ ] **Step 6: Compile the entrypoint**

```bash
conda run -n mgpt python -m py_compile \
  eval_msa_t2m_rag_t5_optimized.py \
  utils/eval_msa_t2m_optimized.py
```

Expected: exit 0.

- [ ] **Step 7: Commit Task 4**

```bash
git add eval_msa_t2m_rag_t5_optimized.py \
  tests/test_eval_msa_t2m_optimized_entrypoint.py
git commit -m "feat: add optimized MSA-T2M evaluation entrypoint"
```

---

### Task 5: User-Facing Launcher and Isolation Verification

**Files:**
- Create: `EVAL_t2m_rag_t5_optimized.sh`
- Create: `tests/test_eval_msa_t2m_optimized_launcher.py`

**Interfaces:**
- Accepts the same environment overrides as `EVAL_t2m_rag_t5.sh`.
- Invokes only `eval_msa_t2m_rag_t5_optimized.py`.
- Uses distinct default experiment name
  `MotionStreamer_t2m_272_msa_rag_t5_trans662048_vaefulldb_k3_testcode_ema_optimized`.

- [ ] **Step 1: Write the failing launcher contract test**

Read the launcher as text and assert it:

- invokes the optimized Python entrypoint;
- preserves checkpoint, latent, text, RAG, T5, CFG, stop-threshold, and Top-K
  arguments;
- quotes every path-valued shell expansion;
- does not invoke the original Python entrypoint;
- has the distinct optimized experiment name.

- [ ] **Step 2: Run launcher test and verify RED**

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized_launcher -v
```

Expected: failure because the launcher is absent.

- [ ] **Step 3: Implement the launcher**

Use overridable environment variables matching the official launcher, activate
the existing `mgpt` environment when available, print the effective optimized
configuration, and invoke the new entrypoint with quoted arguments. Do not
claim multi-GPU support.

- [ ] **Step 4: Run launcher and shell syntax tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized_launcher -v
bash -n EVAL_t2m_rag_t5_optimized.sh
```

Expected: PASS and exit 0.

- [ ] **Step 5: Verify existing evaluation files are untouched**

```bash
git diff d6d29b4 -- \
  EVAL_t2m_rag_t5.sh \
  eval_msa_t2m_rag_t5.py \
  utils/eval_trans.py \
  humanml3d_272/dataset_eval_t2m.py
```

Expected: only changes that predate this implementation, with no new diff from
the optimized-evaluation commits. Confirm directly with:

```bash
git log --oneline -- \
  EVAL_t2m_rag_t5.sh \
  eval_msa_t2m_rag_t5.py \
  utils/eval_trans.py \
  humanml3d_272/dataset_eval_t2m.py
```

No Task 1-5 commit may appear.

- [ ] **Step 6: Run complete proportional validation**

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_t2m_optimized \
  tests.test_eval_msa_t2m_optimized_entrypoint \
  tests.test_eval_msa_t2m_optimized_launcher -v
conda run -n mgpt python -m py_compile \
  eval_msa_t2m_rag_t5_optimized.py \
  utils/eval_msa_t2m_optimized.py
bash -n EVAL_t2m_rag_t5_optimized.sh
git diff --check
```

Expected: all tests PASS and all commands exit 0.

- [ ] **Step 7: Commit Task 5**

```bash
git add EVAL_t2m_rag_t5_optimized.sh \
  tests/test_eval_msa_t2m_optimized_launcher.py
git commit -m "feat: launch optimized MSA-T2M evaluation"
```

---

### Task 6: Final Review and Handoff

**Files:**
- Review only: all Task 1-5 new files

**Interfaces:**
- No new interface; this task verifies the complete deliverable against the
  approved design.

- [ ] **Step 1: Review the diff scope**

```bash
git status --short
git diff origin/main...HEAD --stat
git diff origin/main...HEAD -- \
  EVAL_t2m_rag_t5_optimized.sh \
  eval_msa_t2m_rag_t5_optimized.py \
  utils/eval_msa_t2m_optimized.py \
  tests/test_eval_msa_t2m_optimized.py \
  tests/test_eval_msa_t2m_optimized_entrypoint.py \
  tests/test_eval_msa_t2m_optimized_launcher.py
```

Confirm unrelated untracked exploration directories are absent from every
commit and no data, checkpoint, log, or generated artifact is tracked.

- [ ] **Step 2: Re-run fresh verification**

Run the complete validation command from Task 5 Step 6 and record its exact
output for the handoff.

- [ ] **Step 3: Report operational expectations**

Document in the handoff:

- the exact optimized launcher command;
- that no full multi-hour GPU evaluation was run;
- that the expected acceleration comes from batch-32 text/retrieval and
  active-set generation;
- that stochastic metric values need protocol-level comparison across repeated
  runs rather than bitwise comparison;
- that BF16, KV cache, compilation, and multi-GPU remain outside scope.
