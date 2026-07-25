# MSA-T2M EOS Decode Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the continuous EOS latent from entering the MSA-VAE decoder in official evaluation and single-motion inference.

**Architecture:** Keep both existing sampler APIs and generation limits unchanged. In each autoregressive loop, classify the generated token before appending it; an EOS match terminates the loop, while a motion token is appended to the returned latent prefix.

**Tech Stack:** Python 3.8, PyTorch 2.4, `unittest`, existing `mgpt` conda environment.

## Global Constraints

- Do not change training targets, latent extraction, thresholds, archived explorations, model architecture, or checkpoint formats.
- Preserve both sampler return signatures.
- Keep all tests CPU-only and avoid checkpoint, dataset, or full evaluation runs.

---

### Task 1: Exclude EOS from both decoded latent sequences

**Files:**
- Create: `tests/test_msa_eos_stopping.py`
- Modify: `eval_msa_t2m_rag_t5.py:286-305`
- Modify: `msa_gen_motion.py:285-293`

**Interfaces:**
- Consumes: `RAGEvalSampler.sample_for_eval_CFG(...) -> torch.Tensor` and `sample_motion_latents_with_stop(...) -> torch.Tensor`
- Produces: the same tensors and shapes as before, except a token matching the configured EOS threshold is absent.

- [ ] **Step 1: Write the failing regression tests**

Create deterministic fake model, retriever, text lookup, and text encoder
objects. Make the fake model emit literal motion token `[1.0, 2.0]` and then
literal EOS token `[9.0, 9.0]`. For each production sampler, assert that the
returned tensor is exactly `[[[1.0, 2.0]]]`.

```python
class SequenceRAGModel:
    def __init__(self, tokens):
        self.tokens = [token.clone() for token in tokens]

    def sample_next_with_cfg(self, **_kwargs):
        return self.tokens.pop(0)


def test_eval_sampler_excludes_generated_eos_from_motion_latents():
    result = sampler.sample_for_eval_CFG("walk", length=8, unit_length=4)
    torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0]]]))


def test_single_motion_sampler_excludes_generated_eos_from_motion_latents():
    result = sample_motion_latents_with_stop(
        ...,
        reference_end=torch.tensor([9.0, 9.0]),
        stop_threshold=0.01,
        length=8,
        unit_len=4,
        token_latent_dim=2,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0]]]))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest -v tests.test_msa_eos_stopping
```

Expected: both assertions fail because the actual result contains a second
latent token `[9.0, 9.0]`.

- [ ] **Step 3: Implement stop-before-append**

In `RAGEvalSampler.sample_for_eval_CFG`, compute the reference distance on the
unmodified `[B, D]` `next_token`, update `finished`, and break when all batch
elements are finished. Only then unsqueeze and append:

```python
if self.enable_stopping and self.reference_end_latent is not None:
    dist_l2 = torch.linalg.norm(
        next_token - self.reference_end_latent.unsqueeze(0), dim=-1
    )
    finished = finished | (dist_l2 < self.stop_threshold)
    if torch.all(finished):
        break

next_token = next_token.unsqueeze(1)
xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)
```

In `sample_motion_latents_with_stop`, break after computing `distance_l2` and
before unsqueezing or appending:

```python
distance_l2 = torch.sqrt(torch.sum((next_token - reference_end) ** 2))
if distance_l2 < stop_threshold:
    break

next_token = next_token.unsqueeze(1)
xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
conda run -n mgpt python -m unittest -v tests.test_msa_eos_stopping
```

Expected: both tests pass.

- [ ] **Step 5: Run proportional repository validation**

Run:

```bash
conda run -n mgpt python -m py_compile \
  eval_msa_t2m_rag_t5.py msa_gen_motion.py tests/test_msa_eos_stopping.py
conda run -n mgpt python -m unittest discover -s tests -v
git diff --check
```

Expected: compilation succeeds, all repository tests pass, and the diff has no
whitespace errors.

- [ ] **Step 6: Commit the tested fix**

```bash
git add eval_msa_t2m_rag_t5.py msa_gen_motion.py tests/test_msa_eos_stopping.py
git commit -m "fix: exclude EOS latent from MSA decoding"
```
