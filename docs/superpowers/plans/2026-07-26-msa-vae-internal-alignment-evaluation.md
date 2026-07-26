# MSA-VAE Internal Alignment Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed MSA-VAE evaluator that reports direct SentenceT5-space global/local alignment, multi-positive internal MSA-T5 retrieval, and posterior-mean reconstruction realism, then rerun and collect the four completed single-seed pilot variants.

**Architecture:** Preserve the existing `msa-vae-standard-v2` evaluator as the external-TMR semantic-preservation and training-validation path. Add a separate internal-alignment evaluator that reuses checkpoint loading and reconstruction metrics, obtains deterministic `mu`, `global_proj(h_cls)`, and `local_proj(mu)` from `semantic_only=True`, and never sends `x_recon` through a text-retrieval path. Add an alignment-specific dataset view and metric module, then extend the pilot exploration with separate durable evaluation artifacts so the historical external-TMR results remain intact.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1+cu118, NumPy, SciPy, Matplotlib 3.4.3, `unittest`, Bash, conda environment `mgpt`, GNU Screen.

## Global Constraints

- Use the existing `mgpt` environment; do not upgrade Python, PyTorch, Accelerate, NumPy, or SciPy.
- Preserve the 272-dimensional HumanML3D representation, 30 FPS motion convention, causal latent convention, and 768-dimensional SentenceT5 contract.
- Keep the fixed TAE checkpoint `Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth` with SHA-256 `7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8`.
- Preserve the completed pilot's Phase-1/Phase-2 budget of 25,000 iterations per phase, seed 123, two training GPUs per variant, and Phase-2 `net_last.pth` comparison.
- Do not change MSA-VAE architecture, training losses, phase freezing, checkpoint tensor keys, or the built-in `msa-vae-standard-v2` training-validation behavior in this implementation.
- Do not overwrite the existing `Experiments/msa_vae_alignment_realism_pilot_s123_20260726/evaluation/` external-TMR artifacts.
- Main global/realism evaluation uses the deterministic HumanML3D `test.txt` set and must retain its verified 2,480-sample identity.
- Current local targets exist only for `train_ft.txt`; pilot local results must be named `in_sample_local_cosine` and labeled as a training-set diagnostic.
- A held-out local run must fail if any required local cache file is missing; complete-motion BABEL descriptions must never be repeated across frames as synthetic local targets.
- Do not use BABEL joint normalization or a BABEL TAE with HumanML checkpoints.
- Do not commit datasets, SentenceT5 arrays, checkpoints, logs, Screen output, or generated evaluation artifacts.
- Preserve unrelated untracked files and submodule working trees.

## File Structure

- Create `humanml3d_272/msa_text_targets.py`: the single source of truth for 20 FPS-to-motion resampling and latent-rate pooling.
- Modify `humanml3d_272/dataset_msa_vae.py`: call the shared target helper without changing returned training tensors.
- Modify `humanml3d_272/dataset_eval_msa_vae_metrics.py`: retain the legacy dataset and add an alignment-specific full-motion target view and collator.
- Create `utils/msa_vae_alignment_metrics.py`: pure global cosine, local masked cosine, and multi-positive retrieval calculations.
- Create `eval_msa_vae_alignment.py`: internal alignment, posterior-mean reconstruction, manifest, artifact, and CLI orchestration.
- Create `EVAL_msa_vae_alignment.sh`: authoritative `mgpt` launcher for one checkpoint.
- Create `tests/test_msa_text_targets.py`: temporal target contract tests.
- Modify `tests/test_dataset_msa_vae_full_sequence.py`: training-loader parity after helper extraction.
- Modify `tests/test_msa_vae_metrics_dataset.py`: alignment dataset, all-caption cache-row, target-hash, local-cache, and collation tests.
- Create `tests/test_msa_vae_alignment_metrics.py`: metric mathematics and negative-control tests.
- Create `tests/test_eval_msa_vae_alignment.py`: semantic-path, posterior-mean, batch invariance, fail-closed manifest, and motion-only TMR tests.
- Create `tests/test_msa_vae_alignment_launcher.py`: launcher argument/environment tests.
- Modify `explorations/msa_vae_alignment_realism/pilot.py`: add a versioned internal protocol validator, table/delta collector, and Pareto plotting without removing the external protocol collector.
- Create `explorations/msa_vae_alignment_realism/eval_internal_variant.sh`: evaluate one completed variant into a new directory.
- Create `explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh`: launch four one-GPU evaluation Screen sessions.
- Modify `explorations/msa_vae_alignment_realism/STATUS_PILOT.sh`: show internal-evaluation states.
- Modify `explorations/msa_vae_alignment_realism/README.md`: document internal versus external tables and the in-sample local limitation.
- Modify `tests/test_msa_vae_alignment_pilot.py`: internal manifest, collector, runner, and Screen orchestration regression tests.

---

### Task 1: Share the Exact Local SentenceT5 Temporal Transformation

**Files:**
- Create: `humanml3d_272/msa_text_targets.py`
- Create: `tests/test_msa_text_targets.py`
- Modify: `humanml3d_272/dataset_msa_vae.py:229-240,372-381`
- Modify: `tests/test_dataset_msa_vae_full_sequence.py:13-95`

**Interfaces:**
- Produces `build_local_text_target(local_text, raw_motion_length, view_start, view_length, latent_length, expected_dim) -> Tuple[np.ndarray, np.ndarray]`.
- The tuple is `(latent_target, view_pooled_target)`, both `float32`.
- Training and alignment evaluation consume the same helper.

- [ ] **Step 1: Write failing unit tests for resampling, cropping, pooling, and validation**

Create `tests/test_msa_text_targets.py` with:

```python
import unittest

import numpy as np

from humanml3d_272.msa_text_targets import build_local_text_target


class LocalTextTargetTest(unittest.TestCase):
    def test_resamples_to_raw_motion_then_crops_and_pools(self):
        source = np.array(
            [[0.0, 0.0], [10.0, -10.0], [20.0, -20.0], [30.0, -30.0]],
            dtype=np.float32,
        )

        latent, pooled = build_local_text_target(
            source,
            raw_motion_length=6,
            view_start=0,
            view_length=4,
            latent_length=2,
            expected_dim=2,
        )

        np.testing.assert_allclose(
            latent,
            np.array([[5.0, -5.0], [15.0, -15.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            pooled,
            np.array([10.0, -10.0], dtype=np.float32),
        )

    def test_identity_rate_preserves_training_window_alignment(self):
        source = np.stack(
            [np.arange(70, dtype=np.float32), -np.arange(70, dtype=np.float32)],
            axis=1,
        )

        latent, pooled = build_local_text_target(
            source,
            raw_motion_length=70,
            view_start=4,
            view_length=64,
            latent_length=16,
            expected_dim=2,
        )

        np.testing.assert_allclose(latent[0], np.array([5.5, -5.5]))
        np.testing.assert_allclose(pooled, np.array([35.5, -35.5]))

    def test_rejects_invalid_shape_length_crop_and_dimension(self):
        valid = np.ones((4, 2), dtype=np.float32)
        cases = (
            (valid[:, 0], 4, 0, 4, 1, 2, "2D"),
            (valid, 0, 0, 4, 1, 2, "raw_motion_length"),
            (valid, 4, 3, 2, 1, 2, "view"),
            (valid, 4, 0, 4, 0, 2, "latent_length"),
            (valid, 4, 0, 4, 1, 3, "dimension"),
        )
        for value, raw, start, length, latent, dim, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_local_text_target(
                        value,
                        raw_motion_length=raw,
                        view_start=start,
                        view_length=length,
                        latent_length=latent,
                        expected_dim=dim,
                    )

        non_finite = valid.copy()
        non_finite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_local_text_target(
                non_finite,
                raw_motion_length=4,
                view_start=0,
                view_length=4,
                latent_length=1,
                expected_dim=2,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_text_targets -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'humanml3d_272.msa_text_targets'`.

- [ ] **Step 3: Implement the shared transformation**

Create `humanml3d_272/msa_text_targets.py` with these public rules:

```python
"""Shared temporal preprocessing for HumanML3D/BABEL local text targets."""

import numpy as np


def _pool_to_latent(text_window, latent_length):
    if text_window.shape[0] == latent_length:
        return text_window.astype(np.float32, copy=False)
    boundaries = np.linspace(
        0,
        text_window.shape[0],
        latent_length + 1,
    ).astype(int)
    pooled = np.empty(
        (latent_length, text_window.shape[1]),
        dtype=np.float32,
    )
    for index in range(latent_length):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            raise ValueError("latent pooling produced an empty interval")
        pooled[index] = text_window[start:end].mean(axis=0)
    return pooled


def build_local_text_target(
    local_text,
    raw_motion_length,
    view_start,
    view_length,
    latent_length,
    expected_dim,
):
    local_text = np.asarray(local_text, dtype=np.float32)
    raw_motion_length = int(raw_motion_length)
    view_start = int(view_start)
    view_length = int(view_length)
    latent_length = int(latent_length)
    expected_dim = int(expected_dim)
    if local_text.ndim != 2:
        raise ValueError("local text target must be a 2D array")
    if local_text.shape[0] < 1:
        raise ValueError("local text target must contain at least one frame")
    if local_text.shape[1] != expected_dim:
        raise ValueError("local text target dimension does not match expected dimension")
    if not np.isfinite(local_text).all():
        raise ValueError("local text target contains non-finite values")
    if raw_motion_length < 1:
        raise ValueError("raw_motion_length must be positive")
    if view_start < 0 or view_length < 1 or view_start + view_length > raw_motion_length:
        raise ValueError("view must lie inside the raw motion")
    if latent_length < 1 or latent_length > view_length:
        raise ValueError("latent_length must be between one and view_length")

    frame_indices = np.round(
        np.linspace(0, local_text.shape[0] - 1, raw_motion_length)
    ).astype(int)
    motion_rate = local_text[frame_indices]
    view = motion_rate[view_start:view_start + view_length]
    pooled = view.mean(axis=0).astype(np.float32)
    latent = _pool_to_latent(view, latent_length)
    return latent.astype(np.float32), pooled
```

- [ ] **Step 4: Replace the duplicate training implementation**

In `humanml3d_272/dataset_msa_vae.py`, import `build_local_text_target`, replace lines 233-240 with:

```python
local_text_20 = np.load(entry["local_text_path"])
local_text_latent, local_text_pooled = build_local_text_target(
    local_text_20,
    raw_motion_length=len(motion),
    view_start=idx,
    view_length=motion_length,
    latent_length=latent_len,
    expected_dim=self.text_embed_dim,
)
```

Delete the private `_pool_to_latent` function. Do not change the nine-element dataset return tuple.

- [ ] **Step 5: Prove training-loader parity**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_text_targets \
  tests.test_dataset_msa_vae_full_sequence -v
```

Expected: all tests PASS, including the existing `[5.5, -5.5]` first-window target assertion.

- [ ] **Step 6: Commit the shared preprocessing**

```bash
git add humanml3d_272/msa_text_targets.py \
  humanml3d_272/dataset_msa_vae.py \
  tests/test_msa_text_targets.py \
  tests/test_dataset_msa_vae_full_sequence.py
git commit -m "refactor: share MSA local target preprocessing"
```

---

### Task 2: Add an Alignment-Specific Evaluation Dataset

**Files:**
- Modify: `humanml3d_272/dataset_eval_msa_vae_metrics.py:14-184`
- Modify: `tests/test_msa_vae_metrics_dataset.py:1-169`

**Interfaces:**
- Preserves `MSAVAEMetricsDataset` and `make_msa_vae_metrics_loader` behavior for training validation and external-TMR evaluation.
- Produces the fully specified `MSAVAEAlignmentDataset` constructor documented in Step 4, with `target_mode` restricted to `"global"` or `"local"`.
- Produces `collate_msa_vae_alignment(batch) -> dict`.
- Produces `make_msa_vae_alignment_loader(dataset, batch_size, num_workers, pin_memory)`.
- Every alignment dataset exposes `sample_ids`, `sample_hash`, `target_hash`, and `target_mode`.

- [ ] **Step 1: Extend fixture creation to include target-cache directories**

In `tests/test_msa_vae_metrics_dataset.py`, add `global_text` and `local_text` directories in `setUp`, then add a test that writes:

```python
np.save(
    self.root / "global_text" / "motion_b.npy",
    np.array(
        [
            [100.0, 101.0],
            [200.0, 201.0],
            [300.0, 301.0],
        ],
        dtype=np.float32,
    ),
)
np.save(
    self.root / "local_text" / "motion_b.npy",
    np.arange(128, dtype=np.float32).reshape(64, 2),
)
```

Use caption rows:

```text
segment#tok#0.5#1.5
first full#tok#0#0
second full#tok#0#0
```

Assert:

```python
global_dataset = MSAVAEAlignmentDataset(
    data_root=self.root,
    split_file=self.split_file,
    unit_length=4,
    target_mode="global",
    text_embed_dim=2,
    global_text_embed_dir=self.root / "global_text",
)
item = global_dataset[0]
self.assertEqual(item["all_captions"], ("first full", "second full"))
self.assertEqual(item["caption_line_indices"], (1, 2))
torch.testing.assert_close(
    item["global_text_embeddings"],
    torch.tensor([[200.0, 201.0], [300.0, 301.0]]),
)
self.assertEqual(global_dataset.target_mode, "global")
self.assertEqual(len(global_dataset.target_hash), 64)
```

Add a local-mode assertion that the target length equals `motion_length // 4`, and tests that missing files, insufficient global rows, a 767-dimensional target, and non-finite values raise `ValueError` or `FileNotFoundError` rather than producing zeros.

- [ ] **Step 2: Run the dataset test and verify the missing-class failure**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_metrics_dataset -v
```

Expected: FAIL because `MSAVAEAlignmentDataset` is not defined.

- [ ] **Step 3: Preserve every complete caption and its zero-based source row**

Add:

```python
@dataclass(frozen=True)
class CompleteCaption:
    text: str
    line_index: int
```

Replace `_first_complete_caption` internally with:

```python
def _complete_captions(text_path):
    captions = []
    with text_path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("#")
            if len(fields) < 4:
                raise ValueError(
                    "{}:{} must contain caption#tokens#from#to".format(
                        text_path,
                        line_index + 1,
                    )
                )
            if (
                _as_complete_tag(fields[2]) == 0.0
                and _as_complete_tag(fields[3]) == 0.0
            ):
                caption = fields[0].strip()
                if caption:
                    captions.append(CompleteCaption(caption, line_index))
    return tuple(captions)
```

Store the tuple on `EvaluationRecord`, retain a `caption` property that returns the first text, and keep legacy `__getitem__` output unchanged.

- [ ] **Step 4: Implement strict global and local target modes**

Add `MSAVAEAlignmentDataset` as a subclass of `MSAVAEMetricsDataset`.

Constructor contract:

```python
MSAVAEAlignmentDataset(
    data_root,
    split_file,
    unit_length,
    target_mode,
    text_embed_dim=768,
    global_text_embed_dir=None,
    local_text_embed_dir=None,
    min_motion_length=60,
    max_motion_length=300,
)
```

Rules:

- `target_mode="global"` requires a global cache directory and all referenced caption rows.
- `target_mode="local"` requires a local cache file for every retained sample.
- global arrays must be two-dimensional, finite, and have `text_embed_dim` columns;
- local arrays pass through `build_local_text_target` with `view_start=0`, `view_length=record.length`, `raw_motion_length=record.raw_length`, and `latent_length=record.length // unit_length`;
- `target_hash` is SHA-256 over ordered sample ID, selected line indices, array shape, dtype, and selected float32 bytes;
- no missing target is converted to a zero tensor.

- [ ] **Step 5: Implement a separate alignment collator**

`collate_msa_vae_alignment` must first pad motions exactly like the legacy collator.

For global batches, return:

```python
{
    "sample_ids": list[str],
    "motions": torch.Tensor,
    "lengths": torch.LongTensor,
    "all_captions": Sequence[Sequence[str]],
    "caption_line_indices": Sequence[Sequence[int]],
    "global_text_embeddings": list[torch.Tensor],
}
```

For local batches, return padded `local_text_embeddings` of shape `(B, Lmax, D)` and `local_mask` of shape `(B, Lmax)`, where the first `length // unit_length` positions are true.

- [ ] **Step 6: Run legacy and alignment dataset tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_metrics_dataset \
  tests.test_msa_vae_deterministic_validation \
  tests.test_eval_msa_vae_metrics -v
```

Expected: all tests PASS; the legacy first-caption data contract remains unchanged.

- [ ] **Step 7: Commit the dataset view**

```bash
git add humanml3d_272/dataset_eval_msa_vae_metrics.py \
  tests/test_msa_vae_metrics_dataset.py
git commit -m "feat: add MSA alignment evaluation targets"
```

---

### Task 3: Implement Multi-Positive Internal Alignment Metrics

**Files:**
- Create: `utils/msa_vae_alignment_metrics.py`
- Create: `tests/test_msa_vae_alignment_metrics.py`

**Interfaces:**
- Produces `calculate_motion_macro_cosine(motion_embeddings, text_embeddings, text_motion_indices) -> float`.
- Produces `calculate_masked_local_cosine(local_embeddings, local_targets, valid_mask) -> float`.
- Produces `calculate_msa_t5_retrieval(text_embeddings, motion_embeddings, text_motion_indices) -> Dict[str, float]`.
- Produces `shuffled_text_control(text_embeddings, seed) -> torch.Tensor`.

- [ ] **Step 1: Write exact metric tests**

Create tests covering:

```python
def test_motion_macro_cosine_does_not_overweight_extra_captions(self):
    motion = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    owners = torch.tensor([0, 0, 1])
    value = calculate_motion_macro_cosine(motion, text, owners)
    self.assertAlmostEqual(value, 0.0)


def test_multi_positive_m2t_accepts_either_caption(self):
    motion = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor(
        [[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    owners = torch.tensor([0, 0, 1])
    metrics = calculate_msa_t5_retrieval(text, motion, owners)
    self.assertEqual(metrics["msa_t5_t2m_r1_percent"], 100.0)
    self.assertEqual(metrics["msa_t5_m2t_r1_percent"], 100.0)
    self.assertEqual(metrics["msa_t5_m2t_medr"], 1.0)


def test_local_cosine_is_motion_macro_not_token_micro(self):
    prediction = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, True], [True, False, False]])
    value = calculate_masked_local_cosine(prediction, target, mask)
    self.assertAlmostEqual(value, 0.0)
```

Also test:

- average-rank handling for one positive tied with negatives;
- best-positive M2T rank does not penalize a second valid caption;
- owner indices outside `[0, motion_count)` fail;
- every motion must own at least one global caption;
- empty local masks, zero-norm valid vectors, shape mismatches, and any
  non-finite values fail; zero padding outside `valid_mask` is allowed;
- `shuffled_text_control` is deterministic, changes ownership pairing, and preserves rows.
- `shuffled_text_control` rejects fewer than two caption rows because a
  non-identity control cannot then be constructed.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_metrics -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement validated normalization and macro cosine**

Use float32 PyTorch tensors, reject zero-norm target or prediction rows that
participate in the metric, normalize with `torch.nn.functional.normalize`, and
average captions/tokens within each motion before the dataset mean.

Do not reuse the square-diagonal TMR helper because the new caption-to-motion matrix is rectangular.

- [ ] **Step 4: Implement multi-positive ranks**

For T2M, each caption has exactly one positive motion. Its zero-based average-tie rank is:

```python
positive_score = row[owner]
better = torch.count_nonzero(row > positive_score)
tied_negatives = torch.count_nonzero(row == positive_score) - 1
rank = better.float() + tied_negatives.float() / 2.0
```

For M2T, choose the highest-scoring caption among all captions owned by that motion, then count only negative captions above or tied with that score. Other positive captions must not worsen the rank.

Return R@1, R@2, R@3, R@5, and median rank with `msa_t5_t2m_` and `msa_t5_m2t_` prefixes.

- [ ] **Step 5: Implement the deterministic shuffled control**

Use a local `torch.Generator().manual_seed(seed)` and reject permutations with fixed ownership for every caption. Return permuted text rows without mutating the input. The evaluator records this retrieval result under diagnostics; it is not a main metric.

- [ ] **Step 6: Run the pure metric suite**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_metrics \
  tests.test_msa_vae_metrics -v
```

Expected: all tests PASS and the legacy square TMR metrics remain unchanged.

- [ ] **Step 7: Commit the internal metric module**

```bash
git add utils/msa_vae_alignment_metrics.py \
  tests/test_msa_vae_alignment_metrics.py
git commit -m "feat: add internal MSA-T5 alignment metrics"
```

---

### Task 4: Build the Internal Alignment Evaluator

**Files:**
- Create: `eval_msa_vae_alignment.py`
- Create: `tests/test_eval_msa_vae_alignment.py`

**Interfaces:**
- Produces `evaluate_global_alignment_and_realism(model, motion_encoder, loader, device, skating_config) -> dict`.
- Produces `evaluate_local_alignment(model, loader, device) -> dict`.
- Produces `build_alignment_result_manifest(metrics, diagnostics, checkpoint, evaluator, resolved_config, global_dataset, local_dataset, local_scope, seed, batch_size, skating_config) -> Dict[str, Any]`.
- Produces `write_alignment_result_artifacts(manifest, output_dir) -> str`.
- CLI writes `metrics.json`, `metrics.csv`, and `evaluation.log`.
- Protocol version is exactly `msa-vae-internal-alignment-v1`.

- [ ] **Step 1: Write a fake semantic model that makes sampling misuse observable**

In `tests/test_eval_msa_vae_alignment.py`, define:

```python
class _FakeSemanticMSAVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_calls = 0
        self.decoder_inputs = []

    def forward(self, motions, lengths=None, semantic_only=False):
        if not semantic_only:
            raise AssertionError("internal evaluation must use semantic_only=True")
        self.semantic_calls += 1
        labels = motions[:, 0, 100].round().long()
        global_features = torch.nn.functional.one_hot(
            labels,
            num_classes=3,
        ).float()
        mu = motions[:, ::4, :2].clone()
        local_features = torch.nn.functional.pad(mu, (0, 1))
        return {
            "mu": mu,
            "clip_global_feat": global_features,
            "clip_local_feat": local_features,
        }

    def forward_decoder(self, latent):
        self.decoder_inputs.append(latent.detach().clone())
        batch, latent_length, _ = latent.shape
        prediction = torch.zeros(batch, latent_length * 4, 272)
        prediction[:, :, :2] = latent.repeat_interleave(4, dim=1)
        return prediction
```

Assert that:

- `forward_decoder` receives exactly `mu`;
- changing an unavailable `x_recon` cannot affect internal retrieval;
- repeated runs are bitwise identical;
- batch sizes 1 and 3 produce identical alignment and realism metrics.

- [ ] **Step 2: Write local-scope and fail-closed tests**

Build a local loader whose targets equal `clip_local_feat` and assert `1.0` cosine. Then change only padding target values and assert the result is unchanged.

Add manifest tests:

```python
self.assertEqual(
    manifest["protocol"]["version"],
    "msa-vae-internal-alignment-v1",
)
self.assertEqual(
    manifest["protocol"]["reconstruction_decode"],
    "posterior_mean",
)
self.assertEqual(
    manifest["protocol"]["retrieval"],
    "MSA-global-projection-to-SentenceT5-multi-positive",
)
self.assertEqual(
    manifest["local_alignment"]["scope"],
    "in_sample",
)
self.assertIsNone(manifest["metrics"]["local_cosine"])
self.assertEqual(
    manifest["metrics"]["in_sample_local_cosine"],
    1.0,
)
```

Require `held_out` scope to populate `local_cosine` and leave `in_sample_local_cosine` null.

- [ ] **Step 3: Run the evaluator tests and verify the missing-module failure**

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_vae_alignment -v
```

Expected: FAIL because `eval_msa_vae_alignment.py` does not exist.

- [ ] **Step 4: Implement the global/realism pass**

For every batch:

```python
semantic = model(motions, lengths=lengths, semantic_only=True)
predictions = model.forward_decoder(semantic["mu"])
global_features = semantic["clip_global_feat"]
```

Validate all required keys, exact batch/time dimensions, feature dimensions, and finiteness.

Use the frozen external TMR **motion encoder only** for reference/prediction FID. Do not construct DistilBERT and do not call a text encoder.

Collect every cached caption feature and an owner index offset by the number of previously evaluated motions. Compute:

- `global_cosine`;
- `msa_t5_t2m_r1/r2/r3/r5/medr`;
- `msa_t5_m2t_r1/r2/r3/r5/medr`;
- FID, MPJPE, P-MPJPE, ACCEL, and skating.

Mask prediction padding before the TMR motion encoder and slice valid frames before joint recovery.

- [ ] **Step 5: Implement the separate local pass**

Call only:

```python
semantic = model(motions, lengths=lengths, semantic_only=True)
local_features = semantic["clip_local_feat"]
```

Compare against `local_text_embeddings` using `local_mask`. Return `local_sample_count`, `local_token_count`, and local cosine. Do not decode motion and do not calculate local retrieval.

- [ ] **Step 6: Implement motion-only TMR loading and CLI preflight**

The CLI accepts:

```text
checkpoint
--data-root
--split-file
--global-text-embed-dir
--local-split-file
--local-text-embed-dir
--local-target-scope {held-out,in-sample}
--evaluator-root
--evaluator-checkpoint
--output-dir
--device
--batch-size
--num-workers
--seed
```

It also accepts the same optional architecture overrides as
`eval_msa_vae_metrics.py`.

Argument rules:

- `--local-split-file`, `--local-text-embed-dir`, and `--local-target-scope` must be supplied together;
- `--local-target-scope in-sample` writes only `in_sample_local_cosine`;
- `--local-target-scope held-out` writes only `local_cosine`;
- missing evaluator motion source/checkpoint, normalization, split, motion, text, or target cache fails before loading the model;
- default global cache is `humanml3d_272/text_latents_t5`;
- no DistilBERT path is required.

- [ ] **Step 7: Implement the manifest and artifact schema**

The manifest records:

```text
protocol.version
protocol.retrieval
protocol.caption_policy
protocol.reconstruction_decode
checkpoint.path/sha256/metadata
evaluator.path/sha256
model_config.values/sources
global_realism_dataset.sample_count/sample_ids/sample_hash/target_directory/target_hash/caption_count
local_alignment.scope/sample_count/sample_ids/sample_hash/target_directory/target_hash/token_count
metrics
diagnostics.shuffled_global_retrieval
seed
batch_size
skating
```

`metrics.csv` contains flat main metrics and blank cells for the non-applicable local-cosine field. `evaluation.log` must call the retrieval rows `MSA-T5`, not `TMR`.

- [ ] **Step 8: Run evaluator, dataset, and checkpoint-loading tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_eval_msa_vae_alignment \
  tests.test_msa_vae_metrics_dataset \
  tests.test_msa_vae_alignment_metrics \
  tests.test_msa_vae_eval_config -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit the evaluator**

```bash
git add eval_msa_vae_alignment.py \
  tests/test_eval_msa_vae_alignment.py
git commit -m "feat: evaluate internal MSA-VAE alignment"
```

---

### Task 5: Add the Authoritative Launcher

**Files:**
- Create: `EVAL_msa_vae_alignment.sh`
- Create: `tests/test_msa_vae_alignment_launcher.py`

**Interfaces:**
- Produces `bash EVAL_msa_vae_alignment.sh CHECKPOINT [options]`.
- Uses the current environment when `CONDA_DEFAULT_ENV=mgpt`; otherwise runs through `conda run -n mgpt`.

- [ ] **Step 1: Write launcher capture tests**

Mirror the existing standard launcher tests but assert that the executable is
`eval_msa_vae_alignment.py`. Cover:

- inactive conda environment;
- active `mgpt` environment;
- exact forwarding of local split/cache/scope arguments;
- missing or option-like checkpoint exits 2 without launching Python.

- [ ] **Step 2: Run the test and verify the missing-launcher failure**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_launcher -v
```

Expected: FAIL because `EVAL_msa_vae_alignment.sh` does not exist.

- [ ] **Step 3: Implement the launcher**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "Usage: bash EVAL_msa_vae_alignment.sh CHECKPOINT.pth [evaluation options]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" == "mgpt" ]]; then
  exec python "$SCRIPT_DIR/eval_msa_vae_alignment.py" "$@"
fi

exec conda run -n mgpt python "$SCRIPT_DIR/eval_msa_vae_alignment.py" "$@"
```

- [ ] **Step 4: Validate launcher and CLI help**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_launcher -v
bash -n EVAL_msa_vae_alignment.sh
conda run -n mgpt python eval_msa_vae_alignment.py --help
```

Expected: tests PASS, Bash syntax PASS, help exits 0 and lists `--local-target-scope`.

- [ ] **Step 5: Commit the launcher**

```bash
git add EVAL_msa_vae_alignment.sh \
  tests/test_msa_vae_alignment_launcher.py
git commit -m "feat: launch MSA alignment evaluation"
```

---

### Task 6: Integrate the New Protocol with the Four-Variant Pilot

**Files:**
- Modify: `explorations/msa_vae_alignment_realism/pilot.py:33-60,518-582,692-835,838-906`
- Create: `explorations/msa_vae_alignment_realism/eval_internal_variant.sh`
- Create: `explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh`
- Modify: `explorations/msa_vae_alignment_realism/STATUS_PILOT.sh`
- Modify: `explorations/msa_vae_alignment_realism/README.md`
- Modify: `tests/test_msa_vae_alignment_pilot.py`

**Interfaces:**
- Preserves `collect` and the existing `evaluation/{slug}` external-TMR supplementary artifacts, where `slug` is one of `no_align`, `global_only`, `local_only`, or `global_local`.
- Produces `validate_internal_pilot_manifests(output_root)`.
- Produces `write_internal_pilot_table(output_root)`.
- Adds CLI command `collect-internal`.
- Reads new manifests from `evaluation_internal/{slug}/metrics.json`.
- Writes `summary/internal_alignment_pilot_table.json`, `.csv`, and `.md`.
- Writes `summary/internal_alignment_deltas.json` with absolute and relative changes from No Alignment.
- Writes four SVG plots for global/local alignment against FID/MPJPE.

- [ ] **Step 1: Add internal protocol fixtures and collector tests**

Add:

```python
INTERNAL_PROTOCOL_VERSION = "msa-vae-internal-alignment-v1"
INTERNAL_TARGET_METRICS = (
    "global_cosine",
    "in_sample_local_cosine",
    "fid",
    "mpjpe_mm",
    "p_mpjpe_mm",
    "accel_mm_per_frame2",
    "skating_percent",
    "msa_t5_t2m_r1_percent",
    "msa_t5_t2m_r5_percent",
    "msa_t5_t2m_medr",
    "msa_t5_m2t_r1_percent",
    "msa_t5_m2t_r5_percent",
    "msa_t5_m2t_medr",
)
```

Internal manifest fixtures must include:

```python
"protocol": {
    "version": "msa-vae-internal-alignment-v1",
    "retrieval": "MSA-global-projection-to-SentenceT5-multi-positive",
    "caption_policy": "all complete-motion captions; multi-positive M2T",
    "reconstruction_decode": "posterior_mean",
},
"global_realism_dataset": {
    "sample_count": 2480,
    "sample_ids": ["sample-0001", "sample-0002"],
    "sample_hash": "global-sample-hash",
    "target_directory": "/data/humanml3d_272/text_latents_t5",
    "target_hash": "g" * 64,
    "caption_count": 7000,
},
"local_alignment": {
    "scope": "in_sample",
    "split": "train_ft.txt",
    "sample_count": 6000,
    "sample_ids": ["local-0001", "local-0002"],
    "sample_hash": "local-sample-hash",
    "target_directory": "/data/humanml3d_272/t5_enc_single",
    "target_hash": "l" * 64,
    "token_count": 100000,
},
```

Assert that internal collection rejects:

- the external v2 protocol;
- stochastic decode;
- missing target hashes;
- a local `held_out` scope containing only `in_sample_local_cosine`;
- mismatched global/local identities across variants;
- best checkpoints;
- missing shuffled-control diagnostics;
- non-finite metrics.

For a valid four-manifest fixture, assert:

```python
fixture_no_alignment_fid = 1.0
fixture_global_fid = 2.0
paths = write_internal_pilot_table(output_root)
self.assertTrue(paths["json"].is_file())
self.assertTrue(paths["csv"].is_file())
self.assertTrue(paths["markdown"].is_file())
self.assertTrue(paths["deltas"].is_file())
self.assertTrue(paths["global_fid_plot"].is_file())
self.assertTrue(paths["global_mpjpe_plot"].is_file())
self.assertTrue(paths["local_fid_plot"].is_file())
self.assertTrue(paths["local_mpjpe_plot"].is_file())
deltas = json.loads(paths["deltas"].read_text(encoding="utf-8"))
self.assertEqual(deltas["baseline"], "No Alignment")
self.assertEqual(
    deltas["variants"]["Global Only"]["fid"]["absolute_delta"],
    fixture_global_fid - fixture_no_alignment_fid,
)
```

Set the No Alignment and Global Only fixture FID values to these two numbers
before calling the writer.

- [ ] **Step 2: Run the pilot tests and verify missing internal symbols**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_pilot -v
```

Expected: FAIL because the internal validator and collector do not exist.

- [ ] **Step 3: Add the internal validator and table writer**

Use headers:

```text
Variant
Global Cos↑
Local Cos↑ (train diagnostic)
FID↓
MPJPE↓
P-MPJPE↓
ACCEL↓
Skating%↓
MSA-T5 T2M R@1↑
MSA-T5 T2M R@5↑
MSA-T5 T2M MedR↓
MSA-T5 M2T R@1↑
MSA-T5 M2T R@5↑
MSA-T5 M2T MedR↓
```

The JSON qualification must be:

```text
single-seed pilot; local cosine is an in-sample train_ft diagnostic; no uncertainty estimate
```

For every metric, store:

```python
absolute_delta = variant_value - no_alignment_value
relative_delta = (
    absolute_delta / abs(no_alignment_value)
    if no_alignment_value != 0.0
    else None
)
```

Use the non-interactive Matplotlib `Agg` backend. Annotate all four variants
and write:

```text
summary/global_cosine_vs_fid.svg
summary/global_cosine_vs_mpjpe.svg
summary/in_sample_local_cosine_vs_fid.svg
summary/in_sample_local_cosine_vs_mpjpe.svg
```

Each plot uses alignment on the horizontal axis and the downward realism metric
on the vertical axis; do not combine global and local alignment into one score.

Do not modify the existing external table function or its paths.

- [ ] **Step 4: Write the one-variant internal runner**

`eval_internal_variant.sh` must:

- accept `SLUG GPU CHECKPOINT`;
- refuse a missing checkpoint;
- refuse an existing `evaluation_internal/{slug}` directory;
- set `CUDA_VISIBLE_DEVICES` to the assigned single evaluation GPU;
- invoke `eval_msa_vae_alignment.py`;
- use global split `humanml3d_272/split/test.txt`;
- use global cache `humanml3d_272/text_latents_t5`;
- use local diagnostic split `humanml3d_272/split/train_ft.txt`;
- use local cache `humanml3d_272/t5_enc_single`;
- pass `--local-target-scope in-sample`;
- use batch size 32, eight workers, and seed 123;
- atomically write `status/{slug}.internal_evaluation.status`.

- [ ] **Step 5: Write the four-Screen orchestrator**

`EVAL_INTERNAL_PILOT.sh` must preserve the variant-to-evaluation-GPU mapping
`0, 2, 4, 6`, verify all training checkpoint lineages through `pilot.py verify`,
reject existing Screen names or output directories, and launch:

```text
msa_internal_eval_no_align_s123
msa_internal_eval_global_only_s123
msa_internal_eval_local_only_s123
msa_internal_eval_global_local_s123
```

Support `PILOT_DRY_RUN=1` and the existing `SCREEN_BIN`/`NVIDIA_SMI_BIN`
test overrides.

- [ ] **Step 6: Update status and documentation**

`STATUS_PILOT.sh` prints internal evaluation state after training and external
evaluation state. README commands are:

```bash
PILOT_DRY_RUN=1 \
bash explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh

bash explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh

conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect-internal
```

The README must explicitly state that the existing external table is
supplementary and the current local value is not held-out.

- [ ] **Step 7: Run pilot integration and Bash tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_alignment_pilot -v
bash -n \
  explorations/msa_vae_alignment_realism/eval_internal_variant.sh \
  explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh \
  explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
```

Expected: all tests and syntax checks PASS.

- [ ] **Step 8: Commit pilot integration**

```bash
git add explorations/msa_vae_alignment_realism/pilot.py \
  explorations/msa_vae_alignment_realism/eval_internal_variant.sh \
  explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh \
  explorations/msa_vae_alignment_realism/STATUS_PILOT.sh \
  explorations/msa_vae_alignment_realism/README.md \
  tests/test_msa_vae_alignment_pilot.py
git commit -m "feat: evaluate pilot internal alignment"
```

---

### Task 7: Verify the Implementation Without Starting Training

**Files:**
- Verify all files changed in Tasks 1-6.

**Interfaces:**
- Confirms source correctness before the authorized full four-checkpoint evaluation.

- [ ] **Step 1: Run focused unit tests**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_text_targets \
  tests.test_dataset_msa_vae_full_sequence \
  tests.test_msa_vae_metrics_dataset \
  tests.test_msa_vae_alignment_metrics \
  tests.test_eval_msa_vae_alignment \
  tests.test_msa_vae_alignment_launcher \
  tests.test_msa_vae_alignment_pilot -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run regression tests for untouched standard evaluation**

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_metrics \
  tests.test_eval_msa_vae_metrics \
  tests.test_msa_vae_deterministic_validation \
  tests.test_msa_vae_metrics_launcher \
  tests.test_msa_vae_training_entrypoint -v
```

Expected: all tests PASS; training validation still uses standard v2.

- [ ] **Step 3: Run static validation**

```bash
conda run -n mgpt python -m py_compile \
  humanml3d_272/msa_text_targets.py \
  humanml3d_272/dataset_msa_vae.py \
  humanml3d_272/dataset_eval_msa_vae_metrics.py \
  utils/msa_vae_alignment_metrics.py \
  eval_msa_vae_alignment.py \
  explorations/msa_vae_alignment_realism/pilot.py
bash -n \
  EVAL_msa_vae_alignment.sh \
  explorations/msa_vae_alignment_realism/eval_internal_variant.sh \
  explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh \
  explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Run a real-data target preflight without loading CUDA models**

```bash
conda run -n mgpt python -c "
from pathlib import Path
from humanml3d_272.dataset_eval_msa_vae_metrics import MSAVAEAlignmentDataset
root = Path('humanml3d_272')
global_data = MSAVAEAlignmentDataset(
    root,
    split_file=root / 'split' / 'test.txt',
    unit_length=4,
    target_mode='global',
    text_embed_dim=768,
    global_text_embed_dir=root / 'text_latents_t5',
)
local_data = MSAVAEAlignmentDataset(
    root,
    split_file=root / 'split' / 'train_ft.txt',
    unit_length=4,
    target_mode='local',
    text_embed_dim=768,
    local_text_embed_dir=root / 't5_enc_single',
)
print('global_samples', len(global_data))
print('global_captions', global_data.caption_count)
print('global_hash', global_data.target_hash)
print('local_samples', len(local_data))
print('local_hash', local_data.target_hash)
"
```

Expected:

- `global_samples 2480`;
- positive `global_captions`;
- 64-character global/local hashes;
- positive local sample count after complete-motion filtering.

- [ ] **Step 5: Verify held-out local mode fails with current assets**

Run the same dataset constructor with
`split_file=humanml3d_272/split/test_ft.txt` and
`local_text_embed_dir=humanml3d_272/t5_enc_single`.

Expected: nonzero exit with a missing-local-target error. This is the required
guard against silently reporting an invalid held-out local metric.

- [ ] **Step 6: Review the final source diff**

```bash
git status --short
git log --oneline -8
git diff HEAD~6 -- \
  humanml3d_272 \
  utils/msa_vae_alignment_metrics.py \
  eval_msa_vae_alignment.py \
  EVAL_msa_vae_alignment.sh \
  explorations/msa_vae_alignment_realism \
  tests
```

Confirm only intended source/test/docs files changed and no local artifacts are staged.

---

### Task 8: Run and Collect the Four Completed Checkpoints

**Files:**
- Runtime artifacts only under `Experiments/msa_vae_alignment_realism_pilot_s123_20260726/evaluation_internal/`, `status/`, `logs/`, and `summary/`; none are committed.

**Interfaces:**
- Produces four `msa-vae-internal-alignment-v1` manifests.
- Produces the requested single-seed internal alignment-realism table.

- [ ] **Step 1: Verify checkpoint and GPU readiness**

```bash
conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py verify
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader,nounits
screen -ls
```

Expected: all four Phase-2 `net_last.pth` files pass lineage/SHA validation,
and GPUs 0, 2, 4, and 6 have no conflicting compute processes.

- [ ] **Step 2: Dry-run the durable launch**

```bash
PILOT_DRY_RUN=1 \
bash explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh
```

Expected: exactly four Screen commands, correct checkpoint paths, and GPU
mapping 0/2/4/6; no session is created.

- [ ] **Step 3: Launch four background evaluations**

```bash
bash explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh
```

Expected: four detached Screen sessions:

```text
msa_internal_eval_no_align_s123
msa_internal_eval_global_only_s123
msa_internal_eval_local_only_s123
msa_internal_eval_global_local_s123
```

- [ ] **Step 4: Monitor without attaching foreground jobs**

```bash
bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
screen -ls
```

Repeat the non-mutating status checks until all four internal status files say
`state=internal_evaluation_complete`. If one fails, inspect only the matching
`logs/{slug}.internal_evaluation.screen.log`, fix the proven cause, and
relaunch only that slug after moving its incomplete output directory aside to
a timestamped recovery path.

- [ ] **Step 5: Validate and collect the main table**

```bash
conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect-internal
```

Expected artifacts:

```text
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/internal_alignment_pilot_table.json
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/internal_alignment_pilot_table.csv
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/internal_alignment_pilot_table.md
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/internal_alignment_deltas.json
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/global_cosine_vs_fid.svg
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/global_cosine_vs_mpjpe.svg
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/in_sample_local_cosine_vs_fid.svg
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/in_sample_local_cosine_vs_mpjpe.svg
```

- [ ] **Step 6: Run scientific sanity checks**

Inspect all four manifests and confirm:

- No Alignment and Local Only do not gain global MSA-T5 retrieval merely from
  reconstructed-motion quality;
- Global Only and Global + Local improve `global_cosine` and internal retrieval
  relative to No Alignment;
- Local Only and Global + Local improve `in_sample_local_cosine`;
- shuffled global retrieval is close to corpus chance and far below aligned
  unshuffled retrieval;
- FID/MPJPE values come from `posterior_mean` decode;
- sample, target, checkpoint, evaluator, and model identities match across
  variants;
- the table labels local cosine as an in-sample diagnostic.

Do not force an expected realism ordering. If posterior-mean realism remains
nearly unchanged, report that the current semantic heads absorb most alignment
pressure rather than changing the evaluator or hiding the result.

- [ ] **Step 7: Report the results**

Return:

- the complete Markdown table;
- absolute deltas from No Alignment;
- a short global-alignment interpretation;
- a short local-alignment interpretation with the in-sample limitation;
- a short realism interpretation;
- the exact checkpoint, split, decode, seed, GPU, and output identities;
- the held-out local-data blocker and the accepted route for final paper
  results.

Do not report uncertainty or statistical significance for this one-seed pilot.
