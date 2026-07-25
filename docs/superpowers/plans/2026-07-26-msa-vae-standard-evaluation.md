# MSA-VAE Standard Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated one-command HumanML3D-272 MSA-VAE evaluator that reports FID, MPJPE, P-MPJPE, ACCEL, foot skating, and full-set bidirectional retrieval for any compatible checkpoint path.

**Architecture:** Add a deterministic complete-motion dataset, a pure metric module, a checkpoint/configuration resolver, and a standalone evaluation entrypoint with its own uppercase shell launcher. The new entrypoint reuses the frozen 272-D evaluator and MSA-VAE class but neither imports into nor modifies any legacy evaluation path.

**Tech Stack:** Python 3.8.11, PyTorch 2.4.1+cu118, NumPy, SciPy, `unittest`, Bash, the existing `mgpt` conda environment, and the repository's frozen HumanML3D-272 TEMOS/TMR evaluator.

## Global Constraints

- Do not modify `eval_msa_vae.py`, `EVAL_msa_vae.sh`, `utils/eval_trans.py`, training-time evaluation calls, official MSA-T2M launchers, or TAE launchers.
- Do not edit, stage, vendor, or commit `explorations/motion-latent-diffusion/`, `explorations/OmniControl/`, or `explorations/TMR/`.
- Do not add or upgrade dependencies; use the existing `mgpt` environment.
- Keep the 272-D representation, 22-joint recovery, frozen evaluator checkpoint, and strict checkpoint loading.
- Retrieval is full-test-set cosine `TMR-full-normal`, exact diagonal, average tie ranks, percentage R@1/R@2/R@3, and one-based MedR in both directions.
- MPJPE and P-MPJPE are millimetres; ACCEL is millimetres per frame squared without FPS-squared scaling.
- Skating uses feet 10/11, Y-up, XZ velocity, 0.05 m contact height, 0.50 m/s speed, 30 FPS, and an eight-frame smoothing window.
- Every physical metric must exclude padded frames; the deterministic loader does not shuffle, crop randomly, repeat, or drop the last batch.
- Real full-test-set CUDA evaluation is not a code-edit validation step.

---

### Task 1: Deterministic Complete-Motion Evaluation Dataset

**Files:**
- Create: `humanml3d_272/dataset_eval_msa_vae_metrics.py`
- Create: `tests/test_msa_vae_metrics_dataset.py`

**Interfaces:**
- Consumes: HumanML3D-272 `motion_data/`, `texts/`, `mean_std/Mean.npy`, `mean_std/Std.npy`, and `split/test.txt`.
- Produces: `MSAVAEMetricsDataset`, `collate_msa_vae_metrics(batch)`, `make_msa_vae_metrics_loader(...)`, `dataset.sample_ids`, `dataset.sample_hash`, and `dataset.inv_transform(array)`.
- Batch contract: `{"sample_ids": List[str], "captions": List[str], "motions": FloatTensor[B,T,272], "lengths": LongTensor[B]}`.

- [ ] **Step 1: Write failing deterministic-dataset tests**

```python
class MSAVAEMetricsDatasetTest(unittest.TestCase):
    def test_uses_split_order_first_full_caption_and_complete_motion(self):
        dataset = self._dataset(
            split_ids=["motion_b", "motion_a"],
            motions={"motion_b": np.ones((64, 272)), "motion_a": np.ones((63, 272))},
            texts={
                "motion_b": [
                    "segment#tok#0.5#1.5",
                    "first full#tok#0#0",
                    "second full#tok#0#0",
                ],
                "motion_a": ["a full#tok#0#0"],
            },
        )
        self.assertEqual(dataset.sample_ids, ["motion_b", "motion_a"])
        self.assertEqual(dataset[0]["caption"], "first full")
        self.assertEqual(dataset[0]["length"], 64)
        self.assertEqual(dataset[1]["length"], 60)

    def test_filters_short_long_missing_full_caption_and_tagged_subclips(self):
        dataset = self._dataset(
            split_ids=["short", "long", "segment_only", "valid"],
            motions={
                "short": np.zeros((59, 272)),
                "long": np.zeros((300, 272)),
                "segment_only": np.zeros((80, 272)),
                "valid": np.zeros((80, 272)),
            },
            texts={
                "short": ["short#tok#0#0"],
                "long": ["long#tok#0#0"],
                "segment_only": ["clip#tok#0.2#1.0"],
                "valid": ["valid#tok#0#0"],
            },
        )
        self.assertEqual(dataset.sample_ids, ["valid"])

    def test_collate_pads_with_zero_and_preserves_lengths_and_ids(self):
        batch = collate_msa_vae_metrics([self._item("a", 60), self._item("b", 64)])
        self.assertEqual(tuple(batch["motions"].shape), (2, 64, 272))
        self.assertEqual(batch["lengths"].tolist(), [60, 64])
        self.assertTrue(torch.equal(batch["motions"][0, 60:], torch.zeros(4, 272)))
        self.assertEqual(batch["sample_ids"], ["a", "b"])
```

- [ ] **Step 2: Run the new tests and verify the module is missing**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_metrics_dataset -v
```

Expected: FAIL with `ModuleNotFoundError: humanml3d_272.dataset_eval_msa_vae_metrics`.

- [ ] **Step 3: Implement immutable records, deterministic parsing, normalization, and collation**

```python
@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    caption: str
    motion_path: Path
    length: int


class MSAVAEMetricsDataset(Dataset):
    def __init__(self, data_root, split_file=None, unit_length=4,
                 min_motion_length=60, max_motion_length=300):
        self.data_root = Path(data_root).resolve()
        self.mean = np.load(self.data_root / "mean_std" / "Mean.npy")
        self.std = np.load(self.data_root / "mean_std" / "Std.npy")
        self.records = self._build_records(
            Path(split_file) if split_file else self.data_root / "split" / "test.txt"
        )
        self.sample_ids = [record.sample_id for record in self.records]
        if not self.records:
            raise ValueError("deterministic MSA-VAE evaluation set is empty")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("deterministic MSA-VAE evaluation IDs are not unique")
        joined = "\n".join(self.sample_ids).encode("utf-8")
        self.sample_hash = hashlib.sha256(joined).hexdigest()

    def __getitem__(self, index):
        record = self.records[index]
        motion = np.load(record.motion_path)[:record.length].astype(np.float32)
        normalized = (motion - self.mean) / self.std
        return {
            "sample_id": record.sample_id,
            "caption": record.caption,
            "motion": torch.from_numpy(normalized.astype(np.float32)),
            "length": record.length,
        }
```

Parsing must treat NaN tags as zero, choose only the first `from=0,to=0`
caption, quantize `length = raw_length // unit_length * unit_length`, and never
materialize tagged subclips. `make_msa_vae_metrics_loader` must set
`shuffle=False`, `drop_last=False`, and use the dedicated collator.

- [ ] **Step 4: Run dataset tests**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_metrics_dataset -v
```

Expected: all dataset tests PASS.

- [ ] **Step 5: Commit the dataset unit**

```bash
git add humanml3d_272/dataset_eval_msa_vae_metrics.py tests/test_msa_vae_metrics_dataset.py
git commit -m "feat: add deterministic MSA-VAE metric dataset"
```

---

### Task 2: Standard Reconstruction and Retrieval Metrics

**Files:**
- Create: `utils/msa_vae_metrics.py`
- Create: `tests/test_msa_vae_metrics.py`

**Interfaces:**
- Consumes: valid unpadded `Tensor[T,22,3]` joint sequences and evaluator embeddings `Tensor[N,D]` or `ndarray[N,D]`.
- Produces: `SkatingConfig`, `ReconstructionMetricAccumulator.update(pred, target)`, `ReconstructionMetricAccumulator.compute()`, `calculate_fid(reference, prediction)`, `retrieval_metrics_from_similarity(similarity)`, and `calculate_bidirectional_retrieval(text_embeddings, motion_embeddings)`.
- Metric dictionaries use keys `fid`, `mpjpe_mm`, `p_mpjpe_mm`, `accel_mm_per_frame2`, `skating_percent`, `t2m_r1_percent`, `t2m_r2_percent`, `t2m_r3_percent`, `t2m_medr`, `m2t_r1_percent`, `m2t_r2_percent`, `m2t_r3_percent`, and `m2t_medr`.

- [ ] **Step 1: Write failing MLD-equivalence and aggregation tests**

```python
class ReconstructionMetricsTest(unittest.TestCase):
    def test_mpjpe_root_alignment_removes_global_translation(self):
        target = self.non_degenerate_joints(frames=4)
        prediction = target + torch.tensor([3.0, 0.0, -2.0])
        acc = ReconstructionMetricAccumulator()
        acc.update(prediction, target)
        self.assertAlmostEqual(acc.compute()["mpjpe_mm"], 0.0, places=4)

    def test_p_mpjpe_removes_per_frame_similarity_transform(self):
        target = self.non_degenerate_joints(frames=4)
        prediction = 2.5 * target + torch.tensor([1.0, -3.0, 2.0])
        acc = ReconstructionMetricAccumulator()
        acc.update(prediction, target)
        self.assertAlmostEqual(acc.compute()["p_mpjpe_mm"], 0.0, places=3)

    def test_accel_uses_mld_frame_squared_units(self):
        target = torch.zeros(5, 22, 3)
        prediction = target.clone()
        prediction[:, :, 0] = torch.arange(5, dtype=torch.float32).square()[:, None]
        acc = ReconstructionMetricAccumulator()
        acc.update(prediction, target)
        self.assertAlmostEqual(acc.compute()["accel_mm_per_frame2"], 2000.0, places=3)
```

The test module must load
`Evaluator_272/mld/models/metrics/utils.py` with
`importlib.util.spec_from_file_location` and compare local per-frame MPJPE,
P-MPJPE, and ACCEL arrays against its `calc_mpjpe`, `calc_pampjpe`, and
`calc_accel` outputs on seeded non-degenerate tensors.

- [ ] **Step 2: Write failing skating, FID, and retrieval tests**

```python
def test_skating_is_sample_mean_and_excludes_padding(self):
    sliding = stationary_feet(frames=12)
    sliding[:, [10, 11], 0] = torch.arange(12)[:, None] * 0.1
    still = stationary_feet(frames=20)
    acc = ReconstructionMetricAccumulator()
    acc.update(sliding, sliding.clone().zero_())
    acc.update(still, still.clone())
    self.assertAlmostEqual(acc.compute()["skating_percent"], 50.0, places=4)

def test_bidirectional_retrieval_uses_full_matrix_and_transpose(self):
    similarity = np.array([[1.0, 0.0, 0.0],
                           [0.9, 0.8, 0.0],
                           [0.0, 0.7, 1.0]])
    metrics = retrieval_metrics_from_similarity(similarity)
    self.assertAlmostEqual(metrics["t2m_r1_percent"], 200.0 / 3.0)
    self.assertEqual(metrics["t2m_r2_percent"], 100.0)
    self.assertEqual(metrics["t2m_medr"], 1.0)
    self.assertEqual(metrics["m2t_r1_percent"], 100.0)
    self.assertEqual(metrics["m2t_medr"], 1.0)

def test_fid_of_identical_embeddings_is_zero(self):
    embeddings = np.arange(24, dtype=np.float64).reshape(6, 4)
    self.assertAlmostEqual(calculate_fid(embeddings, embeddings), 0.0, places=8)
```

Include a two-query all-ties retrieval case and assert average tie rank gives
`MedR == 1.5`.

- [ ] **Step 3: Run metric tests and verify missing symbols**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_metrics -v
```

Expected: FAIL because `utils.msa_vae_metrics` does not exist.

- [ ] **Step 4: Implement the minimal metric module**

```python
@dataclass(frozen=True)
class SkatingConfig:
    foot_indices: Tuple[int, int] = (10, 11)
    fps: float = 30.0
    height_threshold_m: float = 0.05
    velocity_threshold_mps: float = 0.50
    smoothing_window_frames: int = 8


def retrieval_metrics_from_similarity(similarity):
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("TMR-full-normal requires a square similarity matrix")
    t2m_ranks = _diagonal_ranks_with_average_ties(similarity)
    m2t_ranks = _diagonal_ranks_with_average_ties(similarity.T)
    metrics = _rank_metrics(t2m_ranks, "t2m")
    metrics.update(_rank_metrics(m2t_ranks, "m2t"))
    return metrics


def calculate_bidirectional_retrieval(text_embeddings, motion_embeddings):
    text = F.normalize(torch.as_tensor(text_embeddings).float(), dim=-1)
    motion = F.normalize(torch.as_tensor(motion_embeddings).float(), dim=-1)
    return retrieval_metrics_from_similarity((text @ motion.T).cpu().numpy())
```

Implement Procrustes with the same centring, SVD, reflection correction,
scale, and translation equations as the tracked MLD reference. The
accumulator must store MPJPE and P-MPJPE sums with `frame_count`, ACCEL sum
with `accel_frame_count`, and one skating ratio per sequence. Skating must use
`scipy.ndimage.uniform_filter1d(..., size=8, mode="constant", origin=0)`.
All public results pass an `np.isfinite` check before return.

- [ ] **Step 5: Run metric tests**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_metrics -v
```

Expected: all metric tests PASS, including exact agreement with the tracked MLD functions.

- [ ] **Step 6: Commit the metric unit**

```bash
git add utils/msa_vae_metrics.py tests/test_msa_vae_metrics.py
git commit -m "feat: add standard MSA-VAE reconstruction metrics"
```

---

### Task 3: Legacy-Safe Checkpoint Configuration Resolution

**Files:**
- Create: `utils/msa_vae_eval_config.py`
- Create: `tests/test_msa_vae_eval_config.py`

**Interfaces:**
- Consumes: a checkpoint path, deserialized checkpoint payload, and a dictionary of explicit non-`None` overrides.
- Produces: `ResolvedMSAVAEConfig(values: dict, sources: dict)`, `load_checkpoint_payload(path)`, `resolve_msa_vae_config(path, payload, overrides)`, and `build_and_load_msa_vae(path, overrides, device)`.
- `build_and_load_msa_vae` returns `(model, resolved_config, checkpoint_manifest)` and performs `strict=True` state loading.

- [ ] **Step 1: Write failing precedence and inference tests**

```python
class MSAEvalConfigTest(unittest.TestCase):
    def test_resolution_precedence_is_override_metadata_log_inference_default(self):
        payload = {
            "net": self.synthetic_state_dict(enc_layers=5, dec_layers=4, ff_size=1536),
            "metadata": {"training_args": {"trans_nhead": 4, "depth": 2}},
        }
        self.write_run_log({"trans_nhead": 2, "hidden_size": 512})
        resolved = resolve_msa_vae_config(
            self.checkpoint, payload, {"trans_nhead": 8}
        )
        self.assertEqual(resolved.values["trans_nhead"], 8)
        self.assertEqual(resolved.sources["trans_nhead"], "cli")
        self.assertEqual(resolved.values["depth"], 2)
        self.assertEqual(resolved.values["hidden_size"], 512)
        self.assertEqual(resolved.values["trans_enc_layers"], 5)
        self.assertEqual(resolved.values["trans_dec_layers"], 4)
        self.assertEqual(resolved.values["trans_ff_size"], 1536)

    def test_legacy_mainline_defaults_cover_non_inferable_fields(self):
        resolved = resolve_msa_vae_config(
            self.checkpoint, {"net": self.synthetic_state_dict()}, {}
        )
        self.assertEqual(resolved.values["down_t"], 2)
        self.assertEqual(resolved.values["stride_t"], 2)
        self.assertEqual(resolved.values["trans_nhead"], 8)
        self.assertEqual(resolved.values["dilation_growth_rate"], 3)
```

Also test that `run.log` scanning skips non-JSON prefixes and takes the first
complete JSON object, while malformed metadata values produce a field-specific
`ValueError`.

- [ ] **Step 2: Write failing strict-load and checkpoint-manifest tests**

```python
def test_build_rejects_partial_or_structurally_wrong_checkpoint(self):
    payload = {"net": self.synthetic_state_dict()}
    del payload["net"]["msa_vae.trans_encoder.input_proj.weight"]
    torch.save(payload, self.checkpoint)
    with self.assertRaisesRegex(RuntimeError, "Missing key"):
        build_and_load_msa_vae(self.checkpoint, {}, torch.device("cpu"))

def test_checkpoint_manifest_records_resolved_path_size_mtime_and_sha256(self):
    _, _, manifest = build_and_load_msa_vae(
        self.valid_checkpoint, {}, torch.device("cpu")
    )
    self.assertEqual(manifest["path"], str(self.valid_checkpoint.resolve()))
    self.assertEqual(len(manifest["sha256"]), 64)
```

For the valid-build test, patch `models.msa_vae.MSA_HumanVAE` with a tiny fake
module whose `load_state_dict` records `strict=True`; do not instantiate a
full 470 MB model in a unit test.

- [ ] **Step 3: Run config tests and verify the module is missing**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_eval_config -v
```

Expected: FAIL with `ModuleNotFoundError: utils.msa_vae_eval_config`.

- [ ] **Step 4: Implement resolver, shape inference, hashing, and strict model construction**

```python
MAINLINE_DEFAULTS = {
    "hidden_size": 1024,
    "down_t": 2,
    "stride_t": 2,
    "depth": 3,
    "dilation_growth_rate": 3,
    "latent_dim": 16,
    "trans_d_model": 768,
    "trans_nhead": 8,
    "trans_enc_layers": 6,
    "trans_dec_layers": 6,
    "trans_ff_size": 2048,
    "trans_dropout": 0.1,
    "clip_dim": 768,
    "disable_decoupling": False,
}


@dataclass(frozen=True)
class ResolvedMSAVAEConfig:
    values: Dict[str, Any]
    sources: Dict[str, str]
```

Infer `hidden_size`, `latent_dim`, `trans_d_model`, encoder/decoder layer
counts, feed-forward size, and `clip_dim` from stable state-dict keys and
shapes. Use `json.JSONDecoder.raw_decode` to scan `run.log`; do not use a
greedy regular expression. Normalize legacy `text_embed_dim` to `clip_dim`.
Validate positive dimensions, `trans_d_model % trans_nhead == 0`, and boolean
`disable_decoupling`. Build `MSA_HumanVAE` with the resolved values and call
`model.load_state_dict(state, strict=True)` before moving it to the device.

- [ ] **Step 5: Run config tests**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_eval_config -v
```

Expected: all config tests PASS.

- [ ] **Step 6: Commit the checkpoint/config unit**

```bash
git add utils/msa_vae_eval_config.py tests/test_msa_vae_eval_config.py
git commit -m "feat: resolve MSA-VAE evaluation checkpoints safely"
```

---

### Task 4: Single-Pass Evaluation Orchestration and Result Artifacts

**Files:**
- Create: `eval_msa_vae_metrics.py`
- Create: `tests/test_eval_msa_vae_metrics.py`

**Interfaces:**
- Consumes: `MSAVAEMetricsDataset`, resolved MSA-VAE, frozen evaluator pair `[text_encoder, motion_encoder]`, a device, and `SkatingConfig`.
- Produces: `load_frozen_humanml_evaluator(repo_root, device, checkpoint_path=None)`, `evaluate_msa_vae_metrics(model, evaluator, loader, device, skating_config)`, `build_result_manifest(...)`, `write_result_artifacts(result, output_dir)`, `parse_args(argv=None)`, and `main(argv=None)`.
- `evaluate_msa_vae_metrics` returns one flat unrounded metric dictionary plus `sample_count`.

- [ ] **Step 1: Write failing fake-model orchestration tests**

```python
class EvalMSAVAEMetricsTest(unittest.TestCase):
    def test_each_batch_is_reconstructed_once_and_retrieval_uses_prediction(self):
        model = FakeMSAVAE(offset=0.25)
        evaluator = [FakeTextEncoder(), FakeMotionEncoder()]
        result = evaluate_msa_vae_metrics(
            model, evaluator, self.loader(two_batches=True),
            torch.device("cpu"), SkatingConfig()
        )
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(result["sample_count"], 3)
        self.assertIn("fid", result)
        self.assertIn("t2m_r1_percent", result)
        self.assertIn("m2t_medr", result)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_valid_lengths_are_sliced_before_joint_metrics(self):
        model = FakeMSAVAE(padded_value=1e6)
        result = evaluate_msa_vae_metrics(
            model, self.fake_evaluator, self.loader_with_padding(),
            torch.device("cpu"), SkatingConfig()
        )
        self.assertAlmostEqual(result["mpjpe_mm"], 0.0, places=5)
```

Patch `recover_from_local_position` in this unit test to map the first 66
features directly to `[T,22,3]`. Fake encoders return objects with a `.loc`
tensor, matching the real evaluator contract.

- [ ] **Step 2: Write failing artifact and preflight tests**

```python
def test_artifacts_include_protocol_identity_units_and_flat_csv(self):
    result = build_result_manifest(
        metrics=self.metric_fixture(),
        checkpoint=self.checkpoint_fixture(),
        evaluator=self.evaluator_fixture(),
        model_config=self.config_fixture(),
        dataset=self.dataset_fixture(),
        seed=123,
        skating_config=SkatingConfig(),
    )
    write_result_artifacts(result, self.output_dir)
    loaded = json.loads((self.output_dir / "metrics.json").read_text())
    self.assertEqual(loaded["protocol"]["retrieval"], "TMR-full-normal")
    self.assertEqual(loaded["units"]["mpjpe_mm"], "mm")
    self.assertEqual(loaded["dataset"]["sample_hash"], "abc123")
    rows = list(csv.DictReader((self.output_dir / "metrics.csv").open()))
    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0]["checkpoint_sha256"], "f" * 64)

def test_preflight_names_every_missing_artifact(self):
    with self.assertRaisesRegex(FileNotFoundError, "Mean.npy"):
        preflight_evaluation_assets(self.args_with_missing_mean())
```

Also test non-finite embeddings and mismatched embedding/sample counts fail
before artifact writing.

- [ ] **Step 3: Run orchestration tests and verify the entrypoint is missing**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_vae_metrics -v
```

Expected: FAIL with `ModuleNotFoundError: eval_msa_vae_metrics`.

- [ ] **Step 4: Implement evaluator loading and the single-pass loop**

```python
@torch.inference_mode()
def evaluate_msa_vae_metrics(model, evaluator, loader, device, skating_config):
    text_encoder, motion_encoder = evaluator
    reconstruction = ReconstructionMetricAccumulator(skating_config)
    text_embeddings, gt_embeddings, pred_embeddings = [], [], []
    sample_count = 0

    for batch in loader:
        motions = batch["motions"].to(device)
        lengths = batch["lengths"].to(device)
        outputs = model(motions, lengths=lengths)
        predictions = outputs["x_recon"]
        text_embeddings.append(text_encoder(batch["captions"]).loc.cpu())
        gt_embeddings.append(motion_encoder(motions, lengths).loc.cpu())
        pred_embeddings.append(motion_encoder(predictions, lengths).loc.cpu())
        for index, length in enumerate(lengths.tolist()):
            gt_xyz = recover_valid_joints(loader.dataset, motions[index], length)
            pred_xyz = recover_valid_joints(loader.dataset, predictions[index], length)
            reconstruction.update(pred_xyz, gt_xyz)
        sample_count += len(batch["sample_ids"])

    metrics = reconstruction.compute()
    metrics["fid"] = calculate_fid(torch.cat(gt_embeddings), torch.cat(pred_embeddings))
    metrics.update(calculate_bidirectional_retrieval(
        torch.cat(text_embeddings), torch.cat(pred_embeddings)
    ))
    return {"sample_count": sample_count, **metrics}
```

Before evaluating, set Python, NumPy, CPU, and CUDA seeds to 123 and put all
models in `eval()` mode. Load the frozen evaluator exactly as the legacy
entrypoint does, but resolve all paths from `Path(__file__).resolve().parent`,
prepend the evaluator root to `sys.path` only for imports, strictly load each
encoder prefix, freeze parameters, and never call `os.chdir`.

- [ ] **Step 5: Implement manifest, log/table, JSON, CSV, and CLI plumbing**

`parse_args` must define the required positional `checkpoint`, repository-
relative defaults for data/evaluator/output, `--device` defaulting to `cuda`,
`--batch-size`, `--num-workers`, `--seed`, and optional architecture overrides
whose defaults are `None`. `main` must:

```python
repo_root = Path(__file__).resolve().parent
args = parse_args(argv)
paths = resolve_cli_paths(repo_root, args)
preflight_evaluation_assets(paths)
payload = load_checkpoint_payload(paths.checkpoint)
model, resolved, checkpoint_manifest = build_and_load_msa_vae(
    paths.checkpoint, architecture_overrides(args), device
)
dataset = MSAVAEMetricsDataset(paths.data_root, paths.split_file,
                               unit_length=resolved.values["stride_t"]
                               ** resolved.values["down_t"])
loader = make_msa_vae_metrics_loader(dataset, args.batch_size, args.num_workers)
evaluator = load_frozen_humanml_evaluator(repo_root, device,
                                          paths.evaluator_checkpoint)
metrics = evaluate_msa_vae_metrics(model, evaluator, loader, device,
                                   SkatingConfig())
manifest = build_result_manifest(...)
write_result_artifacts(manifest, paths.output_dir)
print_final_table(manifest)
```

The default output path is
`output/msa_vae_metrics/<checkpoint-parent-name>/<checkpoint-stem>/`.
The JSON protocol version is `msa-vae-standard-v1`; the CSV contains all 13
requested metrics and flattened checkpoint/evaluator/sample identities.

- [ ] **Step 6: Run orchestration tests**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_eval_msa_vae_metrics -v
```

Expected: all orchestration, preflight, and artifact tests PASS.

- [ ] **Step 7: Commit the orchestration unit**

```bash
git add eval_msa_vae_metrics.py tests/test_eval_msa_vae_metrics.py
git commit -m "feat: add single-pass MSA-VAE metric evaluation"
```

---

### Task 5: One-Command Launcher and Legacy Isolation

**Files:**
- Create: `EVAL_msa_vae_metrics.sh`
- Create: `tests/test_msa_vae_metrics_launcher.py`

**Interfaces:**
- Consumes: first positional checkpoint path and arbitrary remaining Python CLI arguments.
- Produces: `bash EVAL_msa_vae_metrics.sh <checkpoint> [options]`, with repository-root execution and `mgpt` environment selection.

- [ ] **Step 1: Write failing launcher and isolation tests**

```python
class MSAVAEMetricsLauncherTest(unittest.TestCase):
    def test_launcher_forwards_checkpoint_and_remaining_arguments(self):
        completed, arguments = self.run_with_fake_conda(
            ["relative/model.pth", "--batch-size", "7", "--device", "cpu"]
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(arguments[:4], ["run", "-n", "mgpt", "python"])
        self.assertIn("eval_msa_vae_metrics.py", arguments[4])
        self.assertEqual(arguments[5:], [
            "relative/model.pth", "--batch-size", "7", "--device", "cpu"
        ])

    def test_launcher_requires_exactly_a_checkpoint_before_options(self):
        completed, _ = self.run_with_fake_conda([])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stderr)

    def test_legacy_entrypoints_do_not_reference_new_pipeline(self):
        for relative in ("eval_msa_vae.py", "EVAL_msa_vae.sh", "utils/eval_trans.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("msa_vae_metrics", source)
```

Add a second launcher test with `CONDA_DEFAULT_ENV=mgpt` and a fake `python`
binary; it must call `python` directly instead of nesting `conda run`.

- [ ] **Step 2: Run launcher tests and verify the script is missing**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_metrics_launcher -v
```

Expected: FAIL because `EVAL_msa_vae_metrics.sh` does not exist.

- [ ] **Step 3: Implement the repository-relative shell launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "Usage: bash EVAL_msa_vae_metrics.sh <checkpoint.pth> [evaluation options]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" == "mgpt" ]]; then
  exec python "$SCRIPT_DIR/eval_msa_vae_metrics.py" "$@"
fi
exec conda run -n mgpt python "$SCRIPT_DIR/eval_msa_vae_metrics.py" "$@"
```

The script must not create symlinks, set `CUDA_VISIBLE_DEVICES`, contain
absolute cluster paths, or mutate the caller's working directory.

- [ ] **Step 4: Run launcher tests and shell syntax validation**

Run:

```bash
conda run -n mgpt python -m unittest tests.test_msa_vae_metrics_launcher -v
bash -n EVAL_msa_vae_metrics.sh
```

Expected: all launcher tests PASS and `bash -n` exits 0.

- [ ] **Step 5: Commit the user-facing launcher**

```bash
git add EVAL_msa_vae_metrics.sh tests/test_msa_vae_metrics_launcher.py
git commit -m "feat: add one-command MSA-VAE metric launcher"
```

---

### Task 6: Full Regression and Static Verification

**Files:**
- Verify: `humanml3d_272/dataset_eval_msa_vae_metrics.py`
- Verify: `utils/msa_vae_metrics.py`
- Verify: `utils/msa_vae_eval_config.py`
- Verify: `eval_msa_vae_metrics.py`
- Verify: `EVAL_msa_vae_metrics.sh`
- Verify: all `tests/test_msa_vae_metrics*.py` and `tests/test_eval_msa_vae_metrics.py`

**Interfaces:**
- Consumes: all prior task deliverables.
- Produces: evidence that the new pipeline is syntactically valid, unit-tested, isolated, and regression-safe without launching an expensive CUDA benchmark.

- [ ] **Step 1: Run all new focused tests together**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_msa_vae_metrics_dataset \
  tests.test_msa_vae_metrics \
  tests.test_msa_vae_eval_config \
  tests.test_eval_msa_vae_metrics \
  tests.test_msa_vae_metrics_launcher -v
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run the complete existing test suite**

Run:

```bash
conda run -n mgpt python -m unittest discover -s tests -v
```

Expected: the complete suite PASS with no legacy evaluator regression.

- [ ] **Step 3: Run repository-required static checks**

Run:

```bash
conda run -n mgpt python -m py_compile \
  humanml3d_272/dataset_eval_msa_vae_metrics.py \
  utils/msa_vae_metrics.py \
  utils/msa_vae_eval_config.py \
  eval_msa_vae_metrics.py
bash -n EVAL_msa_vae_metrics.sh
git diff --check
```

Expected: every command exits 0 with no output from `git diff --check`.

- [ ] **Step 4: Verify help and checkpoint preflight without CUDA evaluation**

Run:

```bash
conda run -n mgpt python eval_msa_vae_metrics.py --help
conda run -n mgpt python eval_msa_vae_metrics.py \
  /definitely/missing/checkpoint.pth --device cpu
```

Expected: `--help` exits 0 and documents all requested metrics/protocol; the
missing checkpoint invocation exits non-zero and names the exact missing path
before importing or allocating the evaluator.

- [ ] **Step 5: Audit scope and working tree**

Run:

```bash
git status --short
git diff --name-only HEAD~5..HEAD
git log --oneline -7
```

Expected: only the new MSA-VAE-specific source, tests, launcher, design, and
plan are tracked changes; the three locally cloned exploration directories
remain untracked and untouched.

- [ ] **Step 6: Record any final verification-only adjustment**

If static or regression verification required a source change, rerun the
specific failing command and the full focused suite, then commit only the
relevant tracked files:

```bash
git add EVAL_msa_vae_metrics.sh eval_msa_vae_metrics.py \
  humanml3d_272/dataset_eval_msa_vae_metrics.py \
  utils/msa_vae_metrics.py utils/msa_vae_eval_config.py \
  tests/test_msa_vae_metrics_dataset.py tests/test_msa_vae_metrics.py \
  tests/test_msa_vae_eval_config.py tests/test_eval_msa_vae_metrics.py \
  tests/test_msa_vae_metrics_launcher.py
git commit -m "test: verify isolated MSA-VAE evaluation pipeline"
```

If no adjustment was needed, do not create an empty commit.
