# Exploration Experiment Results Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move exactly 35 exploration-result directories into a route-based `Experiments/explorations/` tree while leaving all Causal TAE, MSA-VAE, Global-RAG DDPM, and formal ablation results at the experiment root.

**Architecture:** Add a small, tested archival utility whose manifest is the single source of truth for preflight, same-filesystem moves, verification, and rollback. Update only runnable exploration entrypoints whose existing checkpoints or future output belong to the moved routes, then execute the move and save a tracked post-move manifest.

**Tech Stack:** Python 3.8 standard library (`argparse`, `dataclasses`, `pathlib`, `os`), `unittest`, Bash, Git.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-25-experiment-results-archive-design.md`.
- Move exactly 35 directories totaling approximately 1.1 TB.
- Never delete, overwrite, copy, compress, deduplicate, or rewrite checkpoint content.
- Do not create compatibility symlinks at old result paths.
- Keep `Causal_TAE`, every `causal_TAE*`, every `MSA_VAE*`, ordinary Global-RAG DDPM, and formal ablations directly under `Experiments/`.
- A Global-RAG directory moves only when its name explicitly identifies Rectified Flow, MCA, latent retrieval, or local-RAG cross-attention.
- All moves must be same-filesystem directory renames with explicit source and destination paths.
- Stop before mutation if a source is missing, a destination exists, a mapping is duplicated, or filesystem device IDs differ.
- Preserve the unrelated dirty submodule `paper writing/Research-Paper-Writing-Skills`.
- Do not run training, evaluation, cache construction, checkpoint loading, or dataset mutation.

---

### Task 1: Add a Tested Archive and Rollback Utility

**Files:**

- Create: `scripts/archive_exploration_results.py`
- Create: `tests/test_archive_exploration_results.py`

**Interfaces:**

- Produces: immutable `ArchiveEntry(route: str, name: str)` values in `ARCHIVE_ENTRIES`.
- Produces: `snapshot_tree(path: Path) -> TreeSnapshot`.
- Produces: `preflight(experiments_root: Path, entries: Sequence[ArchiveEntry], rollback: bool = False) -> list[MoveRecord]`.
- Produces: `apply_moves(records: Sequence[MoveRecord]) -> None`.
- Produces: `verify_moves(records: Sequence[MoveRecord]) -> None`.
- Produces: `render_manifest(records: Sequence[MoveRecord], operation: str) -> str`.
- Produces: `load_manifest(path: Path, experiments_root: Path) -> list[MoveRecord]`.
- CLI modes: default dry-run, `--apply`, `--verify`, and `--rollback`.

- [ ] **Step 1: Write tests for exact classification and duplicate rejection**

Create `tests/test_archive_exploration_results.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.archive_exploration_results import (
    ARCHIVE_ENTRIES,
    ArchiveEntry,
    preflight,
)


class ArchiveExplorationResultsTest(unittest.TestCase):
    def test_manifest_contains_exactly_35_unique_sources(self):
        names = [entry.name for entry in ARCHIVE_ENTRIES]
        self.assertEqual(len(names), 35)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            {entry.route for entry in ARCHIVE_ENTRIES},
            {
                "clip",
                "rectified_flow",
                "cross_attention/mca",
                "cross_attention/latent_retrieval",
                "cross_attention/local_rag",
                "qformer",
                "representation_experiments",
                "motionstreamer_baselines",
                "misc",
            },
        )

    def test_preflight_rejects_duplicate_source_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            root.mkdir()
            (root / "run").mkdir()
            entries = [
                ArchiveEntry("clip", "run"),
                ArchiveEntry("qformer", "run"),
            ]
            with self.assertRaisesRegex(ValueError, "duplicate source"):
                preflight(root, entries)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_archive_exploration_results -v
```

Expected: import failure because `scripts/archive_exploration_results.py` does
not exist.

- [ ] **Step 3: Implement the immutable 35-entry manifest**

Create `scripts/archive_exploration_results.py` with the following manifest:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveEntry:
    route: str
    name: str


def entry(route, name):
    return ArchiveEntry(route=route, name=name)


ARCHIVE_ENTRIES = (
    entry("clip", "MotionStreamer_t2m_272_baseline_clip"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_100000Iter_addEMA"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_200000Iter"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_tuned"),
    entry("rectified_flow", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_ema_tuned_addLR"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose_fix"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5_Flamingo"),
    entry("cross_attention/mca", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_start_from_ckpt_LR5_Flamingo_gateclose"),
    entry("cross_attention/latent_retrieval", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_6layer_top3_ddpm"),
    entry("cross_attention/latent_retrieval", "MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_late_after_sa_every1layer_top3_ddpm_cfg_saca_dropout01"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L16_k3_sa_ca"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L4_k3"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L4_k3_crossattn"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L8_k3"),
    entry("cross_attention/local_rag", "MotionStreamer_t2m_272_msa_rag_local_L8_k3_crossattn"),
    entry("qformer", "QFormer_t2m_272_v1"),
    entry("qformer", "QFormer_t2m_272_v2"),
    entry("qformer", "QFormer_t2m_272_v3"),
    entry("qformer", "QFormer_t2m_272_v4"),
    entry("qformer", "QFormer_t2m_272_v5"),
    entry("representation_experiments", "SAE_v1_t2m_272"),
    entry("representation_experiments", "TAE_GAN_Loss_"),
    entry("motionstreamer_baselines", "t2m_model"),
    entry("motionstreamer_baselines", "MotionStreamer_vaebyh100_t2m_h100_20260204"),
    entry("motionstreamer_baselines", "motionstreamer_model_causal_TAE_t2m_babel_272_h100_20260205_20260209"),
    entry("motionstreamer_baselines", "MotionStreamer_8gpus_distributed"),
    entry("motionstreamer_baselines", "MotionStreamer_8gpus_distributed_mp"),
    entry("motionstreamer_baselines", "MotionStreamer_t2m_272_cached_embeddings_8gpu_bf16"),
    entry("motionstreamer_baselines", "MotionStreamer_vae_causal_TAE_t2m_272_h100_20260203_t2m_h100_20260206"),
    entry("misc", ".ipynb_checkpoints"),
)
```

- [ ] **Step 4: Add tests for preflight failures**

Append tests that build synthetic trees and assert:

```python
def test_preflight_rejects_missing_source(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "Experiments"
        root.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "missing source"):
            preflight(root, [ArchiveEntry("clip", "missing")])


def test_preflight_rejects_existing_destination(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "Experiments"
        (root / "run").mkdir(parents=True)
        (root / "explorations" / "clip" / "run").mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "destination exists"):
            preflight(root, [ArchiveEntry("clip", "run")])
```

- [ ] **Step 5: Implement snapshots and preflight**

Add:

```python
@dataclass(frozen=True)
class TreeSnapshot:
    byte_size: int
    file_count: int
    checkpoint_names: tuple


@dataclass(frozen=True)
class MoveRecord:
    entry: ArchiveEntry
    source: Path
    destination: Path
    snapshot: TreeSnapshot


def snapshot_tree(path):
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return TreeSnapshot(
        byte_size=sum(candidate.stat().st_size for candidate in files),
        file_count=len(files),
        checkpoint_names=tuple(
            sorted(
                str(candidate.relative_to(path))
                for candidate in files
                if candidate.suffix in {".pth", ".pt", ".ckpt", ".safetensors"}
            )
        ),
    )


def preflight(experiments_root, entries, rollback=False):
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source name in archive manifest")

    records = []
    root_device = experiments_root.stat().st_dev
    for archive_entry in entries:
        archived = (
            experiments_root / "explorations" / archive_entry.route / archive_entry.name
        )
        root_path = experiments_root / archive_entry.name
        source, destination = (
            (archived, root_path) if rollback else (root_path, archived)
        )
        if not source.is_dir():
            raise FileNotFoundError("missing source: {}".format(source))
        if destination.exists():
            raise FileExistsError("destination exists: {}".format(destination))
        if source.stat().st_dev != root_device:
            raise OSError("source is on a different filesystem: {}".format(source))
        records.append(
            MoveRecord(
                entry=archive_entry,
                source=source,
                destination=destination,
                snapshot=snapshot_tree(source),
            )
        )
    return records
```

- [ ] **Step 6: Add RED tests for apply, verify, and rollback**

Add a synthetic run containing a checkpoint and log:

```python
def test_apply_verify_and_rollback_preserve_tree(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "Experiments"
        run = root / "run"
        run.mkdir(parents=True)
        (run / "net_last.pth").write_bytes(b"checkpoint")
        (run / "log.txt").write_text("metrics")
        entries = [ArchiveEntry("clip", "run")]

        records = preflight(root, entries)
        apply_moves(records)
        verify_moves(records)
        self.assertFalse(run.exists())
        self.assertTrue(
            (root / "explorations" / "clip" / "run" / "net_last.pth").is_file()
        )

        rollback_records = preflight(root, entries, rollback=True)
        apply_moves(rollback_records)
        verify_moves(rollback_records)
        self.assertTrue((run / "net_last.pth").is_file())
```

Run the test and expect failure because `apply_moves` and `verify_moves` are
not defined.

- [ ] **Step 7: Implement atomic moves, verification, manifest rendering, and CLI**

Implement:

```python
def apply_moves(records):
    for record in records:
        record.destination.parent.mkdir(parents=True, exist_ok=True)
        if record.destination.exists():
            raise FileExistsError(
                "destination appeared during move: {}".format(record.destination)
            )
        record.source.rename(record.destination)
        verify_moves([record])


def verify_moves(records):
    for record in records:
        if record.source.exists():
            raise RuntimeError("source remains after move: {}".format(record.source))
        if not record.destination.is_dir():
            raise RuntimeError(
                "destination missing after move: {}".format(record.destination)
            )
        actual = snapshot_tree(record.destination)
        if actual != record.snapshot:
            raise RuntimeError(
                "snapshot mismatch for {}: expected {}, got {}".format(
                    record.destination, record.snapshot, actual
                )
            )
```

The CLI must:

- accept `--experiments-root` with default `Experiments`;
- make dry-run the default;
- require exactly one of `--apply`, `--verify`, or `--rollback` when not doing
  a dry-run;
- accept `--manifest` for Markdown output;
- render source, destination, route, bytes, files, checkpoint names, operation,
  and verification state;
- include a delimited machine-readable JSON block in the Markdown manifest so
  `load_manifest` can reconstruct the pre-move `TreeSnapshot` values;
- write the preflight manifest before the first move and rewrite it after all
  post-move checks pass;
- make `--verify` load the expected pre-move snapshots from `--manifest`,
  assert every old source is absent, and compare every archived destination
  against those snapshots;
- make `--rollback` load and verify the archived records from `--manifest`
  before constructing reverse moves;
- never catch and suppress a failed precondition or verification exception.

- [ ] **Step 8: Run utility tests GREEN**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_archive_exploration_results -v
```

Expected: all archive utility tests pass.

- [ ] **Step 9: Commit the tested utility**

```bash
git add scripts/archive_exploration_results.py \
  tests/test_archive_exploration_results.py
git commit -m "feat: add safe exploration result archiver"
```

---

### Task 2: Update Exploration Result Path Contracts

**Files:**

- Modify: `tests/test_exploration_launchers.py`
- Modify: runnable launchers and entrypoints under:
  - `explorations/clip/`
  - `explorations/rectified_flow/`
  - `explorations/cross_attention/`
  - `explorations/qformer/`
  - `explorations/representation_experiments/`
  - `explorations/motionstreamer_baselines/`
- Modify: `explorations/README.md`
- Modify: `EVAL_causal_TAE.sh` only if its TAE-GAN comment retains a moved path.

**Interfaces:**

- Consumes: route names from `ARCHIVE_ENTRIES`.
- Produces: active exploration launchers that read moved checkpoints and write
  future runs beneath the matching exploration result root.
- Does not modify root paths for official Causal TAE, MSA-VAE, or Global-RAG
  dependencies.

- [ ] **Step 1: Add failing launcher path-contract tests**

Extend `tests/test_exploration_launchers.py` with:

```python
EXPLORATION_OUTPUT_ROOTS = {
    "clip/TRAIN_t2m_baseline_clip.sh":
        "Experiments/explorations/clip",
    "rectified_flow/Train_t2m_rag_rf.sh":
        "Experiments/explorations/rectified_flow",
    "cross_attention/mca/Train_t2m_rag_multi_text_token.sh":
        "Experiments/explorations/cross_attention/mca",
    "cross_attention/latent_retrieval/Train_t2m_rag_latent_retr.sh":
        "Experiments/explorations/cross_attention/latent_retrieval",
    "cross_attention/local_rag/TRAIN_t2m_rag_local.sh":
        "Experiments/explorations/cross_attention/local_rag",
    "cross_attention/local_rag/TRAIN_THEN_EVAL_t2m_rag_local.sh":
        "Experiments/explorations/cross_attention/local_rag",
    "qformer/TRAIN_qformer_rag.sh":
        "Experiments/explorations/qformer",
    "representation_experiments/TRAIN_sae_v1.sh":
        "Experiments/explorations/representation_experiments",
    "representation_experiments/TRAIN_tae_gan_v1.sh":
        "Experiments/explorations/representation_experiments/TAE_GAN_Loss_",
    "motionstreamer_baselines/TRAIN_motionstreamer.sh":
        "Experiments/explorations/motionstreamer_baselines",
    "motionstreamer_baselines/TRAIN_t2m.sh":
        "Experiments/explorations/motionstreamer_baselines",
    "motionstreamer_baselines/Train_t2m_multi.sh":
        "Experiments/explorations/motionstreamer_baselines",
    "motionstreamer_baselines/TRAIN_t2m_cached.sh":
        "Experiments/explorations/motionstreamer_baselines",
}


def test_exploration_training_outputs_stay_under_archive(self):
    for relative_path, output_root in EXPLORATION_OUTPUT_ROOTS.items():
        content = (EXPLORATIONS / relative_path).read_text()
        with self.subTest(path=relative_path):
            self.assertIn(output_root, content)
```

Also add exact assertions for moved evaluation checkpoint paths:

```python
MOVED_CHECKPOINT_REFERENCES = {
    "clip/EVAL_t2m_clip_baseline.sh":
        "Experiments/explorations/clip/MotionStreamer_t2m_272_baseline_clip",
    "rectified_flow/EVAL_t2m_rag_t5_rf.sh":
        "Experiments/explorations/rectified_flow/"
        "MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_100000Iter_addEMA",
    "cross_attention/local_rag/EVAL_t2m_rag_local.sh":
        "Experiments/explorations/cross_attention/local_rag/"
        "MotionStreamer_t2m_272_msa_rag_local_L16_k3_sa_ca",
}
```

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_exploration_launchers -v
```

Expected: failures show current root-level `Experiments` paths.

- [ ] **Step 2: Update training output roots**

Change each Shell `--out-dir` or route-level `OUT_DIR` to the exact value in
`EXPLORATION_OUTPUT_ROOTS`.

Do not change:

- `explorations/representation_experiments/TRAIN_msa_vae.sh`;
- `explorations/representation_experiments/TRAIN_msa_vae_multi.sh`;
- project-history `.bak` scripts.

Those scripts still produce `MSA_VAE*` results, which the approved design keeps
at the experiment root.

- [ ] **Step 3: Update checkpoint consumers**

Search:

```bash
rg -n "Experiments/(MotionStreamer_t2m_272_baseline_clip|MotionStreamer_t2m_272_msa_rag_t5_trans662048_(rf|mca|latent_retr)|MotionStreamer_t2m_272_msa_rag_local|QFormer_|SAE_v1_|TAE_GAN_Loss_|MotionStreamer_8gpus|MotionStreamer_vaebyh100|motionstreamer_model_|MotionStreamer_t2m_272_cached_embeddings|MotionStreamer_vae_causal|t2m_model)" \
  explorations EVAL_causal_TAE.sh
```

For runnable code and examples, replace each matched moved result with its
exact `Experiments/explorations/<route>/<name>` path. Keep references to
official Causal TAE and MSA-VAE dependencies at the root.

- [ ] **Step 4: Document the result-root convention**

Add to `explorations/README.md`:

```markdown
## Experiment results

Existing and future exploration results live under
`Experiments/explorations/<route>/`. Official Causal TAE, MSA-VAE,
Global-RAG DDPM, and formal ablation results remain directly under
`Experiments/`. Exploration scripts may still consume an official checkpoint
from the root; only their own output belongs under the exploration tree.
```

- [ ] **Step 5: Run path-contract and syntax checks**

Run:

```bash
conda run -n mgpt python -m unittest \
  tests.test_exploration_launchers -v
find explorations -type f -name '*.py' -print0 |
  xargs -0 conda run -n mgpt python -m py_compile
while IFS= read -r file; do bash -n "$file"; done \
  < <(find explorations -type f \( -name '*.sh' -o -name '*.sh.bak' \))
```

Expected: launcher tests pass, every Python file compiles, and every Shell
file passes syntax validation.

- [ ] **Step 6: Commit path changes**

```bash
git add EVAL_causal_TAE.sh explorations tests/test_exploration_launchers.py
git commit -m "refactor: route exploration experiment outputs"
```

---

### Task 3: Preflight and Execute the 35 Same-Filesystem Moves

**Files:**

- Create: `docs/experiments/2026-07-25-exploration-results-archive.md`
- Modify local ignored data: `Experiments/`

**Interfaces:**

- Consumes: `ARCHIVE_ENTRIES` and utility from Task 1.
- Produces: 35 verified archived result directories plus a tracked manifest.

- [ ] **Step 1: Record the pre-move root invariants**

Run:

```bash
find Experiments -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
  sort > /tmp/msa-t2m-experiments-root-before.txt
find Experiments -mindepth 1 -maxdepth 1 -type d \
  \( -name 'causal_TAE*' -o -name 'Causal_TAE' -o -name 'MSA_VAE*' \) \
  -printf '%f\n' | sort > /tmp/msa-t2m-official-before.txt
find Experiments -mindepth 1 -maxdepth 1 -type d \
  -name 'MotionStreamer_t2m_272_msa_rag*' \
  ! -name '*_rf*' ! -name '*_mca_*' ! -name '*_latent_retr_*' \
  ! -name 'MotionStreamer_t2m_272_msa_rag_local_*' \
  -printf '%f\n' | sort > /tmp/msa-t2m-global-rag-before.txt
```

- [ ] **Step 2: Run dry-run preflight and inspect all 35 records**

Run:

```bash
conda run -n mgpt python scripts/archive_exploration_results.py \
  --experiments-root Experiments \
  --manifest /tmp/msa-t2m-exploration-results-preflight.md
```

Expected:

- exit code 0;
- exactly 35 `READY` records;
- aggregate size approximately 1.1 TB;
- one filesystem device ID;
- no missing source, existing destination, or duplicate source.

Inspect:

```bash
rg -c '^\| READY \|' /tmp/msa-t2m-exploration-results-preflight.md
```

Expected output: `35`.

- [ ] **Step 3: Execute explicit same-filesystem renames**

Run:

```bash
mkdir -p docs/experiments
conda run -n mgpt python scripts/archive_exploration_results.py \
  --experiments-root Experiments \
  --apply \
  --manifest docs/experiments/2026-07-25-exploration-results-archive.md
```

Expected:

- exactly 35 moves;
- each move verifies immediately;
- final manifest has 35 `VERIFIED` records;
- no content is copied or deleted.

If the command stops after a partial move, do not rerun `--apply` blindly.
Read the manifest, verify the completed destinations, and use the explicit
reverse mapping for only the completed records before retrying.

- [ ] **Step 4: Run archive verification mode**

Run:

```bash
conda run -n mgpt python scripts/archive_exploration_results.py \
  --experiments-root Experiments \
  --verify \
  --manifest docs/experiments/2026-07-25-exploration-results-archive.md
```

Expected: all 35 archived destinations match their recorded byte size, file
count, and checkpoint names.

- [ ] **Step 5: Verify official root invariants**

Run:

```bash
find Experiments -mindepth 1 -maxdepth 1 -type d \
  \( -name 'causal_TAE*' -o -name 'Causal_TAE' -o -name 'MSA_VAE*' \) \
  -printf '%f\n' | sort > /tmp/msa-t2m-official-after.txt
find Experiments -mindepth 1 -maxdepth 1 -type d \
  -name 'MotionStreamer_t2m_272_msa_rag*' \
  ! -name '*_rf*' ! -name '*_mca_*' ! -name '*_latent_retr_*' \
  ! -name 'MotionStreamer_t2m_272_msa_rag_local_*' \
  -printf '%f\n' | sort > /tmp/msa-t2m-global-rag-after.txt
diff -u /tmp/msa-t2m-official-before.txt /tmp/msa-t2m-official-after.txt
diff -u /tmp/msa-t2m-global-rag-before.txt /tmp/msa-t2m-global-rag-after.txt
```

Expected: both diffs are empty.

- [ ] **Step 6: Verify no exploration method remains at the root**

Run:

```bash
conda run -n mgpt python -c "
from pathlib import Path
from scripts.archive_exploration_results import ARCHIVE_ENTRIES
root = Path('Experiments')
remaining = [entry.name for entry in ARCHIVE_ENTRIES if (root / entry.name).exists()]
assert not remaining, remaining
print('root_exploration_results=0')
"
```

Expected: `root_exploration_results=0`.

- [ ] **Step 7: Commit the verified manifest**

```bash
git add docs/experiments/2026-07-25-exploration-results-archive.md
git commit -m "docs: record exploration result archive"
```

The ignored `Experiments/` contents are intentionally absent from the Git
commit.

---

### Task 4: Run Final Repository and Layout Verification

**Files:**

- Verify: `scripts/archive_exploration_results.py`
- Verify: `tests/test_archive_exploration_results.py`
- Verify: `tests/test_exploration_launchers.py`
- Verify: `docs/experiments/2026-07-25-exploration-results-archive.md`

**Interfaces:**

- Consumes: completed code, path updates, and result moves.
- Produces: final evidence that code and filesystem layout match the approved
  design.

- [ ] **Step 1: Search runnable code for stale moved-result paths**

Run:

```bash
conda run -n mgpt python -c "
from pathlib import Path
from scripts.archive_exploration_results import ARCHIVE_ENTRIES

files = [
    path for path in Path('explorations').rglob('*')
    if path.is_file()
    and (path.suffix == '.py' or path.name.endswith('.sh'))
    and 'project_history' not in path.parts
]
files.append(Path('EVAL_causal_TAE.sh'))
stale = []
for path in files:
    text = path.read_text(errors='replace')
    for entry in ARCHIVE_ENTRIES:
        needle = 'Experiments/' + entry.name
        if needle in text:
            stale.append('{}: {}'.format(path, needle))
assert not stale, '\\n'.join(stale)
print('stale_active_result_paths=0')
"
```

Expected: no stale active reference.

- [ ] **Step 2: Run the complete test suite**

```bash
conda run -n mgpt python -m unittest discover -s tests -v
```

Expected: all existing 28 tests plus the new archive tests pass.

- [ ] **Step 3: Compile modified Python and validate modified Shell files**

```bash
conda run -n mgpt python -m py_compile \
  scripts/archive_exploration_results.py \
  tests/test_archive_exploration_results.py \
  tests/test_exploration_launchers.py
while IFS= read -r file; do bash -n "$file"; done \
  < <(git show --pretty='' --name-only HEAD~2..HEAD |
      rg '\.(sh|sh\.bak)$' | sort -u)
```

Expected: exit code 0.

- [ ] **Step 4: Verify Git and ignored-data boundaries**

```bash
git diff --check
git status --short --branch
git check-ignore Experiments/explorations
```

Expected:

- no whitespace errors;
- `Experiments/explorations` remains ignored;
- `paper writing/Research-Paper-Writing-Skills` is the only unrelated dirty
  path.

- [ ] **Step 5: Review the final commit range**

```bash
git log --oneline --decorate origin/main..HEAD
git diff --stat origin/main..HEAD
```

Confirm that commits contain only the design, plan, utility, tests, path
updates, and archive manifest. Do not stage or commit the unrelated submodule.

- [ ] **Step 6: Request code review and fix any Critical or Important findings**

Ask the reviewer to compare the implementation with:

`docs/superpowers/specs/2026-07-25-experiment-results-archive-design.md`

The review must include filesystem-layout evidence and confirmation that no
official result moved.

- [ ] **Step 7: Present completion state**

Report:

- the 35 moved directories and aggregate verified size;
- the new `Experiments/explorations/` categories;
- retained official root invariants;
- test, compile, Shell, and whitespace results;
- commit hashes;
- the remaining unrelated submodule status.

Do not push unless the user explicitly asks for a remote update.
