import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.archive_exploration_results import (
    ARCHIVE_ENTRIES,
    ArchiveEntry,
    apply_moves,
    load_manifest,
    main,
    preflight,
    render_manifest,
    verify_moves,
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

    def test_manifest_round_trip_preserves_pre_move_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            run = root / "run"
            run.mkdir(parents=True)
            (run / "checkpoint.ckpt").write_bytes(b"model")
            records = preflight(root, [ArchiveEntry("clip", "run")])
            manifest_path = root / "archive.md"
            manifest_path.write_text(render_manifest(records, "archive"))

            loaded = load_manifest(manifest_path, root)

            self.assertEqual(loaded, records)

    def test_cli_apply_verify_and_rollback_use_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            for entry in ARCHIVE_ENTRIES:
                (root / entry.name).mkdir(parents=True)
            run = root / ARCHIVE_ENTRIES[0].name
            (run / "net_last.pth").write_bytes(b"checkpoint")
            manifest_path = root / "archive.md"
            options = ["--experiments-root", str(root), "--manifest", str(manifest_path)]

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(options + ["--apply"]), 0)
            self.assertTrue(manifest_path.is_file())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(options + ["--verify"]), 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(options + ["--rollback"]), 0)
            self.assertTrue((run / "net_last.pth").is_file())
