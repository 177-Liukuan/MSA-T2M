import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

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
    @staticmethod
    def _write_manifest_payload(path, entries):
        payload = {
            "version": 1,
            "records": [
                {
                    "route": entry.route,
                    "name": entry.name,
                    "snapshot": {
                        "byte_size": 0,
                        "file_count": 0,
                        "checkpoint_names": [],
                    },
                }
                for entry in entries
            ],
        }
        path.write_text(
            "<!-- ARCHIVE_MANIFEST_JSON\n{}\nARCHIVE_MANIFEST_JSON -->".format(
                json.dumps(payload)
            )
        )

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

    def test_preflight_rejects_symlink_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            target = root / "real-run"
            source = root / "run"
            target.mkdir(parents=True)
            source.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "source symlink"):
                preflight(root, [ArchiveEntry("clip", "run")])

    def test_preflight_rejects_dangling_destination_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            (root / "run").mkdir(parents=True)
            destination = root / "explorations" / "clip" / "run"
            destination.parent.mkdir(parents=True)
            destination.symlink_to(root / "missing")

            with self.assertRaisesRegex(FileExistsError, "destination exists"):
                preflight(root, [ArchiveEntry("clip", "run")])

    def test_preflight_rejects_explorations_parent_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            outside = Path(temporary_directory) / "outside"
            (root / "run").mkdir(parents=True)
            outside.mkdir()
            (root / "explorations").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(OSError, "destination parent symlink"):
                preflight(root, [ArchiveEntry("clip", "run")])

    def test_apply_rejects_dangling_destination_symlink_created_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            source = root / "run"
            source.mkdir(parents=True)
            records = preflight(root, [ArchiveEntry("clip", "run")])
            destination = root / "explorations" / "clip" / "run"
            destination.parent.mkdir(parents=True)
            destination.symlink_to(root / "missing")

            with self.assertRaisesRegex(FileExistsError, "destination appeared"):
                apply_moves(records)
            self.assertTrue(source.is_dir())
            self.assertTrue(destination.is_symlink())

    def test_apply_rejects_source_symlink_created_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            source = root / "run"
            source.mkdir(parents=True)
            records = preflight(root, [ArchiveEntry("clip", "run")])
            source.rmdir()
            target = root / "replacement"
            target.mkdir()
            source.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "source symlink appeared"):
                apply_moves(records)
            self.assertTrue(source.is_symlink())
            self.assertFalse(records[0].destination.exists())

    def test_apply_rejects_route_parent_symlink_created_after_preflight(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            outside = Path(temporary_directory) / "outside"
            source = root / "run"
            source.mkdir(parents=True)
            records = preflight(root, [ArchiveEntry("clip", "run")])
            route_parent = root / "explorations" / "clip"
            route_parent.mkdir(parents=True)
            route_parent.rmdir()
            outside.mkdir()
            route_parent.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(OSError, "destination parent symlink"):
                apply_moves(records)
            self.assertTrue(source.is_dir())
            self.assertTrue(route_parent.is_symlink())

    def test_preflight_rejects_destination_on_another_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            source = root / "run"
            destination_parent = root / "explorations"
            source.mkdir(parents=True)
            destination_parent.mkdir()
            original_stat = Path.stat

            def stat_with_destination_on_another_device(path, *args, **kwargs):
                result = original_stat(path, *args, **kwargs)
                if path == destination_parent:
                    values = list(result)
                    values[2] = result.st_dev + 1
                    return os.stat_result(values)
                return result

            with patch.object(Path, "stat", new=stat_with_destination_on_another_device):
                with self.assertRaisesRegex(OSError, "destination is on a different filesystem"):
                    preflight(root, [ArchiveEntry("clip", "run")])
            self.assertTrue(source.is_dir())
            self.assertFalse((destination_parent / "clip" / "run").exists())

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
            for entry in ARCHIVE_ENTRIES:
                (root / entry.name).mkdir(parents=True)
            run = root / ARCHIVE_ENTRIES[0].name
            (run / "checkpoint.ckpt").write_bytes(b"model")
            records = preflight(root, ARCHIVE_ENTRIES)
            manifest_path = root / "archive.md"
            manifest_path.write_text(render_manifest(records, "archive"))

            loaded = load_manifest(manifest_path, root)

            self.assertEqual(loaded, records)

    def test_load_manifest_rejects_wrong_entry_count_and_membership(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            root.mkdir()
            manifest_path = root / "archive.md"

            self._write_manifest_payload(manifest_path, ARCHIVE_ENTRIES[:-1])
            with self.assertRaisesRegex(ValueError, "exactly 35"):
                load_manifest(manifest_path, root)

            altered_entries = list(ARCHIVE_ENTRIES)
            altered_entries[0] = ArchiveEntry("clip", "unexpected")
            self._write_manifest_payload(manifest_path, altered_entries)
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_manifest(manifest_path, root)

    def test_load_manifest_rejects_traversal_and_absolute_components(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "Experiments"
            root.mkdir()
            manifest_path = root / "archive.md"
            for unsafe_entry in (
                ArchiveEntry("../clip", ARCHIVE_ENTRIES[0].name),
                ArchiveEntry("clip", "/outside"),
            ):
                altered_entries = list(ARCHIVE_ENTRIES)
                altered_entries[0] = unsafe_entry
                self._write_manifest_payload(manifest_path, altered_entries)
                with self.subTest(entry=unsafe_entry):
                    with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
                        load_manifest(manifest_path, root)

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
