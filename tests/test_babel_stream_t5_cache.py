import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from humanml3d_272 import babel_stream_t5_cache
from humanml3d_272.babel_stream_t5_cache import (
    BabelStreamRecord,
    CacheBuildError,
    build_cache,
    expand_segment_embeddings,
    parse_babel_stream_text,
    validate_cache_manifest,
)


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array(
            [[float(index), float(index + 1)] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


class OffsetEncoder:
    def __init__(self, offset):
        self.offset = offset

    def encode(self, texts):
        return np.array(
            [
                [float(self.offset + index), float(self.offset + index + 1)]
                for index, _ in enumerate(texts)
            ],
            dtype=np.float32,
        )


class BabelStreamT5CacheTest(unittest.TestCase):
    def _write_valid_sources(self, root):
        motion_dir = root / "motion"
        text_dir = root / "text"
        motion_dir.mkdir()
        text_dir.mkdir()
        np.save(motion_dir / "seq_a.npy", np.zeros((5, 272), dtype=np.float32))
        np.save(motion_dir / "seq_b.npy", np.zeros((6, 272), dtype=np.float32))
        (text_dir / "seq_a.txt").write_text(
            "walk#walk/VERB#0.0#0.0*sit#sit/VERB#0.0#0.0#2\n"
        )
        (text_dir / "seq_b.txt").write_text(
            "sit#sit/VERB#0.0#0.0*jump#jump/VERB#0.0#0.0#3\n"
        )
        return motion_dir, text_dir

    def _expected(self, motion_dir, text_dir):
        return {
            "split": "train",
            "model_signature": "fake-t5",
            "embedding_dim": 2,
            "motion_dir": str(motion_dir.resolve()),
            "text_dir": str(text_dir.resolve()),
        }

    def test_cli_help_runs_from_repository_root_without_loading_sentence_transformers(self):
        result = subprocess.run(
            [sys.executable, "scripts/prepare_babel_stream_t5.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--t5-model-path", result.stdout)

    def test_parse_two_segment_record(self):
        line = "throw#throw/VERB#0.0#0.0*catch#catch/VERB#0.0#0.0#3"
        self.assertEqual(
            parse_babel_stream_text(line),
            BabelStreamRecord("throw", "catch", 3),
        )

    def test_expand_uses_boundary_and_exact_motion_length(self):
        record = BabelStreamRecord("walk", "sit", 2)
        embeddings = {
            "walk": np.array([1.0, 0.0], dtype=np.float32),
            "sit": np.array([0.0, 1.0], dtype=np.float32),
        }
        result = expand_segment_embeddings(record, 5, embeddings)
        np.testing.assert_array_equal(
            result,
            np.array(
                [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32
            ),
        )

    def test_parse_rejects_missing_segment_and_invalid_boundary(self):
        with self.assertRaisesRegex(ValueError, "two segments"):
            parse_babel_stream_text("walk#walk/VERB#0.0#0.0")
        with self.assertRaisesRegex(ValueError, "boundary"):
            parse_babel_stream_text(
                "walk#walk/VERB#0.0#0.0*sit#sit/VERB#0.0#0.0#bad"
            )

    def test_build_cache_encodes_unique_text_and_validates_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            encoder = FakeEncoder()

            manifest = build_cache(
                "train", motion_dir, text_dir, output_dir, encoder, "fake-t5"
            )

            self.assertEqual(sorted(sum(encoder.calls, [])), ["jump", "sit", "walk"])
            self.assertEqual(manifest["valid_samples"], 2)
            self.assertEqual(manifest["rejected_samples"], 0)
            self.assertEqual(manifest["embedding_dim"], 2)
            self.assertEqual(np.load(output_dir / "seq_a.npy").shape, (5, 2))
            self.assertEqual(np.load(output_dir / "seq_b.npy").shape, (6, 2))

            cached = build_cache(
                "train", motion_dir, text_dir, output_dir, encoder, "fake-t5"
            )
            self.assertEqual(cached, manifest)
            self.assertEqual(len(encoder.calls), 1)
            self.assertEqual(
                validate_cache_manifest(
                    output_dir / "manifest.json",
                    self._expected(motion_dir, text_dir),
                ),
                manifest,
            )

    def test_failed_overwrite_is_rejected_instead_of_accepting_mixed_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            build_cache(
                "train", motion_dir, text_dir, output_dir, OffsetEncoder(0), "fake-t5"
            )
            original_seq_a = np.load(output_dir / "seq_a.npy").copy()
            original_save = babel_stream_t5_cache._atomic_save_array
            saved_names = []

            def fail_after_first_replacement(directory, filename, array):
                if saved_names:
                    raise OSError("injected overwrite failure")
                saved_names.append(filename)
                original_save(directory, filename, array)

            with patch.object(
                babel_stream_t5_cache,
                "_atomic_save_array",
                side_effect=fail_after_first_replacement,
            ):
                with self.assertRaisesRegex(OSError, "injected overwrite failure"):
                    build_cache(
                        "train",
                        motion_dir,
                        text_dir,
                        output_dir,
                        OffsetEncoder(100),
                        "fake-t5",
                        overwrite=True,
                    )

            # A failed overwrite must leave the complete previous generation
            # usable rather than replacing one array under the old manifest.
            validate_cache_manifest(
                output_dir / "manifest.json", self._expected(motion_dir, text_dir)
            )
            np.testing.assert_array_equal(
                np.load(output_dir / "seq_a.npy"),
                original_seq_a,
            )

    def test_overwrite_after_source_deletion_publishes_exact_clean_membership(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            build_cache(
                "train", motion_dir, text_dir, output_dir, FakeEncoder(), "fake-t5"
            )
            (motion_dir / "seq_b.npy").unlink()
            (text_dir / "seq_b.txt").unlink()

            manifest = build_cache(
                "train",
                motion_dir,
                text_dir,
                output_dir,
                FakeEncoder(),
                "fake-t5",
                overwrite=True,
            )

            self.assertEqual(set(manifest["records"]), {"seq_a"})
            self.assertEqual(
                {path.name for path in output_dir.glob("*.npy")},
                {"seq_a.npy"},
            )
            validate_cache_manifest(
                output_dir / "manifest.json", self._expected(motion_dir, text_dir)
            )

    def test_validator_rejects_extra_array_not_declared_by_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            build_cache(
                "train", motion_dir, text_dir, output_dir, FakeEncoder(), "fake-t5"
            )
            np.save(output_dir / "stale.npy", np.zeros((4, 2), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "cache array membership mismatch"):
                validate_cache_manifest(
                    output_dir / "manifest.json", self._expected(motion_dir, text_dir)
                )

    def test_source_mutation_during_build_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            original_save = babel_stream_t5_cache._atomic_save_array
            mutated = []

            def mutate_after_first_array(directory, filename, array):
                original_save(directory, filename, array)
                if not mutated:
                    mutated.append(filename)
                    (text_dir / "seq_a.txt").write_text(
                        "jump#jump/VERB#0.0#0.0*sit#sit/VERB#0.0#0.0#2\n"
                    )

            with patch.object(
                babel_stream_t5_cache,
                "_atomic_save_array",
                side_effect=mutate_after_first_array,
            ):
                with self.assertRaisesRegex(CacheBuildError, "changed during cache build"):
                    build_cache(
                        "train",
                        motion_dir,
                        text_dir,
                        output_dir,
                        FakeEncoder(),
                        "fake-t5",
                    )

            self.assertFalse((output_dir / "manifest.json").exists())

    def test_validate_rejects_cached_array_with_wrong_shape_or_dtype(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            build_cache(
                "train", motion_dir, text_dir, output_dir, FakeEncoder(), "fake-t5"
            )
            np.save(output_dir / "seq_a.npy", np.zeros((5, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                validate_cache_manifest(
                    output_dir / "manifest.json", self._expected(motion_dir, text_dir)
                )

            build_cache(
                "train",
                motion_dir,
                text_dir,
                output_dir,
                FakeEncoder(),
                "fake-t5",
                overwrite=True,
            )
            np.save(output_dir / "seq_a.npy", np.zeros((5, 2), dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                validate_cache_manifest(
                    output_dir / "manifest.json", self._expected(motion_dir, text_dir)
                )

    def test_validate_rejects_stale_text_source_signature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir, text_dir = self._write_valid_sources(root)
            output_dir = root / "cache"
            build_cache(
                "train", motion_dir, text_dir, output_dir, FakeEncoder(), "fake-t5"
            )
            (text_dir / "seq_a.txt").write_text(
                "walk#walk/VERB#0.0#0.0*turn#turn/VERB#0.0#0.0#2\n"
            )

            with self.assertRaisesRegex(ValueError, "source records mismatch"):
                validate_cache_manifest(
                    output_dir / "manifest.json", self._expected(motion_dir, text_dir)
                )

    def test_build_cache_rejects_malformed_input_before_manifest_publication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion_dir = root / "motion"
            text_dir = root / "text"
            output_dir = root / "cache"
            motion_dir.mkdir()
            text_dir.mkdir()
            np.save(motion_dir / "good.npy", np.zeros((5, 272), dtype=np.float32))
            np.save(motion_dir / "bad.npy", np.zeros((5, 272), dtype=np.float32))
            (text_dir / "good.txt").write_text(
                "walk#walk/VERB#0.0#0.0*run#run/VERB#0.0#0.0#2\n"
            )
            (text_dir / "bad.txt").write_text("missing#segment#boundary\n")

            with self.assertRaises(CacheBuildError) as raised:
                build_cache(
                    "train", motion_dir, text_dir, output_dir, FakeEncoder(), "fake-t5"
                )

            self.assertEqual(raised.exception.rejections, ("bad: expected exactly two segments",))
            self.assertFalse((output_dir / "manifest.json").exists())

    def test_expand_rejects_boundary_outside_motion(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            expand_segment_embeddings(
                BabelStreamRecord("walk", "sit", 5),
                5,
                {"walk": np.ones(2, dtype=np.float32), "sit": np.ones(2, dtype=np.float32)},
            )


if __name__ == "__main__":
    unittest.main()
