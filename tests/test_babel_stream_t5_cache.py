import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


class BabelStreamT5CacheTest(unittest.TestCase):
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
            motion_dir = root / "motion"
            text_dir = root / "text"
            output_dir = root / "cache"
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
                    {
                        "split": "train",
                        "model_signature": "fake-t5",
                        "embedding_dim": 2,
                        "motion_dir": str(motion_dir.resolve()),
                        "text_dir": str(text_dir.resolve()),
                    },
                ),
                manifest,
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
