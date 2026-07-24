import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from humanml3d_272.msa_rag_cache import (
    CacheValidationError,
    PackedMSARAGCache,
    build_cache,
    validate_cache,
)
from tests.msa_rag_fixtures import create_rag_fixture


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class MSARAGCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = create_rag_fixture(self.root)
        self.cache_dir = self.root / "cache"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def build(self, **overrides):
        arguments = {
            "dataset_name": "t2m_272",
            "motion_latent_dir": str(self.fixture["motion_dir"]),
            "text_latent_dir": str(self.fixture["text_dir"]),
            "hcls_dir": str(self.fixture["hcls_dir"]),
            "cache_dir": str(self.cache_dir),
            "topk": 2,
            "text_embed_dim": 2,
            "retrieval_batch_size": 3,
        }
        arguments.update(overrides)
        with working_directory(self.root):
            return build_cache(**arguments)

    def validate(self, **overrides):
        arguments = {
            "cache_dir": str(self.cache_dir),
            "dataset_name": "t2m_272",
            "motion_latent_dir": str(self.fixture["motion_dir"]),
            "text_latent_dir": str(self.fixture["text_dir"]),
            "hcls_dir": str(self.fixture["hcls_dir"]),
            "topk": 2,
            "text_embed_dim": 2,
        }
        arguments.update(overrides)
        with working_directory(self.root):
            return validate_cache(**arguments)

    def test_builds_numeric_packed_arrays_with_expected_retrieval(self):
        manifest = self.build()
        cache = PackedMSARAGCache(str(self.cache_dir), requested_topk=2)

        self.assertEqual(manifest["sample_count"], 3)
        self.assertEqual(cache.sample_ids, ["000001", "000002", "000003"])
        text, top_hcls, scores, motion = cache.get(sample_idx=0, caption_idx=0)
        np.testing.assert_array_equal(text, np.array([1.0, 0.0], np.float32))
        np.testing.assert_array_equal(
            top_hcls,
            np.array([[0.6, 0.8], [0.0, 1.0]], np.float32),
        )
        np.testing.assert_allclose(
            scores,
            np.array([0.6, 0.0], np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            motion,
            np.arange(8, dtype=np.float32).reshape(4, 2),
        )

        for filename in (
            "motion_values.npy",
            "text_values.npy",
            "hcls_values.npy",
            "retrieval_indices.npy",
            "retrieval_scores.npy",
        ):
            self.assertNotEqual(np.load(self.cache_dir / filename).dtype, object)

    def test_cache_built_for_larger_topk_can_serve_smaller_topk(self):
        self.build()
        cache = PackedMSARAGCache(str(self.cache_dir), requested_topk=1)

        _, top_hcls, scores, _ = cache.get(sample_idx=0, caption_idx=0)
        np.testing.assert_array_equal(top_hcls, np.array([[0.6, 0.8]], np.float32))
        np.testing.assert_allclose(scores, np.array([0.6], np.float32), rtol=1e-5, atol=1e-6)

    def test_retrieval_chunk_sizes_preserve_topk_and_scores(self):
        self.build(retrieval_batch_size=1)
        cache_one = PackedMSARAGCache(str(self.cache_dir), requested_topk=2)
        indices_one = np.array(cache_one.retrieval_indices)
        scores_one = np.array(cache_one.retrieval_scores)

        self.build(retrieval_batch_size=3, force=True)
        cache_three = PackedMSARAGCache(str(self.cache_dir), requested_topk=2)
        np.testing.assert_array_equal(cache_three.retrieval_indices, indices_one)
        np.testing.assert_allclose(cache_three.retrieval_scores, scores_one, rtol=1e-6, atol=1e-6)

    def test_rejects_requested_topk_larger_than_cache(self):
        self.build()
        with self.assertRaisesRegex(CacheValidationError, "topk"):
            self.validate(topk=3)

    def test_rejects_changed_self_exclusion(self):
        self.build()
        with self.assertRaisesRegex(CacheValidationError, "exclude_self"):
            self.validate(exclude_self=False)

    def test_rejects_changed_text_dimension(self):
        self.build()
        with self.assertRaisesRegex(CacheValidationError, "text_embed_dim"):
            self.validate(text_embed_dim=3)

    def test_rejects_missing_array(self):
        self.build()
        (self.cache_dir / "motion_values.npy").unlink()
        with self.assertRaisesRegex(CacheValidationError, "motion_values.npy"):
            self.validate()

    def test_rejects_modified_source_file(self):
        self.build()
        source = self.fixture["text_dir"] / "000001.npy"
        np.save(source, np.array([[0.5, 0.5]], dtype=np.float32))
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(CacheValidationError, "000001.npy"):
            self.validate()

    def test_rejects_incomplete_cache_without_manifest(self):
        self.cache_dir.mkdir()
        np.save(self.cache_dir / "motion_values.npy", np.zeros((1, 2), np.float32))
        with self.assertRaisesRegex(CacheValidationError, "manifest.json"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
