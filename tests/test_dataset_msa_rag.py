import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanml3d_272.dataset_msa_rag import (
    Text2MotionMSARAGDataset,
    collate_fn,
)
from humanml3d_272.msa_rag_cache import CacheValidationError, build_cache
from tests.msa_rag_fixtures import create_rag_fixture
from tests.test_msa_rag_cache import working_directory


class MSARAGDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = create_rag_fixture(self.root)
        self.cache_dir = self.root / "cache"
        with working_directory(self.root):
            build_cache(
                dataset_name="t2m_272",
                motion_latent_dir=str(self.fixture["motion_dir"]),
                text_latent_dir=str(self.fixture["text_dir"]),
                hcls_dir=str(self.fixture["hcls_dir"]),
                cache_dir=str(self.cache_dir),
                topk=2,
                text_embed_dim=2,
                retrieval_batch_size=3,
            )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_dataset(self, cache_mode, cache_dir=None):
        return Text2MotionMSARAGDataset(
            dataset_name="t2m_272",
            motion_latent_dir=str(self.fixture["motion_dir"]),
            text_latent_dir=str(self.fixture["text_dir"]),
            hcls_dir=str(self.fixture["hcls_dir"]),
            topk=2,
            exclude_self=True,
            text_embed_dim=2,
            cache_mode=cache_mode,
            cache_dir=cache_dir,
        )

    def test_reference_and_packed_modes_return_equivalent_items(self):
        with working_directory(self.root):
            reference = self.make_dataset("reference")
            packed = self.make_dataset("packed", str(self.cache_dir))

            self.assertEqual(
                [entry["id"] for entry in reference.data],
                packed.sample_ids,
            )
            for sample_idx, text in enumerate(self.fixture["texts"].values()):
                caption_count = text.shape[0] if text.ndim == 2 else 1
                for caption_idx in range(caption_count):
                    reference_item = reference.get_item(sample_idx, caption_idx)
                    packed_item = packed.get_item(sample_idx, caption_idx)
                    for actual, expected in zip(packed_item, reference_item):
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-6, atol=1e-6
                        )

    def test_collate_is_identical_for_reference_and_packed_items(self):
        with working_directory(self.root):
            reference = self.make_dataset("reference")
            packed = self.make_dataset("packed", str(self.cache_dir))
            reference_batch = collate_fn(
                [reference.get_item(1, 0), reference.get_item(2, 0)]
            )
            packed_batch = collate_fn(
                [packed.get_item(1, 0), packed.get_item(2, 0)]
            )

        for actual, expected in zip(packed_batch, reference_batch):
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(packed_batch[-1].shape), (2, 5, 2))
        torch.testing.assert_close(packed_batch[-1][0, 3:], torch.zeros(2, 2))

    def test_rejects_unknown_cache_mode(self):
        with working_directory(self.root):
            with self.assertRaisesRegex(ValueError, "cache_mode"):
                self.make_dataset("automatic")

    def test_packed_mode_requires_cache_dir(self):
        with working_directory(self.root):
            with self.assertRaisesRegex(ValueError, "cache_dir"):
                self.make_dataset("packed")

    def test_packed_mode_rejects_invalid_cache(self):
        (self.cache_dir / "retrieval_scores.npy").unlink()
        with working_directory(self.root):
            with self.assertRaises(CacheValidationError):
                self.make_dataset("packed", str(self.cache_dir))


if __name__ == "__main__":
    unittest.main()
