import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from humanml3d_272 import dataset_msa_vae
from humanml3d_272.dataset_msa_vae import MSAVAEDataset


class FullSequenceDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)

        motion = np.zeros((70, 272), dtype=np.float32)
        motion[:, 0] = np.arange(70, dtype=np.float32)
        local = np.stack(
            [
                np.arange(70, dtype=np.float32),
                -np.arange(70, dtype=np.float32),
            ],
            axis=1,
        )
        global_text = np.array([[3.0, 4.0]], dtype=np.float32)

        local_path = root / "local.npy"
        global_path = root / "global.npy"
        np.save(local_path, local)
        np.save(global_path, global_text)

        dataset = MSAVAEDataset.__new__(MSAVAEDataset)
        dataset.window_size = 64
        dataset.unit_length = 4
        dataset.text_embed_dim = 2
        dataset.mean = np.zeros(272, dtype=np.float32)
        dataset.std = np.ones(272, dtype=np.float32)
        dataset.sequence_mode = "window"
        dataset.data = [
            {
                "name": "sample",
                "motion": motion,
                "captions": ["walk then sit"],
                "caption_indices": [0],
                "local_text_path": str(local_path),
                "has_local": True,
                "global_text_path": str(global_path),
                "has_global": True,
            }
        ]
        self.dataset = dataset

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_and_window_views_share_aligned_record(self):
        full = self.dataset.get_item(0, "full")

        self.assertEqual(full[0].shape, (68, 272))
        self.assertEqual(full[4].shape, (17, 2))
        self.assertEqual(full[6], 70)
        self.assertEqual(full[-1], 68)
        np.testing.assert_array_equal(full[0][:, 0], np.arange(68))

        with mock.patch(
            "humanml3d_272.dataset_msa_vae.random.randint",
            side_effect=[4, 0],
        ):
            window = self.dataset.get_item(0, "window")

        self.assertEqual(window[0].shape, (64, 272))
        self.assertEqual(window[4].shape, (16, 2))
        self.assertEqual(window[-1], 64)
        np.testing.assert_array_equal(window[0][:, 0], np.arange(4, 68))
        np.testing.assert_allclose(window[4][0], np.array([5.5, -5.5]))

    def test_invalid_sequence_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sequence_mode"):
            self.dataset.get_item(0, "prefix")


class FullSequenceCollateTest(unittest.TestCase):
    @staticmethod
    def _item(length):
        latent_length = length // 4
        return (
            np.ones((length, 272), dtype=np.float32),
            f"motion-{length}",
            np.array([3.0, 4.0], dtype=np.float32),
            True,
            np.ones((latent_length, 2), dtype=np.float32),
            True,
            length + 2,
            np.array([1.0, 2.0], dtype=np.float32),
            length,
        )

    def test_collate_pads_motion_and_local_targets(self):
        batch = dataset_msa_vae.collate_fn(
            [self._item(64), self._item(68)]
        )

        self.assertEqual(batch[0].shape, (2, 68, 272))
        self.assertEqual(batch[4].shape, (2, 17, 2))
        torch.testing.assert_close(batch[-1], torch.tensor([64, 68]))
        self.assertTrue(torch.all(batch[0][0, 64:] == 0))
        self.assertTrue(torch.all(batch[4][0, 16:] == 0))
        self.assertEqual(batch[1], ["motion-64", "motion-68"])

    def test_full_and_window_views_share_underlying_records(self):
        base = object()
        dataset = MSAVAEDataset.__new__(MSAVAEDataset)
        dataset.window_size = 64
        dataset.unit_length = 4
        dataset.data = [
            {"motion": np.zeros((70, 272), dtype=np.float32), "marker": base}
        ]

        full = dataset_msa_vae.MSAVAESequenceView(dataset, "full")
        window = dataset_msa_vae.MSAVAESequenceView(dataset, "window")

        self.assertIs(full.dataset, dataset)
        self.assertIs(window.dataset, dataset)
        self.assertIs(full.dataset.data[0]["marker"], base)
        self.assertEqual(full.source_lengths, [68])
        self.assertEqual(window.source_lengths, [64])


class LengthBucketBatchSamplerTest(unittest.TestCase):
    def test_sampler_covers_indices_and_is_epoch_deterministic(self):
        lengths = [64, 68, 192, 196, 72, 200]
        sampler_a = dataset_msa_vae.LengthBucketBatchSampler(
            lengths,
            batch_size=2,
            bucket_size=4,
            drop_last=False,
            seed=17,
        )
        sampler_b = dataset_msa_vae.LengthBucketBatchSampler(
            lengths,
            batch_size=2,
            bucket_size=4,
            drop_last=False,
            seed=17,
        )

        sampler_a.set_epoch(1)
        sampler_b.set_epoch(1)
        batches_a = list(sampler_a)
        batches_b = list(sampler_b)

        self.assertEqual(batches_a, batches_b)
        self.assertEqual(
            sorted(index for batch in batches_a for index in batch),
            list(range(len(lengths))),
        )
        self.assertTrue(all(len(batch) <= 2 for batch in batches_a))
        self.assertFalse(any(0 in batch and 5 in batch for batch in batches_a))
        self.assertEqual(len(sampler_a), 3)

    def test_drop_last_omits_only_incomplete_batch(self):
        sampler = dataset_msa_vae.LengthBucketBatchSampler(
            [64, 68, 72, 76, 80],
            batch_size=2,
            bucket_size=4,
            drop_last=True,
            seed=3,
        )

        batches = list(sampler)

        self.assertEqual(len(batches), 2)
        self.assertEqual(sum(len(batch) for batch in batches), 4)

    def test_loader_factory_uses_bucket_sampler_only_for_full_view(self):
        class FakeDataset:
            window_size = 64
            unit_length = 4
            data = [
                {"motion": np.zeros((68, 272), dtype=np.float32)},
                {"motion": np.zeros((196, 272), dtype=np.float32)},
            ]

            def __len__(self):
                return len(self.data)

            def get_item(self, index, sequence_mode):
                length = (
                    len(self.data[index]["motion"])
                    if sequence_mode == "full"
                    else self.window_size
                )
                return FullSequenceCollateTest._item(length)

        dataset = FakeDataset()
        full_loader = dataset_msa_vae.make_loader(
            dataset,
            "full",
            batch_size=1,
            num_workers=0,
            bucket_size=2,
            drop_last=False,
            seed=5,
        )
        window_loader = dataset_msa_vae.make_loader(
            dataset,
            "window",
            batch_size=1,
            num_workers=0,
            bucket_size=2,
            drop_last=False,
            seed=5,
        )

        self.assertIsInstance(
            full_loader.batch_sampler,
            dataset_msa_vae.LengthBucketBatchSampler,
        )
        self.assertNotIsInstance(
            window_loader.batch_sampler,
            dataset_msa_vae.LengthBucketBatchSampler,
        )
        full_lengths = sorted(
            batch[-1].item() for batch in full_loader
        )
        self.assertEqual(full_lengths, [68, 196])

    def test_cycle_advances_loader_epoch_before_each_pass(self):
        class EpochIterable:
            def __init__(self):
                self.epochs = []

            def set_epoch(self, epoch):
                self.epochs.append(epoch)

            def __iter__(self):
                return iter(["batch"])

        iterable = EpochIterable()
        iterator = dataset_msa_vae.cycle(iterable)

        self.assertEqual(next(iterator), "batch")
        self.assertEqual(next(iterator), "batch")
        self.assertEqual(iterable.epochs, [0, 1])


if __name__ == "__main__":
    unittest.main()
