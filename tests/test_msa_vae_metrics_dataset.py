import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanml3d_272.dataset_eval_msa_vae_metrics import (
    MSAVAEMetricsDataset,
    collate_msa_vae_metrics,
    make_msa_vae_metrics_loader,
)


class MSAVAEMetricsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "motion_data").mkdir()
        (self.root / "texts").mkdir()
        (self.root / "mean_std").mkdir()
        np.save(self.root / "mean_std" / "Mean.npy", np.zeros(272))
        np.save(self.root / "mean_std" / "Std.npy", np.ones(272))
        self.split_file = self.root / "test.txt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_fixture(self, split_ids, motions, texts):
        self.split_file.write_text(
            "\n".join(split_ids) + "\n",
            encoding="utf-8",
        )
        for sample_id, motion in motions.items():
            np.save(
                self.root / "motion_data" / f"{sample_id}.npy",
                np.asarray(motion, dtype=np.float32),
            )
        for sample_id, lines in texts.items():
            (self.root / "texts" / f"{sample_id}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

    def _dataset(self):
        return MSAVAEMetricsDataset(
            data_root=self.root,
            split_file=self.split_file,
            unit_length=4,
        )

    def test_uses_split_order_first_full_caption_and_complete_motion(self):
        motion_b = np.ones((64, 272), dtype=np.float32)
        motion_a = np.full((63, 272), 2.0, dtype=np.float32)
        self._write_fixture(
            split_ids=["motion_b", "motion_a"],
            motions={"motion_b": motion_b, "motion_a": motion_a},
            texts={
                "motion_b": [
                    "segment#tok#0.5#1.5",
                    "first full#tok#0#0",
                    "second full#tok#0#0",
                ],
                "motion_a": ["a full#tok#0#0"],
            },
        )

        dataset = self._dataset()

        self.assertEqual(dataset.sample_ids, ["motion_b", "motion_a"])
        self.assertEqual(
            dataset.sample_hash,
            hashlib.sha256(b"motion_b\nmotion_a").hexdigest(),
        )
        self.assertEqual(dataset[0]["caption"], "first full")
        self.assertEqual(dataset[0]["length"], 64)
        self.assertEqual(dataset[1]["length"], 60)
        self.assertTrue(torch.equal(dataset[1]["motion"], torch.full((60, 272), 2.0)))

    def test_filters_invalid_lengths_and_never_adds_tagged_subclips(self):
        self._write_fixture(
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

        dataset = self._dataset()

        self.assertEqual(dataset.sample_ids, ["valid"])
        self.assertEqual(len(dataset), 1)

    def test_nan_tags_are_treated_as_complete_motion_tags(self):
        self._write_fixture(
            split_ids=["nan_tag"],
            motions={"nan_tag": np.zeros((64, 272))},
            texts={"nan_tag": ["nan full#tok#nan#nan"]},
        )

        dataset = self._dataset()

        self.assertEqual(dataset[0]["caption"], "nan full")

    def test_collate_zero_pads_and_loader_preserves_order_and_tail(self):
        self._write_fixture(
            split_ids=["a", "b", "c"],
            motions={
                "a": np.ones((60, 272)),
                "b": np.ones((64, 272)),
                "c": np.ones((68, 272)),
            },
            texts={
                "a": ["a#tok#0#0"],
                "b": ["b#tok#0#0"],
                "c": ["c#tok#0#0"],
            },
        )
        dataset = self._dataset()

        batch = collate_msa_vae_metrics([dataset[0], dataset[1]])
        loader = make_msa_vae_metrics_loader(
            dataset,
            batch_size=2,
            num_workers=0,
        )
        loaded = list(loader)

        self.assertEqual(tuple(batch["motions"].shape), (2, 64, 272))
        self.assertEqual(batch["lengths"].tolist(), [60, 64])
        self.assertTrue(torch.equal(batch["motions"][0, 60:], torch.zeros(4, 272)))
        self.assertEqual(batch["sample_ids"], ["a", "b"])
        self.assertEqual([item for group in loaded for item in group["sample_ids"]], ["a", "b", "c"])
        self.assertEqual(len(loaded), 2)

    def test_rejects_empty_candidate_set_and_duplicate_split_ids(self):
        self._write_fixture(
            split_ids=["short"],
            motions={"short": np.zeros((20, 272))},
            texts={"short": ["short#tok#0#0"]},
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            self._dataset()

        self._write_fixture(
            split_ids=["valid", "valid"],
            motions={"valid": np.zeros((64, 272))},
            texts={"valid": ["valid#tok#0#0"]},
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            self._dataset()

    def test_rejects_motion_that_truncates_below_three_frames(self):
        self._write_fixture(
            split_ids=["truncated"],
            motions={"truncated": np.zeros((60, 272))},
            texts={"truncated": ["full motion#tok#0#0"]},
        )

        with self.assertRaisesRegex(
            ValueError,
            "truncated.*fewer than 3 frames",
        ):
            MSAVAEMetricsDataset(
                data_root=self.root,
                split_file=self.split_file,
                unit_length=64,
            )

    def test_rejects_non_positive_standard_deviation(self):
        std = np.ones(272)
        std[17] = -1.0
        np.save(self.root / "mean_std" / "Std.npy", std)
        self._write_fixture(
            split_ids=["valid"],
            motions={"valid": np.zeros((64, 272))},
            texts={"valid": ["full motion#tok#0#0"]},
        )

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self._dataset()


if __name__ == "__main__":
    unittest.main()
