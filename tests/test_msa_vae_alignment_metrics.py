import unittest

import torch

from utils.msa_vae_alignment_metrics import (
    calculate_masked_local_cosine,
    calculate_motion_macro_cosine,
    calculate_msa_t5_retrieval,
    shuffled_text_control,
)


class GlobalAlignmentMetricsTest(unittest.TestCase):
    def test_motion_macro_cosine_does_not_overweight_extra_captions(self):
        motion = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
            dtype=torch.float32,
        )
        owners = torch.tensor([0, 0, 1])

        value = calculate_motion_macro_cosine(motion, text, owners)

        self.assertAlmostEqual(value, 0.0)

    def test_multi_positive_m2t_accepts_either_caption(self):
        motion = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text = torch.tensor(
            [[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        owners = torch.tensor([0, 0, 1])

        metrics = calculate_msa_t5_retrieval(text, motion, owners)

        self.assertEqual(metrics["msa_t5_t2m_r1_percent"], 100.0)
        self.assertEqual(metrics["msa_t5_m2t_r1_percent"], 100.0)
        self.assertEqual(metrics["msa_t5_m2t_medr"], 1.0)

    def test_m2t_ignores_other_positive_captions_when_ranking(self):
        motion = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        owners = torch.tensor([0, 0, 1])

        metrics = calculate_msa_t5_retrieval(text, motion, owners)

        self.assertEqual(metrics["msa_t5_m2t_r1_percent"], 100.0)
        self.assertEqual(metrics["msa_t5_m2t_medr"], 1.0)

    def test_retrieval_uses_average_rank_for_positive_negative_ties(self):
        motion = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        text = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        owners = torch.tensor([0, 1])

        metrics = calculate_msa_t5_retrieval(text, motion, owners)

        self.assertEqual(metrics["msa_t5_t2m_r1_percent"], 100.0)
        self.assertEqual(metrics["msa_t5_m2t_r1_percent"], 100.0)
        self.assertEqual(metrics["msa_t5_t2m_medr"], 1.5)
        self.assertEqual(metrics["msa_t5_m2t_medr"], 1.5)

    def test_retrieval_reports_all_requested_recall_cutoffs(self):
        motion = torch.eye(6)
        text = torch.eye(6)
        owners = torch.arange(6)

        metrics = calculate_msa_t5_retrieval(text, motion, owners)

        for direction in ("t2m", "m2t"):
            for cutoff in (1, 2, 3, 5):
                self.assertEqual(
                    metrics[
                        "msa_t5_{}_r{}_percent".format(direction, cutoff)
                    ],
                    100.0,
                )

    def test_global_metrics_reject_invalid_owners_and_missing_captions(self):
        motion = torch.eye(2)
        text = torch.eye(2)

        with self.assertRaisesRegex(ValueError, "owner"):
            calculate_motion_macro_cosine(
                motion,
                text,
                torch.tensor([0, 2]),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            calculate_msa_t5_retrieval(
                text[:1],
                motion,
                torch.tensor([0]),
            )

    def test_global_metrics_reject_bad_shapes_norms_and_values(self):
        motion = torch.eye(2)
        text = torch.eye(2)
        owners = torch.tensor([0, 1])

        with self.assertRaisesRegex(ValueError, "dimension"):
            calculate_motion_macro_cosine(
                motion,
                torch.ones(2, 3),
                owners,
            )
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            calculate_msa_t5_retrieval(
                torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
                motion,
                owners,
            )
        invalid = text.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            calculate_motion_macro_cosine(motion, invalid, owners)


class LocalAlignmentMetricsTest(unittest.TestCase):
    def test_local_cosine_is_motion_macro_not_token_micro(self):
        prediction = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            ]
        )
        target = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        mask = torch.tensor([[True, True, True], [True, False, False]])

        value = calculate_masked_local_cosine(prediction, target, mask)

        self.assertAlmostEqual(value, 0.0)

    def test_zero_padding_outside_mask_is_allowed(self):
        prediction = torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        target = prediction.clone()
        mask = torch.tensor([[True, False]])

        value = calculate_masked_local_cosine(prediction, target, mask)

        self.assertAlmostEqual(value, 1.0)

    def test_local_cosine_rejects_invalid_inputs(self):
        prediction = torch.ones(2, 3, 2)
        target = torch.ones(2, 3, 2)
        mask = torch.ones(2, 3, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "shape"):
            calculate_masked_local_cosine(
                prediction,
                target[:, :2],
                mask,
            )
        with self.assertRaisesRegex(ValueError, "mask"):
            calculate_masked_local_cosine(
                prediction,
                target,
                torch.ones(2, 2, dtype=torch.bool),
            )
        empty_mask = mask.clone()
        empty_mask[1] = False
        with self.assertRaisesRegex(ValueError, "at least one"):
            calculate_masked_local_cosine(
                prediction,
                target,
                empty_mask,
            )
        zero_norm = target.clone()
        zero_norm[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            calculate_masked_local_cosine(
                prediction,
                zero_norm,
                mask,
            )
        invalid_padding = target.clone()
        invalid_padding[0, 2, 0] = float("inf")
        padding_mask = mask.clone()
        padding_mask[0, 2] = False
        with self.assertRaisesRegex(ValueError, "non-finite"):
            calculate_masked_local_cosine(
                prediction,
                invalid_padding,
                padding_mask,
            )


class ShuffledControlTest(unittest.TestCase):
    def test_is_deterministic_derangement_and_preserves_rows(self):
        text = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        first = shuffled_text_control(text, seed=123)
        second = shuffled_text_control(text, seed=123)

        torch.testing.assert_close(first, second)
        self.assertTrue(torch.all(torch.any(first != text, dim=1)))
        self.assertEqual(
            sorted(map(tuple, first.tolist())),
            sorted(map(tuple, text.tolist())),
        )
        self.assertTrue(torch.equal(text, torch.arange(12).reshape(4, 3)))

    def test_rejects_too_few_rows_or_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            shuffled_text_control(torch.ones(1, 2), seed=123)
        invalid = torch.eye(2)
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            shuffled_text_control(invalid, seed=123)


if __name__ == "__main__":
    unittest.main()
