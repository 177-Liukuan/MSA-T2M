"""Unit tests for sparse MSA-VAE semantic-alignment reductions."""

import unittest

import torch
import torch.nn.functional as F

from utils.msa_vae_alignment import (
    distributed_masked_cosine_alignment,
    masked_cosine_sum_and_count,
)


class FakeAccelerator:
    """Minimal reducer with a fixed all-rank sum for deterministic tests."""

    num_processes = 2

    def __init__(self, global_count, global_sum):
        self.global_count = torch.tensor(global_count, dtype=torch.long)
        self.global_sum = torch.tensor(float(global_sum))

    def reduce(self, value, reduction="sum"):
        self.assert_reduction(reduction)
        if value.ndim == 0 and value.dtype == torch.long:
            return self.global_count.to(value.device)
        return self.global_sum.to(value.device, dtype=value.dtype)

    @staticmethod
    def assert_reduction(reduction):
        if reduction != "sum":
            raise AssertionError("alignment reductions must sum across ranks")


class MaskedCosineAlignmentTest(unittest.TestCase):
    def test_sample_mask_matches_valid_vector_mean_and_ignores_invalid_gradients(self):
        feat_a = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], requires_grad=True
        )
        feat_b = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 2.0]], requires_grad=True
        )
        mask = torch.tensor([True, False, True])

        loss_sum, valid_count = masked_cosine_sum_and_count(feat_a, feat_b, mask)
        expected = (1.0 - F.cosine_similarity(feat_a[mask], feat_b[mask], dim=-1)).mean()

        self.assertEqual(valid_count.item(), 2)
        self.assertTrue(torch.allclose(loss_sum / valid_count, expected))
        loss_sum.backward()
        self.assertTrue(torch.equal(feat_a.grad[1], torch.zeros_like(feat_a.grad[1])))
        self.assertTrue(torch.equal(feat_b.grad[1], torch.zeros_like(feat_b.grad[1])))

    def test_sample_mask_expands_to_all_tokens(self):
        feat_a = torch.tensor(
            [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, -1.0]]],
            requires_grad=True,
        )
        feat_b = torch.tensor(
            [[[0.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, -1.0]]],
            requires_grad=True,
        )
        mask = torch.tensor([False, True])

        loss_sum, valid_count = masked_cosine_sum_and_count(feat_a, feat_b, mask)
        expected = (1.0 - F.cosine_similarity(feat_a[mask], feat_b[mask], dim=-1)).mean()

        self.assertEqual(valid_count.item(), 2)
        self.assertTrue(torch.allclose(loss_sum / valid_count, expected))
        loss_sum.backward()
        self.assertTrue(torch.equal(feat_a.grad[0], torch.zeros_like(feat_a.grad[0])))
        self.assertTrue(torch.equal(feat_b.grad[0], torch.zeros_like(feat_b.grad[0])))

    def test_empty_mask_is_differentiable_zero(self):
        feat_a = torch.randn(2, 3, requires_grad=True)
        feat_b = torch.randn(2, 3, requires_grad=True)

        loss_sum, valid_count = masked_cosine_sum_and_count(
            feat_a, feat_b, torch.tensor([False, False])
        )

        self.assertEqual(valid_count.item(), 0)
        self.assertEqual(loss_sum.item(), 0.0)
        loss_sum.backward()
        self.assertTrue(torch.equal(feat_a.grad, torch.zeros_like(feat_a)))
        self.assertIsNone(feat_b.grad)

    def test_ddp_average_recovers_global_mean_gradient_when_one_rank_is_empty(self):
        rank_zero_a = torch.tensor([[1.0, 0.0]], requires_grad=True)
        rank_zero_b = torch.tensor([[0.0, 1.0]], requires_grad=True)
        rank_one_a = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
        rank_one_b = torch.tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
        mask_zero = torch.tensor([False])
        mask_one = torch.tensor([True, True])

        global_sum = (1.0 - F.cosine_similarity(rank_one_a, rank_one_b)).sum().item()
        accelerator = FakeAccelerator(global_count=2, global_sum=global_sum)
        rank_zero = distributed_masked_cosine_alignment(
            rank_zero_a, rank_zero_b, mask_zero, accelerator
        )
        rank_one = distributed_masked_cosine_alignment(
            rank_one_a, rank_one_b, mask_one, accelerator
        )
        self.assertEqual(rank_zero.valid_count.item(), 2)
        self.assertEqual(rank_one.valid_count.item(), 2)
        self.assertTrue(torch.allclose(rank_one.global_mean, torch.tensor(global_sum / 2)))

        rank_zero.backward_loss.backward()
        rank_one.backward_loss.backward()

        reference_a = rank_one_a.detach().clone().requires_grad_(True)
        reference_b = rank_one_b.detach().clone().requires_grad_(True)
        reference = (1.0 - F.cosine_similarity(reference_a, reference_b)).mean()
        reference.backward()

        self.assertTrue(torch.allclose(rank_one_a.grad / 2, reference_a.grad))
        self.assertTrue(torch.allclose(rank_one_b.grad / 2, reference_b.grad))
        self.assertTrue(torch.equal(rank_zero_a.grad, torch.zeros_like(rank_zero_a)))
        self.assertIsNone(rank_zero_b.grad)


if __name__ == "__main__":
    unittest.main()
