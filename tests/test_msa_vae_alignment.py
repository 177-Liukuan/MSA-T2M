"""Unit tests for sparse MSA-VAE semantic-alignment reductions."""

import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from utils.msa_vae_alignment import (
    distributed_mask_coverage,
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


class _GlooAccelerator:
    num_processes = 2

    @staticmethod
    def reduce(value, reduction="sum"):
        if reduction != "sum":
            raise AssertionError("alignment reductions must sum across ranks")
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


class _CoverageAccelerator:
    num_processes = 2

    def __init__(self):
        self.values = iter((torch.tensor(6), torch.tensor(9)))

    def reduce(self, value, reduction="sum"):
        if reduction != "sum":
            raise AssertionError("coverage reductions must sum across ranks")
        return next(self.values).to(device=value.device, dtype=value.dtype)


def _run_wrapped_ddp_alignment_step(rank, world_size, init_file):
    dist.init_process_group(
        backend="gloo",
        init_method="file://{}".format(init_file),
        rank=rank,
        world_size=world_size,
    )
    try:
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.8, -0.3], [0.2, 0.5]]))
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        if rank == 0:
            inputs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            mask = torch.tensor([False, False])
        else:
            inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
            targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            mask = torch.tensor([True, False])

        # This is the production integration contract: the prepared wrapper,
        # not ``wrapped.module``, owns forward and backward synchronization.
        features = wrapped(inputs)
        result = distributed_masked_cosine_alignment(
            features, targets, mask, _GlooAccelerator()
        )
        optimizer.zero_grad()
        result.backward_loss.backward()
        gradient = model.weight.grad.detach().clone()
        optimizer.step()

        gradients = [torch.empty_like(gradient) for _ in range(world_size)]
        parameters = [torch.empty_like(model.weight) for _ in range(world_size)]
        dist.all_gather(gradients, gradient)
        dist.all_gather(parameters, model.weight.detach())
        for other in gradients[1:]:
            torch.testing.assert_close(other, gradients[0])
        for other in parameters[1:]:
            torch.testing.assert_close(other, parameters[0])
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise AssertionError("wrapped unequal-mask regression produced no useful gradient")
    finally:
        dist.destroy_process_group()


class MaskedCosineAlignmentTest(unittest.TestCase):
    def test_coverage_is_reported_independently_of_alignment_loss(self):
        valid_count, valid_ratio = distributed_mask_coverage(
            torch.tensor([True, False, True]),
            tokens_per_sample=3,
            accelerator=_CoverageAccelerator(),
        )

        self.assertEqual(valid_count.item(), 6)
        self.assertTrue(torch.allclose(valid_ratio, torch.tensor(2.0 / 3.0)))

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

    def test_real_two_process_gloo_keeps_gradients_and_parameters_synchronized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = str(Path(temporary_directory) / "gloo-init")
            mp.spawn(
                _run_wrapped_ddp_alignment_step,
                args=(2, init_file),
                nprocs=2,
                join=True,
            )

    def test_training_loops_forward_through_accelerator_prepared_net(self):
        source = (
            Path(__file__).resolve().parents[1] / "train_msa_vae.py"
        ).read_text()
        self.assertEqual(source.count("compute_losses(batch, net)"), 2)
        self.assertEqual(
            source.count(
                "total_loss, loss_dict = compute_humanml_losses("
                "batch, batch_kind)"
            ),
            2,
        )
        self.assertIn(
            "if args.msa_data_mode == 'humanml_full':\n"
            "    validate_sequence_training_config(",
            source,
        )
        self.assertNotIn("net.module if args.num_gpus > 1 else net", source)


if __name__ == "__main__":
    unittest.main()
