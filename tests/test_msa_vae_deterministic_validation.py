import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from unittest import mock

from humanml3d_272.dataset_eval_msa_vae_metrics import (
    collate_msa_vae_metrics,
)
from utils.msa_vae_validation import (
    isolated_validation_rng,
    run_deterministic_msa_validation,
)


def _numpy_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class _Distribution:
    def __init__(self, loc):
        self.loc = loc


class _ValidationDataset(Dataset):
    def __init__(self):
        self.items = []
        generator = torch.Generator().manual_seed(2026)
        for index, length in enumerate((4, 5, 6)):
            motion = torch.zeros(length, 272)
            motion[:, :66] = torch.randn(
                length,
                66,
                generator=generator,
            )
            motion[:, 100] = float(index)
            self.items.append(
                {
                    "sample_id": f"sample-{index}",
                    "caption": f"sample-{index}",
                    "motion": motion,
                    "length": length,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @staticmethod
    def inv_transform(array):
        return np.asarray(array)


class _StochasticModel(torch.nn.Module):
    def forward(self, motions, lengths=None):
        noise = torch.randn_like(motions) * 1e-4
        return {"x_recon": motions + noise}


class _TextEncoder(torch.nn.Module):
    def forward(self, captions):
        labels = torch.tensor(
            [int(caption.rsplit("-", 1)[1]) for caption in captions]
        )
        return _Distribution(F.one_hot(labels, num_classes=3).float())


class _MotionEncoder(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail

    def forward(self, motions, lengths):
        if self.fail:
            raise RuntimeError("evaluator failure")
        labels = motions[:, 0, 100].round().long()
        return _Distribution(F.one_hot(labels, num_classes=3).float())


def _recover_fixture(features, joint_count):
    if joint_count != 22:
        raise AssertionError("expected 22 joints")
    return np.asarray(features)[:, :66].reshape(-1, 22, 3)


class IsolatedValidationRNGTest(unittest.TestCase):
    def _assert_restored(self, raise_inside):
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark
        try:
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(14)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            cuda_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )

            def exercise():
                with isolated_validation_rng(123):
                    random.random()
                    np.random.random()
                    torch.rand(4)
                    if torch.cuda.is_available():
                        torch.rand(4, device="cuda")
                    self.assertTrue(torch.backends.cudnn.deterministic)
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    if raise_inside:
                        raise RuntimeError("diagnostic")

            if raise_inside:
                with self.assertRaisesRegex(RuntimeError, "diagnostic"):
                    exercise()
            else:
                exercise()

            self.assertEqual(random.getstate(), python_state)
            self.assertTrue(
                _numpy_states_equal(np.random.get_state(), numpy_state)
            )
            torch.testing.assert_close(
                torch.random.get_rng_state(),
                torch_state,
            )
            if cuda_states is not None:
                restored_cuda_states = torch.cuda.get_rng_state_all()
                self.assertEqual(
                    len(restored_cuda_states),
                    len(cuda_states),
                )
                for restored, expected in zip(
                    restored_cuda_states,
                    cuda_states,
                ):
                    torch.testing.assert_close(restored, expected)
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.deterministic = original_deterministic
            torch.backends.cudnn.benchmark = original_benchmark

    def test_restores_rng_and_backend_states_after_success(self):
        self._assert_restored(raise_inside=False)

    def test_restores_rng_and_backend_states_after_exception(self):
        self._assert_restored(raise_inside=True)


class DeterministicValidationWrapperTest(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(
            _ValidationDataset(),
            batch_size=2,
            shuffle=False,
            collate_fn=collate_msa_vae_metrics,
        )

    @mock.patch(
        "eval_msa_vae_metrics.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_repeated_calls_are_identical_and_restore_training_mode(
        self,
        _recover,
    ):
        model = _StochasticModel()
        model.train()
        evaluator = [_TextEncoder(), _MotionEncoder()]

        first = run_deterministic_msa_validation(
            model,
            evaluator,
            self.loader,
            torch.device("cpu"),
            seed=123,
        )
        second = run_deterministic_msa_validation(
            model,
            evaluator,
            self.loader,
            torch.device("cpu"),
            seed=123,
        )

        self.assertEqual(first, second)
        self.assertTrue(model.training)

    @mock.patch(
        "eval_msa_vae_metrics.recover_from_local_position",
        side_effect=_recover_fixture,
    )
    def test_evaluator_failure_restores_prior_eval_mode(self, _recover):
        model = _StochasticModel()
        model.eval()

        with self.assertRaisesRegex(RuntimeError, "evaluator failure"):
            run_deterministic_msa_validation(
                model,
                [_TextEncoder(), _MotionEncoder(fail=True)],
                self.loader,
                torch.device("cpu"),
                seed=123,
            )

        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
