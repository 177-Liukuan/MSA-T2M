import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from unittest import mock

from humanml3d_272.dataset_eval_msa_vae_metrics import (
    collate_msa_vae_metrics,
)
from utils.msa_vae_validation import (
    MSAValidationState,
    isolated_validation_rng,
    publish_msa_validation,
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


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, iteration):
        self.scalars.append((tag, value, iteration))


def _validation_result(**overrides):
    result = {
        "sample_count": 802,
        "fid": 0.95,
        "mpjpe_mm": 21.6,
        "p_mpjpe_mm": 15.9,
        "accel_mm_per_frame2": 4.6,
        "skating_percent": 8.0,
        "t2m_r1_percent": 13.0,
        "t2m_r2_percent": 26.0,
        "t2m_r3_percent": 34.0,
        "t2m_r5_percent": 47.0,
        "t2m_medr": 6.0,
        "m2t_r1_percent": 19.0,
        "m2t_r2_percent": 27.0,
        "m2t_r3_percent": 38.0,
        "m2t_r5_percent": 48.0,
        "m2t_medr": 6.0,
    }
    result.update(overrides)
    return result


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


class ValidationCheckpointPublicationTest(unittest.TestCase):
    @mock.patch("utils.eval_trans.tensorborad_add_video_xyz")
    def test_both_phases_keep_best_and_last_without_rendering(self, render):
        for phase in (1, 2):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp:
                output = Path(temp)
                model = torch.nn.Linear(3, 2)
                metadata = {"format_version": 1, "phase": phase}
                logger = _Logger()
                writer = _Writer()

                state = publish_msa_validation(
                    result=_validation_result(),
                    iteration=0,
                    out_dir=output,
                    model=model,
                    metadata=metadata,
                    state=MSAValidationState(),
                    logger=logger,
                    writer=writer,
                    validation_seed=123,
                    validation_batch_size=32,
                )

                paths = {
                    name: output / name
                    for name in (
                        "net_best_fid.pth",
                        "net_best_mpjpe.pth",
                        "net_last.pth",
                    )
                }
                self.assertTrue(all(path.is_file() for path in paths.values()))
                for path in paths.values():
                    payload = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    self.assertEqual(payload["metadata"], metadata)
                first_hashes = {
                    name: _file_sha256(path)
                    for name, path in paths.items()
                }

                with torch.no_grad():
                    model.weight.add_(1.0)
                state = publish_msa_validation(
                    result=_validation_result(fid=1.1, mpjpe_mm=22.0),
                    iteration=1,
                    out_dir=output,
                    model=model,
                    metadata=metadata,
                    state=state,
                    logger=logger,
                    writer=writer,
                    validation_seed=123,
                    validation_batch_size=32,
                )
                second_hashes = {
                    name: _file_sha256(path)
                    for name, path in paths.items()
                }
                self.assertEqual(
                    second_hashes["net_best_fid.pth"],
                    first_hashes["net_best_fid.pth"],
                )
                self.assertEqual(
                    second_hashes["net_best_mpjpe.pth"],
                    first_hashes["net_best_mpjpe.pth"],
                )
                self.assertNotEqual(
                    second_hashes["net_last.pth"],
                    first_hashes["net_last.pth"],
                )

                with torch.no_grad():
                    model.weight.add_(1.0)
                final_state = publish_msa_validation(
                    result=_validation_result(fid=0.9, mpjpe_mm=22.0),
                    iteration=2,
                    out_dir=output,
                    model=model,
                    metadata=metadata,
                    state=state,
                    logger=logger,
                    writer=writer,
                    validation_seed=123,
                    validation_batch_size=32,
                )
                self.assertNotEqual(
                    _file_sha256(paths["net_best_fid.pth"]),
                    second_hashes["net_best_fid.pth"],
                )
                self.assertEqual(
                    _file_sha256(paths["net_best_mpjpe.pth"]),
                    second_hashes["net_best_mpjpe.pth"],
                )
                self.assertEqual(final_state.best_fid, 0.9)
                self.assertEqual(final_state.best_mpjpe, 21.6)
                self.assertTrue(
                    all(
                        tag.startswith("CompleteVal/")
                        for tag, _, _ in writer.scalars
                    )
                )
                self.assertTrue(
                    any("seed=123 batch_size=32" in msg for msg in logger.messages)
                )
        render.assert_not_called()

    def test_invalid_result_writes_no_checkpoint(self):
        invalid_results = (
            _validation_result(sample_count=0),
            _validation_result(fid=float("nan")),
        )
        for result in invalid_results:
            with self.subTest(result=result), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, "validation"):
                    publish_msa_validation(
                        result=result,
                        iteration=0,
                        out_dir=temp,
                        model=torch.nn.Linear(3, 2),
                        metadata={"phase": 1},
                        state=MSAValidationState(),
                        logger=_Logger(),
                        writer=_Writer(),
                        validation_seed=123,
                        validation_batch_size=32,
                    )
                self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
