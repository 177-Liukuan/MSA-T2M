import unittest

import torch

from utils.msa_vae_training import (
    MSAVAELossWeights,
    compute_msa_vae_objective,
    is_window_replay_step,
    latent_lengths_from_frames,
    masked_cosine_alignment,
    masked_kl,
    masked_mse,
    masked_optimal_sigma_nll,
    validate_sequence_training_config,
    valid_mask_from_lengths,
)


class MaskHelperTest(unittest.TestCase):
    def test_latent_lengths_use_repeated_floor_division(self):
        lengths = torch.tensor([64, 68, 70])

        latent = latent_lengths_from_frames(
            lengths,
            stride_t=2,
            down_t=2,
        )

        torch.testing.assert_close(latent, torch.tensor([16, 17, 17]))

    def test_valid_mask_marks_only_real_timesteps(self):
        mask = valid_mask_from_lengths(torch.tensor([2, 3]), 4)

        torch.testing.assert_close(
            mask,
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, False],
                ]
            ),
        )

    def test_invalid_length_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stride_t"):
            latent_lengths_from_frames(
                torch.tensor([4]),
                stride_t=0,
                down_t=2,
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            valid_mask_from_lengths(torch.tensor([-1]), 3)


class MaskedDenseLossTest(unittest.TestCase):
    def setUp(self):
        self.mask = valid_mask_from_lengths(torch.tensor([2, 3]), 4)
        self.target = torch.zeros(2, 4, 10)
        self.pred = torch.zeros(2, 4, 10)
        self.pred[0, :2] = 1.0
        self.pred[1, :3] = 2.0

    def test_mse_and_kl_ignore_padding_and_average_each_sample(self):
        padded_pred = self.pred.clone()
        padded_pred[~self.mask] = 1000.0
        logvar = torch.zeros_like(self.pred)
        padded_logvar = logvar.clone()
        padded_logvar[~self.mask] = 20.0

        mse = masked_mse(self.pred, self.target, self.mask)
        padded_mse = masked_mse(padded_pred, self.target, self.mask)
        kl = masked_kl(self.pred, logvar, self.mask)
        padded_kl = masked_kl(padded_pred, padded_logvar, self.mask)

        torch.testing.assert_close(mse, torch.tensor(2.5))
        torch.testing.assert_close(padded_mse, mse)
        torch.testing.assert_close(kl, torch.tensor(1.25))
        torch.testing.assert_close(padded_kl, kl)

    def test_optimal_sigma_losses_ignore_padding(self):
        padded_pred = self.pred.clone()
        padded_target = self.target.clone()
        padded_pred[~self.mask] = 1000.0
        padded_target[~self.mask] = -1000.0

        reconstruction = masked_optimal_sigma_nll(
            self.pred,
            self.target,
            self.mask,
            feature_slice=slice(None),
        )
        padded_reconstruction = masked_optimal_sigma_nll(
            padded_pred,
            padded_target,
            self.mask,
            feature_slice=slice(None),
        )
        root = masked_optimal_sigma_nll(
            self.pred,
            self.target,
            self.mask,
            feature_slice=slice(0, 8),
        )
        padded_root = masked_optimal_sigma_nll(
            padded_pred,
            padded_target,
            self.mask,
            feature_slice=slice(0, 8),
        )

        self.assertTrue(torch.isfinite(reconstruction))
        torch.testing.assert_close(padded_reconstruction, reconstruction)
        torch.testing.assert_close(padded_root, root)

    def test_duplicating_identical_valid_frames_does_not_double_loss(self):
        short_pred = torch.ones(1, 2, 3)
        short_target = torch.zeros_like(short_pred)
        short_mask = torch.ones(1, 2, dtype=torch.bool)
        long_pred = short_pred.repeat(1, 2, 1)
        long_target = short_target.repeat(1, 2, 1)
        long_mask = torch.ones(1, 4, dtype=torch.bool)

        torch.testing.assert_close(
            masked_mse(short_pred, short_target, short_mask),
            masked_mse(long_pred, long_target, long_mask),
        )
        torch.testing.assert_close(
            masked_kl(short_pred, torch.zeros_like(short_pred), short_mask),
            masked_kl(long_pred, torch.zeros_like(long_pred), long_mask),
        )
        torch.testing.assert_close(
            masked_optimal_sigma_nll(
                short_pred,
                short_target,
                short_mask,
                feature_slice=slice(None),
            ),
            masked_optimal_sigma_nll(
                long_pred,
                long_target,
                long_mask,
                feature_slice=slice(None),
            ),
        )


class MaskedAlignmentTest(unittest.TestCase):
    def test_sample_alignment_uses_only_valid_samples(self):
        pred = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

        loss = masked_cosine_alignment(
            pred,
            target,
            torch.tensor([True, False]),
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_token_alignment_ignores_padded_targets(self):
        pred = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
        )
        target = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
        )

        loss = masked_cosine_alignment(
            pred,
            target,
            torch.tensor([[True, True, False]]),
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_empty_alignment_mask_returns_differentiable_zero(self):
        pred = torch.randn(2, 3, requires_grad=True)
        target = torch.randn(2, 3)

        loss = masked_cosine_alignment(
            pred,
            target,
            torch.tensor([False, False]),
        )
        loss.backward()

        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


class ObjectiveCompositionTest(unittest.TestCase):
    def _fixture(self):
        outputs = {
            "x_recon": torch.ones(2, 8, 10, requires_grad=True),
            "mu": torch.ones(2, 2, 3, requires_grad=True),
            "logvar": torch.zeros(2, 2, 3, requires_grad=True),
            "mu_recon": torch.zeros(2, 2, 3, requires_grad=True),
            "clip_global_feat": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]], requires_grad=True
            ),
            "clip_local_feat": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                ],
                requires_grad=True,
            ),
        }
        targets = {
            "motion": torch.zeros(2, 8, 10),
            "motion_lengths": torch.tensor([8, 4]),
            "global_text": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "has_global": torch.tensor([True, True]),
            "local_text": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[-1.0, 0.0], [-1.0, 0.0]],
                ]
            ),
            "has_local": torch.tensor([True, False]),
        }
        weights = MSAVAELossWeights(
            root=7.0,
            latent=1.0,
            global_align=0.5,
            local_align=0.2,
        )
        return outputs, targets, weights

    def test_phase_and_batch_kind_select_expected_losses(self):
        outputs, targets, weights = self._fixture()

        phase1_total, phase1 = compute_msa_vae_objective(
            outputs,
            targets,
            phase=1,
            batch_kind="full",
            weights=weights,
            stride_t=2,
            down_t=2,
        )
        full_total, full = compute_msa_vae_objective(
            outputs,
            targets,
            phase=2,
            batch_kind="full",
            weights=weights,
            stride_t=2,
            down_t=2,
        )
        replay_total, replay = compute_msa_vae_objective(
            outputs,
            targets,
            phase=2,
            batch_kind="window",
            weights=weights,
            stride_t=2,
            down_t=2,
        )

        self.assertEqual(
            set(phase1),
            {"latent", "global_align", "local_align"},
        )
        self.assertEqual(
            set(full),
            {
                "recon",
                "kl",
                "root",
                "latent",
                "global_align",
                "local_align",
            },
        )
        self.assertEqual(
            set(replay),
            {"recon", "kl", "root", "local_align"},
        )
        self.assertGreater(full_total.item(), phase1_total.item())
        self.assertGreater(replay_total.item(), 0.0)

    def test_replay_marks_semantic_outputs_with_zero_gradients(self):
        outputs, targets, weights = self._fixture()

        total, _ = compute_msa_vae_objective(
            outputs,
            targets,
            phase=2,
            batch_kind="window",
            weights=weights,
            stride_t=2,
            down_t=2,
        )
        total.backward()

        torch.testing.assert_close(
            outputs["mu_recon"].grad,
            torch.zeros_like(outputs["mu_recon"]),
        )
        torch.testing.assert_close(
            outputs["clip_global_feat"].grad,
            torch.zeros_like(outputs["clip_global_feat"]),
        )

    def test_replay_schedule_is_deterministic(self):
        replay_steps = [
            step
            for step in range(1, 9)
            if is_window_replay_step(step, interval=4)
        ]

        self.assertEqual(replay_steps, [4, 8])

    def test_invalid_training_modes_are_rejected(self):
        validate_sequence_training_config(
            phase=1,
            mode="full",
            full_batch_size=8,
            replay_interval=4,
        )
        validate_sequence_training_config(
            phase=2,
            mode="mixed",
            full_batch_size=8,
            replay_interval=4,
        )
        with self.assertRaisesRegex(ValueError, "Phase 1"):
            validate_sequence_training_config(
                phase=1,
                mode="window",
                full_batch_size=8,
                replay_interval=4,
            )
        with self.assertRaisesRegex(ValueError, "Phase 2"):
            validate_sequence_training_config(
                phase=0,
                mode="mixed",
                full_batch_size=8,
                replay_interval=4,
            )
        with self.assertRaisesRegex(ValueError, "full_batch_size"):
            validate_sequence_training_config(
                phase=2,
                mode="full",
                full_batch_size=0,
                replay_interval=4,
            )
        with self.assertRaisesRegex(ValueError, "replay_interval"):
            validate_sequence_training_config(
                phase=2,
                mode="mixed",
                full_batch_size=8,
                replay_interval=1,
            )


if __name__ == "__main__":
    unittest.main()
