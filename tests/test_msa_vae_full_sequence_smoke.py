import unittest

import numpy as np
import torch

from humanml3d_272.dataset_msa_vae import collate_fn
from models.msa_vae import MSA_HumanVAE
from utils.msa_vae_training import (
    MSAVAELossWeights,
    compute_msa_vae_objective,
)


class FullSequenceTrainingSmokeTest(unittest.TestCase):
    @staticmethod
    def _model():
        return MSA_HumanVAE(
            hidden_size=16,
            down_t=2,
            stride_t=2,
            depth=1,
            dilation_growth_rate=2,
            latent_dim=4,
            clip_range=[-30, 20],
            trans_d_model=16,
            trans_nhead=4,
            trans_enc_layers=1,
            trans_dec_layers=1,
            trans_ff_size=32,
            trans_dropout=0.0,
            clip_dim=16,
        )

    @staticmethod
    def _item(length):
        return (
            np.random.randn(length, 272).astype(np.float32),
            f"motion-{length}",
            np.random.randn(16).astype(np.float32),
            True,
            np.random.randn(length // 4, 16).astype(np.float32),
            True,
            length,
            np.random.randn(16).astype(np.float32),
            length,
        )

    @staticmethod
    def _targets(batch):
        return {
            "motion": batch[0],
            "motion_lengths": batch[-1],
            "global_text": batch[2],
            "has_global": batch[3],
            "local_text": batch[4],
            "has_local": batch[5],
        }

    def test_phase1_full_phase2_full_and_replay_all_backpropagate(self):
        torch.manual_seed(11)
        np.random.seed(11)
        model = self._model()
        weights = MSAVAELossWeights(7.0, 1.0, 0.5, 0.2)

        full_batch = collate_fn([self._item(64), self._item(68)])
        phase1_output = model(
            full_batch[0],
            lengths=full_batch[-1],
            semantic_only=True,
        )
        phase1_loss, _ = compute_msa_vae_objective(
            phase1_output,
            self._targets(full_batch),
            phase=1,
            batch_kind="full",
            weights=weights,
            stride_t=2,
            down_t=2,
        )
        phase1_loss.backward()
        self.assertTrue(torch.isfinite(phase1_loss))

        model.zero_grad(set_to_none=True)
        phase2_output = model(full_batch[0], lengths=full_batch[-1])
        phase2_loss, _ = compute_msa_vae_objective(
            phase2_output,
            self._targets(full_batch),
            phase=2,
            batch_kind="full",
            weights=weights,
            stride_t=2,
            down_t=2,
        )
        phase2_loss.backward()
        self.assertTrue(torch.isfinite(phase2_loss))

        model.zero_grad(set_to_none=True)
        replay_batch = collate_fn([self._item(64), self._item(64)])
        replay_output = model(
            replay_batch[0],
            lengths=replay_batch[-1],
        )
        replay_loss, _ = compute_msa_vae_objective(
            replay_output,
            self._targets(replay_batch),
            phase=2,
            batch_kind="window",
            weights=weights,
            stride_t=2,
            down_t=2,
        )
        replay_loss.backward()
        self.assertTrue(torch.isfinite(replay_loss))


if __name__ == "__main__":
    unittest.main()
