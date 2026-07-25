import unittest

import torch
import torch.nn as nn

from models.causal_cnn import CausalEncoder
from models.msa_vae import MSA_HumanVAE


class CausalEncoderStatsTest(unittest.TestCase):
    def test_encode_stats_matches_forward_distribution_parameters(self):
        encoder = CausalEncoder(
            input_emb_width=6,
            hidden_size=8,
            down_t=1,
            stride_t=2,
            width=8,
            depth=1,
            dilation_growth_rate=2,
            latent_dim=4,
            clip_range=[-30, 20],
        )
        encoder.eval()
        motion = torch.randn(2, 6, 12)

        expected_mu, expected_logvar = encoder.encode_stats(motion)
        _, actual_mu, actual_logvar = encoder(motion)

        torch.testing.assert_close(actual_mu, expected_mu)
        torch.testing.assert_close(actual_logvar, expected_logvar)


class SemanticOnlyForwardTest(unittest.TestCase):
    @staticmethod
    def _model(disable_decoupling=False):
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
            disable_decoupling=disable_decoupling,
        )

    def test_semantic_only_skips_cnn_decoder_and_masks_padded_latents(self):
        class RaisingDecoder(nn.Module):
            def forward(self, _):
                raise AssertionError("CNN decoder must not run")

        model = self._model()
        model.msa_vae.cnn_decoder = RaisingDecoder()
        model.eval()

        motion = torch.randn(2, 68, 272)
        changed_padding = motion.clone()
        changed_padding[0, 64:] = 1000.0
        lengths = torch.tensor([64, 68])

        first = model(motion, lengths=lengths, semantic_only=True)
        second = model(
            changed_padding,
            lengths=lengths,
            semantic_only=True,
        )

        self.assertEqual(
            set(first),
            {
                "mu",
                "logvar",
                "h_cls",
                "mu_recon",
                "trans_latent_target",
                "clip_global_feat",
                "clip_local_feat",
            },
        )
        self.assertEqual(first["mu"].shape, (2, 17, 4))
        self.assertEqual(first["mu_recon"].shape, (2, 17, 4))
        torch.testing.assert_close(
            first["h_cls"][0],
            second["h_cls"][0],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_semantic_only_preserves_sampled_target_ablation(self):
        class RaisingDecoder(nn.Module):
            def forward(self, _):
                raise AssertionError("CNN decoder must not run")

        model = self._model(disable_decoupling=True)
        model.msa_vae.cnn_decoder = RaisingDecoder()
        model.eval()
        torch.manual_seed(9)

        output = model(
            torch.randn(1, 64, 272),
            lengths=torch.tensor([64]),
            semantic_only=True,
        )

        self.assertFalse(
            torch.allclose(
                output["trans_latent_target"],
                output["mu"],
            )
        )

    def test_legacy_forward_and_state_dict_remain_compatible(self):
        model = self._model()
        clone = self._model()
        keys = set(model.state_dict())

        self.assertFalse(any("encode_stats" in key for key in keys))
        self.assertFalse(any("semantic" in key for key in keys))
        clone.load_state_dict(model.state_dict(), strict=True)

        output = clone(torch.randn(1, 64, 272))

        self.assertIn("x_recon", output)
        self.assertIn("z_local", output)
        self.assertEqual(output["x_recon"].shape, (1, 64, 272))


if __name__ == "__main__":
    unittest.main()
