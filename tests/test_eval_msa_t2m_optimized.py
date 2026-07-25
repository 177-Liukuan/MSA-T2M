import unittest

import numpy as np
import torch

from utils.eval_msa_t2m_optimized import (
    OptimizedRAGEvalSampler,
    decode_equal_length_groups,
    generate_latents_active_set,
)


class DeterministicRAG(torch.nn.Module):
    """Return a token containing the sample id and current prefix length."""

    def sample_next_with_cfg(
        self,
        motion_prefix,
        text_emb,
        empty_text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_scale=4.0,
        temperature=1.0,
    ):
        del empty_text_emb, top3_h_cls, top3_sim_scores, cfg_scale, temperature
        step = motion_prefix.shape[1]
        return torch.stack(
            (text_emb[:, 0], torch.full_like(text_emb[:, 0], step)),
            dim=-1,
        )


class ScheduledEOSRAG(torch.nn.Module):
    """Emit a shared EOS vector at a sample-specific accepted-prefix length."""

    def __init__(self, eos_at):
        super().__init__()
        self.eos_at = dict(eos_at)

    def sample_next_with_cfg(
        self,
        motion_prefix,
        text_emb,
        empty_text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_scale=4.0,
        temperature=1.0,
    ):
        del empty_text_emb, top3_h_cls, top3_sim_scores, cfg_scale, temperature
        step = motion_prefix.shape[1]
        rows = []
        for sample_id in text_emb[:, 0].tolist():
            if self.eos_at.get(int(sample_id)) == step:
                rows.append(torch.tensor([99.0, 99.0], device=text_emb.device))
            else:
                rows.append(
                    torch.tensor(
                        [sample_id, float(step)],
                        device=text_emb.device,
                    )
                )
        return torch.stack(rows)


class OptimizedGenerationTests(unittest.TestCase):
    def test_active_set_matches_serial_without_eos(self):
        model = DeterministicRAG()
        text = torch.tensor([[10.0], [20.0], [30.0]])

        result = generate_latents_active_set(
            model,
            text,
            torch.zeros(1),
            None,
            None,
            torch.tensor([1, 3, 2]),
            latent_dim=2,
            reference_end_latent=None,
            stop_threshold=0.1,
            enable_stopping=False,
            cfg_scale=4.0,
        )

        self.assertEqual(
            [x.squeeze(0).tolist() for x in result.latents],
            [
                [[10.0, 0.0]],
                [[20.0, 0.0], [20.0, 1.0], [20.0, 2.0]],
                [[30.0, 0.0], [30.0, 1.0]],
            ],
        )
        self.assertEqual(result.stop_steps.tolist(), [-1, -1, -1])
        self.assertEqual(result.empty_fallback_count, 0)

    def test_eos_candidate_is_excluded_and_other_samples_continue(self):
        model = ScheduledEOSRAG({10: 1, 20: 2})
        text = torch.tensor([[10.0], [20.0], [30.0]])

        result = generate_latents_active_set(
            model,
            text,
            torch.zeros(1),
            None,
            None,
            torch.tensor([4, 4, 3]),
            latent_dim=2,
            reference_end_latent=torch.tensor([99.0, 99.0]),
            stop_threshold=0.1,
            enable_stopping=True,
            cfg_scale=4.0,
        )

        self.assertEqual(
            [x.squeeze(0).tolist() for x in result.latents],
            [
                [[10.0, 0.0]],
                [[20.0, 0.0], [20.0, 1.0]],
                [[30.0, 0.0], [30.0, 1.0], [30.0, 2.0]],
            ],
        )
        self.assertEqual(result.stop_steps.tolist(), [1, 2, -1])
        for latent in result.latents:
            self.assertFalse(torch.any(torch.all(latent == 99.0, dim=-1)))

    def test_first_candidate_eos_uses_single_zero_latent_fallback(self):
        result = generate_latents_active_set(
            ScheduledEOSRAG({10: 0}),
            torch.tensor([[10.0]]),
            torch.zeros(1),
            None,
            None,
            torch.tensor([4]),
            latent_dim=2,
            reference_end_latent=torch.tensor([99.0, 99.0]),
            stop_threshold=0.1,
            enable_stopping=True,
            cfg_scale=4.0,
        )

        self.assertEqual(result.latents[0].shape, (1, 1, 2))
        self.assertTrue(torch.equal(result.latents[0], torch.zeros(1, 1, 2)))
        self.assertEqual(result.stop_steps.tolist(), [0])
        self.assertEqual(result.empty_fallback_count, 1)

    def test_rejects_nonpositive_token_ceiling(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            generate_latents_active_set(
                DeterministicRAG(),
                torch.tensor([[10.0]]),
                torch.zeros(1),
                None,
                None,
                torch.tensor([0]),
                latent_dim=2,
                reference_end_latent=None,
                stop_threshold=0.1,
                enable_stopping=False,
                cfg_scale=4.0,
            )


class RecordingTextEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray(
            [[float(index + 1)] for index in range(len(texts))],
            dtype=np.float32,
        )


class RecordingRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, text_emb):
        self.calls += 1
        batch_size = text_emb.shape[0]
        return (
            text_emb.unsqueeze(1),
            torch.ones(batch_size, 1, device=text_emb.device),
        )


class OptimizedSamplerTests(unittest.TestCase):
    def test_sampler_encodes_and_retrieves_once_per_batch(self):
        text_encoder = RecordingTextEncoder()
        retriever = RecordingRetriever()
        sampler = OptimizedRAGEvalSampler(
            rag_model=DeterministicRAG(),
            retriever=retriever,
            empty_text_emb=torch.zeros(1),
            latent_dim=2,
            device=torch.device("cpu"),
            reference_end_latent=None,
            stop_threshold=0.1,
            enable_stopping=False,
            text_source="online_t5",
            text_lookup=None,
            text_encoder=text_encoder,
            text_embed_dim=1,
            disable_rag=False,
        )

        result = sampler.sample_batch_for_eval_CFG(
            ["one", "two", "three"],
            torch.tensor([4, 8, 12]),
            unit_length=4,
        )

        self.assertEqual(text_encoder.calls, [["one", "two", "three"]])
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(
            [latent.shape[1] for latent in result.latents],
            [1, 2, 3],
        )
        self.assertEqual(
            [latent[0, 0, 0].item() for latent in result.latents],
            [1.0, 2.0, 3.0],
        )


class RecordingDecoder(torch.nn.Module):
    def __init__(self, motion_dim):
        super().__init__()
        self.motion_dim = motion_dim
        self.calls = []

    def forward_decoder(self, latents):
        self.calls.append(tuple(latents.shape))
        frames = latents.shape[1] * 4
        sample_ids = latents[:, 0, 0].view(-1, 1, 1)
        return sample_ids.expand(-1, frames, self.motion_dim).clone()


class OptimizedDecodeTests(unittest.TestCase):
    def test_equal_length_groups_restore_original_order(self):
        decoder = RecordingDecoder(motion_dim=3)
        latent_sequences = [
            torch.full((1, 2, 2), 10.0),
            torch.full((1, 1, 2), 20.0),
            torch.full((1, 2, 2), 30.0),
            torch.full((1, 3, 2), 40.0),
            torch.full((1, 1, 2), 50.0),
        ]

        motions, lengths = decode_equal_length_groups(
            decoder,
            latent_sequences,
            max_motion_length=12,
            motion_dim=3,
        )

        self.assertEqual(
            decoder.calls,
            [(2, 2, 2), (2, 1, 2), (1, 3, 2)],
        )
        self.assertEqual(motions.shape, (5, 12, 3))
        self.assertEqual(lengths.tolist(), [8, 4, 8, 12, 4])
        self.assertEqual(motions[:, 0, 0].tolist(), [10, 20, 30, 40, 50])


if __name__ == "__main__":
    unittest.main()
