import unittest

import numpy as np
import torch

from eval_msa_t2m_rag_t5 import RAGEvalSampler
from msa_gen_motion import sample_motion_latents_with_stop


MOTION_TOKEN = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
EOS_TOKEN = torch.tensor([[9.0, 9.0]], dtype=torch.float32)
EXPECTED_MOTION_LATENTS = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)


class SequenceRAGModel:
    def __init__(self):
        self.tokens = [MOTION_TOKEN.clone(), EOS_TOKEN.clone()]

    def sample_next_with_cfg(self, **_kwargs):
        return self.tokens.pop(0)


class TextLookup:
    @staticmethod
    def batch_lookup(_texts, device):
        return torch.zeros((1, 2), dtype=torch.float32, device=device)


class TextEncoder:
    @staticmethod
    def encode(_texts):
        return np.zeros((1, 2), dtype=np.float32)


class MSAEOSStoppingTest(unittest.TestCase):
    def test_eval_sampler_excludes_generated_eos_from_motion_latents(self):
        sampler = RAGEvalSampler(
            rag_model=SequenceRAGModel(),
            retriever=None,
            empty_text_emb=torch.zeros(2),
            latent_dim=2,
            device=torch.device("cpu"),
            reference_end_latent=EOS_TOKEN.squeeze(0),
            stop_threshold=0.01,
            enable_stopping=True,
            text_source="offline",
            text_lookup=TextLookup(),
            text_embed_dim=2,
            disable_rag=True,
        )

        result = sampler.sample_for_eval_CFG(
            "walk",
            length=8,
            unit_length=4,
        )

        torch.testing.assert_close(result, EXPECTED_MOTION_LATENTS)

    def test_single_motion_sampler_excludes_generated_eos_from_motion_latents(self):
        result = sample_motion_latents_with_stop(
            rag_model=SequenceRAGModel(),
            text_encoder=TextEncoder(),
            retriever=None,
            input_text="walk",
            empty_text_emb=torch.zeros(2),
            reference_end=EOS_TOKEN.squeeze(0),
            disable_rag_flag=True,
            embed_dim=2,
            stop_threshold=0.01,
            length=8,
            unit_len=4,
            cfg=4.0,
            token_latent_dim=2,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(result, EXPECTED_MOTION_LATENTS)


if __name__ == "__main__":
    unittest.main()
