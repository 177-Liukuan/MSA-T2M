import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from models.llama_rag_model import LLaMARAGWrapper
from models.rag_training import (
    RAGTwoForwardLoss,
    get_rag_model,
    lengths_to_mask,
    replace_with_pred,
)


class FakeDiffusionLoss(nn.Module):
    def __init__(self, model_dim, latent_dim):
        super().__init__()
        self.projection = nn.Linear(model_dim, latent_dim, bias=False)

    def forward(self, target, z):
        pred_xstart = self.projection(z)
        return torch.mean((pred_xstart - target) ** 2), pred_xstart


class FakeBaseModel(nn.Module):
    def __init__(self, model_dim, latent_dim):
        super().__init__()
        self.diff_loss = FakeDiffusionLoss(model_dim, latent_dim)


class FakeRAGModel(nn.Module):
    def __init__(self, latent_dim=2, model_dim=3, text_dim=2):
        super().__init__()
        self.base_model = FakeBaseModel(model_dim, latent_dim)
        self.motion_projection = nn.Linear(latent_dim, model_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, model_dim, bias=False)
        self.retrieval_projection = nn.Linear(text_dim, model_dim, bias=False)
        self.num_condition_tokens = 2
        self.forward_grad_enabled = []

    def forward(
        self,
        motion_latents,
        text_emb,
        top3_h_cls,
        top3_sim_scores,
        cfg_drop_mask,
        empty_text_emb,
    ):
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        text_token = self.text_projection(text_emb).unsqueeze(1)
        weights = torch.softmax(top3_sim_scores, dim=1).unsqueeze(-1)
        retrieval = (top3_h_cls * weights).sum(dim=1)
        retrieval_token = self.retrieval_projection(retrieval).unsqueeze(1)
        motion_tokens = self.motion_projection(motion_latents) + text_token
        return torch.cat([text_token, retrieval_token, motion_tokens], dim=1)

    def motion_condition_slice(self, hidden_states, motion_len):
        start = self.num_condition_tokens - 1
        return hidden_states[:, start : start + motion_len]


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Linear(2, 4)
        self.cond_embed = nn.Linear(2, 4)
        self.h = nn.ModuleList()
        self.ln_f = nn.Identity()


class TinyBaseModel(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.config = SimpleNamespace(block_size=block_size)
        self.transformer = TinyTransformer()
        self.out_proj = nn.Identity()


def reference_lengths_to_mask(lengths, max_len):
    return (
        torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        < lengths.unsqueeze(1)
    )


def reference_cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    step = torch.tensor(step, dtype=torch.float32)
    total_steps = torch.tensor(total_steps, dtype=torch.float32)
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor


def reference_replace_with_pred(latents, pred_xstart, step, total_steps):
    decay_factor = reference_cosine_decay(step, total_steps).to(latents.device)
    bsz, seq_len, _ = latents.shape
    num_replace = int(seq_len * decay_factor)
    replace_indices = torch.randperm(seq_len, device=latents.device)[:num_replace]
    replace_mask = torch.zeros(
        bsz, seq_len, dtype=torch.bool, device=latents.device
    )
    replace_mask[:, replace_indices] = 1
    updated_latents = latents.clone()
    updated_latents[replace_mask] = pred_xstart[replace_mask]
    return updated_latents


def reference_two_forward_loss(
    latents,
    rag_model,
    m_lens,
    text_emb,
    top3_h_cls,
    top3_sim_scores,
    step,
    total_steps,
    cfg_drop_mask,
    empty_text_emb,
    diffmlps_batch_mul=4,
):
    bsz, seq_len, _ = latents.shape
    mask = reference_lengths_to_mask(m_lens, seq_len).reshape(
        bsz * seq_len
    ).repeat(diffmlps_batch_mul)
    with torch.no_grad():
        conditions = rag_model(
            motion_latents=latents,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )
        z = rag_model.motion_condition_slice(conditions, seq_len)
        target = latents.clone().detach().reshape(bsz * seq_len, -1)
        z = z.reshape(bsz * seq_len, -1)
        _, pred_xstart = rag_model.base_model.diff_loss(target=target, z=z)

    pred_xstart = pred_xstart.clone().detach().reshape(bsz, seq_len, -1)
    updated_latents = reference_replace_with_pred(
        latents, pred_xstart, step, total_steps
    )
    updated_conditions = rag_model(
        motion_latents=updated_latents,
        text_emb=text_emb,
        top3_h_cls=top3_h_cls,
        top3_sim_scores=top3_sim_scores,
        cfg_drop_mask=cfg_drop_mask,
        empty_text_emb=empty_text_emb,
    )
    updated_z = rag_model.motion_condition_slice(updated_conditions, seq_len)
    updated_target = (
        latents.clone()
        .detach()
        .reshape(bsz * seq_len, -1)
        .repeat(diffmlps_batch_mul, 1)
    )
    updated_z = (
        updated_z.reshape(bsz * seq_len, -1).repeat(diffmlps_batch_mul, 1)
    )
    updated_loss, _ = rag_model.base_model.diff_loss(
        target=updated_target[mask], z=updated_z[mask]
    )
    return updated_loss


def make_batch():
    return {
        "latents": torch.tensor(
            [
                [[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]],
                [[-0.2, 0.7], [0.9, 0.1], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "m_lens": torch.tensor([3, 2], dtype=torch.long),
        "text_emb": torch.tensor([[0.2, 0.8], [0.7, -0.1]], dtype=torch.float32),
        "top3_h_cls": torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.6, 0.8], [-1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "top3_sim_scores": torch.tensor(
            [[0.8, 0.2], [0.5, -0.1]], dtype=torch.float32
        ),
        "step": 10,
        "total_steps": 10,
        "cfg_drop_mask": torch.tensor([False, True]),
        "empty_text_emb": torch.zeros(2),
    }


class ModuleHolder:
    def __init__(self, module):
        self.module = module


class AcceleratorDouble:
    @staticmethod
    def unwrap_model(model):
        return model


class SchedulerDouble:
    @staticmethod
    def state_dict():
        return {"scheduler": "state"}


class RAGTrainingTest(unittest.TestCase):
    def test_rag_wrapper_enforces_total_context_length(self):
        rag_model = LLaMARAGWrapper(
            base_model=TinyBaseModel(block_size=78),
            model_dim=4,
            retrieval_dim=2,
        )
        conditions = {
            "text_emb": torch.zeros(1, 2),
            "top3_h_cls": torch.zeros(1, 1, 2),
            "top3_sim_scores": torch.zeros(1, 1),
        }

        output = rag_model(
            motion_latents=torch.zeros(1, 76, 2),
            **conditions,
        )
        self.assertEqual(tuple(output.shape), (1, 78, 4))

        with self.assertRaisesRegex(ValueError, "block_size 78"):
            rag_model(
                motion_latents=torch.zeros(1, 77, 2),
                **conditions,
            )

    def test_loss_rejects_non_positive_motion_length(self):
        batch = make_batch()
        batch["m_lens"] = torch.tensor([3, 0], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "positive"):
            RAGTwoForwardLoss(FakeRAGModel())(**batch)

    def test_loss_rejects_motion_length_beyond_padded_width(self):
        batch = make_batch()
        batch["m_lens"] = torch.tensor([4, 2], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "padded width"):
            RAGTwoForwardLoss(FakeRAGModel())(**batch)

    def test_replace_with_pred_never_changes_padding(self):
        latents = torch.zeros(2, 3, 1)
        pred_xstart = torch.ones_like(latents)
        valid_mask = lengths_to_mask(torch.tensor([3, 1]), max_len=3)

        updated = replace_with_pred(
            latents,
            pred_xstart,
            step=10,
            total_steps=10,
            valid_mask=valid_mask,
        )

        torch.testing.assert_close(
            updated,
            torch.tensor(
                [
                    [[1.0], [1.0], [1.0]],
                    [[1.0], [0.0], [0.0]],
                ]
            ),
        )

    def test_replace_with_pred_uses_each_samples_valid_length(self):
        latents = torch.zeros(2, 4, 1)
        pred_xstart = torch.ones_like(latents)
        valid_mask = lengths_to_mask(torch.tensor([4, 2]), max_len=4)

        torch.manual_seed(23)
        updated = replace_with_pred(
            latents,
            pred_xstart,
            step=5,
            total_steps=10,
            valid_mask=valid_mask,
        )

        replaced_counts = (updated != latents).squeeze(-1).sum(dim=1)
        torch.testing.assert_close(replaced_counts, torch.tensor([2, 1]))
        self.assertTrue(torch.equal(updated[1, 2:], latents[1, 2:]))

    def test_loss_and_gradients_match_reference(self):
        torch.manual_seed(19)
        reference_model = FakeRAGModel()
        actual_model = copy.deepcopy(reference_model)
        batch = make_batch()

        rng_state = torch.random.get_rng_state()
        reference_loss = reference_two_forward_loss(
            rag_model=reference_model, **batch
        )
        reference_loss.backward()

        torch.random.set_rng_state(rng_state)
        training_module = RAGTwoForwardLoss(actual_model)
        actual_loss = training_module(**batch)
        actual_loss.backward()

        torch.testing.assert_close(actual_loss, reference_loss)
        for (reference_name, reference_parameter), (
            actual_name,
            actual_parameter,
        ) in zip(reference_model.named_parameters(), actual_model.named_parameters()):
            self.assertEqual(actual_name, reference_name)
            if reference_parameter.grad is None:
                self.assertIsNone(actual_parameter.grad)
            else:
                torch.testing.assert_close(
                    actual_parameter.grad, reference_parameter.grad
                )
        self.assertEqual(actual_model.forward_grad_enabled, [False, True])

    def test_get_rag_model_preserves_research_state_dict_names(self):
        rag_model = FakeRAGModel()
        training_module = RAGTwoForwardLoss(rag_model)
        expected_keys = list(rag_model.state_dict())

        self.assertEqual(list(get_rag_model(training_module).state_dict()), expected_keys)
        self.assertEqual(
            list(get_rag_model(ModuleHolder(training_module)).state_dict()),
            expected_keys,
        )
        self.assertFalse(
            any(key.startswith("rag_model.") for key in get_rag_model(training_module).state_dict())
        )

    def test_checkpoint_payload_preserves_research_model_keys(self):
        from train_t2m_rag import build_checkpoint_payload

        rag_model = FakeRAGModel()
        training_module = RAGTwoForwardLoss(rag_model)
        optimizer = torch.optim.SGD(training_module.parameters(), lr=0.1)
        ema_base_state = copy.deepcopy(rag_model.base_model.state_dict())
        ema_rag_state = copy.deepcopy(rag_model.state_dict())

        payload = build_checkpoint_payload(
            training_model=training_module,
            accelerator=AcceleratorDouble(),
            optimizer=optimizer,
            scheduler=SchedulerDouble(),
            iteration=17,
            generative_head_type="ddpm",
            ema_enabled=True,
            ema_decay=0.9999,
            ema_base_state=ema_base_state,
            ema_rag_state=ema_rag_state,
        )

        self.assertEqual(
            set(payload),
            {
                "trans",
                "rag",
                "scheduler",
                "optimizer",
                "iter",
                "generative_head_type",
                "use_ema",
                "ema_decay",
                "trans_ema",
                "rag_ema",
            },
        )
        self.assertEqual(list(payload["rag"]), list(rag_model.state_dict()))
        self.assertEqual(
            {key: tuple(value.shape) for key, value in payload["rag"].items()},
            {
                key: tuple(value.shape)
                for key, value in rag_model.state_dict().items()
            },
        )
        self.assertFalse(any(key.startswith("rag_model.") for key in payload["rag"]))
        self.assertEqual(payload["iter"], 17)
        self.assertEqual(payload["generative_head_type"], "ddpm")


if __name__ == "__main__":
    unittest.main()
