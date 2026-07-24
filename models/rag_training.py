"""Training-only loss boundary for the official global-RAG model."""

import torch
from torch import nn


def lengths_to_mask(lengths, max_len):
    return (
        torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        < lengths.unsqueeze(1)
    )


def estimate_lengths_from_padded_latents(m_tokens):
    valid = m_tokens.abs().sum(dim=-1) > 0
    lengths = valid.long().sum(dim=1)
    return torch.clamp(lengths, min=2)


def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    step = torch.tensor(step, dtype=torch.float32)
    total_steps = torch.tensor(total_steps, dtype=torch.float32)
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor


def replace_with_pred(latents, pred_xstart, step, total_steps):
    decay_factor = cosine_decay(step, total_steps).to(latents.device)
    batch_size, sequence_length, _ = latents.shape
    num_replace = int(sequence_length * decay_factor)
    replace_indices = torch.randperm(
        sequence_length, device=latents.device
    )[:num_replace]
    replace_mask = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.bool,
        device=latents.device,
    )
    replace_mask[:, replace_indices] = 1
    updated_latents = latents.clone()
    updated_latents[replace_mask] = pred_xstart[replace_mask]
    return updated_latents


class RAGTwoForwardLoss(nn.Module):
    """Compute both pseudo-target and gradient passes inside one module forward."""

    def __init__(self, rag_model, diffmlps_batch_mul=4):
        super().__init__()
        self.rag_model = rag_model
        self.diffmlps_batch_mul = int(diffmlps_batch_mul)

    def forward(
        self,
        latents,
        m_lens,
        text_emb,
        top3_h_cls,
        top3_sim_scores,
        step,
        total_steps,
        cfg_drop_mask,
        empty_text_emb,
    ):
        batch_size, sequence_length, _ = latents.shape
        mask = lengths_to_mask(m_lens, sequence_length).reshape(
            batch_size * sequence_length
        ).repeat(self.diffmlps_batch_mul)

        with torch.no_grad():
            conditions = self.rag_model(
                motion_latents=latents,
                text_emb=text_emb,
                top3_h_cls=top3_h_cls,
                top3_sim_scores=top3_sim_scores,
                cfg_drop_mask=cfg_drop_mask,
                empty_text_emb=empty_text_emb,
            )
            z = self.rag_model.motion_condition_slice(
                conditions, sequence_length
            )
            target = (
                latents.clone()
                .detach()
                .reshape(batch_size * sequence_length, -1)
            )
            z = z.reshape(batch_size * sequence_length, -1)
            _, pred_xstart = self.rag_model.base_model.diff_loss(
                target=target, z=z
            )

        pred_xstart = (
            pred_xstart.clone()
            .detach()
            .reshape(batch_size, sequence_length, -1)
        )
        updated_latents = replace_with_pred(
            latents, pred_xstart, step, total_steps
        )
        updated_conditions = self.rag_model(
            motion_latents=updated_latents,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )
        updated_z = self.rag_model.motion_condition_slice(
            updated_conditions, sequence_length
        )
        updated_target = (
            latents.clone()
            .detach()
            .reshape(batch_size * sequence_length, -1)
            .repeat(self.diffmlps_batch_mul, 1)
        )
        updated_z = (
            updated_z.reshape(batch_size * sequence_length, -1)
            .repeat(self.diffmlps_batch_mul, 1)
        )
        updated_loss, _ = self.rag_model.base_model.diff_loss(
            target=updated_target[mask],
            z=updated_z[mask],
        )
        return updated_loss


def get_rag_model(model):
    """Return the research model without DDP or training-wrapper prefixes."""
    while hasattr(model, "module"):
        model = model.module
    return model.rag_model if hasattr(model, "rag_model") else model
