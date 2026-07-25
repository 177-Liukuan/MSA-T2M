"""Training-only loss boundary for the official global-RAG model."""

import torch
from torch import nn


def lengths_to_mask(lengths, max_len):
    return (
        torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        < lengths.unsqueeze(1)
    )


def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    step = torch.tensor(step, dtype=torch.float32)
    total_steps = torch.tensor(total_steps, dtype=torch.float32)
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor


def replace_with_pred(
    latents,
    pred_xstart,
    step,
    total_steps,
    valid_mask,
):
    decay_factor = cosine_decay(step, total_steps).to(latents.device)
    batch_size, sequence_length, _ = latents.shape
    if valid_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            "valid_mask shape {} does not match latent batch shape {}".format(
                tuple(valid_mask.shape),
                (batch_size, sequence_length),
            )
        )

    random_scores = torch.rand(
        batch_size,
        sequence_length,
        device=latents.device,
    )
    random_scores = random_scores.masked_fill(~valid_mask, float("inf"))
    random_ranks = random_scores.argsort(dim=1).argsort(dim=1)
    replace_counts = torch.floor(
        valid_mask.sum(dim=1).to(torch.float32) * decay_factor
    ).to(torch.long)
    replace_mask = valid_mask & (
        random_ranks < replace_counts.unsqueeze(1)
    )

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
        if m_lens.ndim != 1 or m_lens.shape[0] != batch_size:
            raise ValueError(
                "m_lens must have shape ({},), got {}".format(
                    batch_size,
                    tuple(m_lens.shape),
                )
            )
        if torch.any(m_lens <= 0):
            raise ValueError("motion latent lengths must be positive")
        if torch.any(m_lens > sequence_length):
            raise ValueError(
                "motion latent length exceeds padded width {}".format(
                    sequence_length
                )
            )

        valid_mask = lengths_to_mask(m_lens, sequence_length)
        mask = valid_mask.reshape(
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
            latents,
            pred_xstart,
            step,
            total_steps,
            valid_mask,
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
