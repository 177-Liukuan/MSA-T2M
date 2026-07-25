from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class BatchedLatentResult:
    latents: List[torch.Tensor]
    stop_steps: torch.Tensor
    empty_fallback_count: int


def _select_batch(tensor: Optional[torch.Tensor], indices: torch.Tensor):
    if tensor is None:
        return None
    return tensor.index_select(0, indices)


@torch.no_grad()
def generate_latents_active_set(
    rag_model,
    text_emb,
    empty_text_emb,
    top_hcls,
    top_scores,
    max_token_lengths,
    latent_dim,
    reference_end_latent,
    stop_threshold,
    enable_stopping,
    cfg_scale,
    temperature=1.0,
):
    if text_emb.ndim != 2:
        raise ValueError("text_emb must have shape [batch, text_dim]")

    batch_size = text_emb.shape[0]
    if max_token_lengths.ndim != 1 or max_token_lengths.shape[0] != batch_size:
        raise ValueError("max_token_lengths must have shape [batch]")
    if torch.any(max_token_lengths <= 0):
        raise ValueError("max_token_lengths values must be positive")
    if top_hcls is not None and top_hcls.shape[0] != batch_size:
        raise ValueError("top_hcls batch size does not match text_emb")
    if top_scores is not None and top_scores.shape[0] != batch_size:
        raise ValueError("top_scores batch size does not match text_emb")

    latent_dim = int(latent_dim)
    if enable_stopping:
        if reference_end_latent is None:
            raise ValueError("reference_end_latent is required when stopping is enabled")
        if reference_end_latent.ndim != 1 or reference_end_latent.shape[0] != latent_dim:
            raise ValueError(
                "reference_end_latent must have shape [{}]".format(latent_dim)
            )

    ceilings = max_token_lengths.detach().to(device="cpu", dtype=torch.long)
    accepted = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool)
    stop_steps = torch.full((batch_size,), -1, dtype=torch.long)

    for step in range(int(ceilings.max().item())):
        active_cpu = (~finished) & (ceilings > step)
        active_indices_cpu = torch.nonzero(active_cpu, as_tuple=False).flatten()
        if active_indices_cpu.numel() == 0:
            break

        active_indices = active_indices_cpu.to(text_emb.device)
        if step == 0:
            prefix = torch.zeros(
                active_indices.numel(),
                0,
                latent_dim,
                device=text_emb.device,
                dtype=torch.float32,
            )
        else:
            prefix = torch.stack(
                [
                    torch.stack(accepted[index], dim=0)
                    for index in active_indices_cpu.tolist()
                ],
                dim=0,
            )

        candidates = rag_model.sample_next_with_cfg(
            motion_prefix=prefix,
            text_emb=text_emb.index_select(0, active_indices),
            empty_text_emb=empty_text_emb,
            top3_h_cls=_select_batch(top_hcls, active_indices),
            top3_sim_scores=_select_batch(top_scores, active_indices),
            cfg_scale=cfg_scale,
            temperature=temperature,
        )
        if candidates.ndim != 2 or candidates.shape != (
            active_indices.numel(),
            latent_dim,
        ):
            raise ValueError(
                "rag_model returned {}, expected [{}, {}]".format(
                    tuple(candidates.shape),
                    active_indices.numel(),
                    latent_dim,
                )
            )

        if enable_stopping:
            distances = torch.linalg.norm(
                candidates - reference_end_latent.unsqueeze(0),
                dim=-1,
            )
            emitted_eos = distances < float(stop_threshold)
        else:
            emitted_eos = torch.zeros(
                candidates.shape[0],
                dtype=torch.bool,
                device=candidates.device,
            )

        for local_index, sample_index in enumerate(active_indices_cpu.tolist()):
            if bool(emitted_eos[local_index].item()):
                finished[sample_index] = True
                stop_steps[sample_index] = step
            else:
                accepted[sample_index].append(candidates[local_index])

    latents = []
    empty_fallback_count = 0
    for tokens in accepted:
        if tokens:
            latents.append(torch.stack(tokens, dim=0).unsqueeze(0))
        else:
            empty_fallback_count += 1
            latents.append(
                torch.zeros(
                    1,
                    1,
                    latent_dim,
                    device=text_emb.device,
                    dtype=torch.float32,
                )
            )

    return BatchedLatentResult(
        latents=latents,
        stop_steps=stop_steps,
        empty_fallback_count=empty_fallback_count,
    )
