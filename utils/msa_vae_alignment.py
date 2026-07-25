"""Numerically stable masked semantic-alignment losses for MSA-VAE."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AlignmentLossResult:
    """Local backward loss plus globally reduced logging statistics."""

    backward_loss: torch.Tensor
    global_mean: torch.Tensor
    valid_count: torch.Tensor


def masked_cosine_sum_and_count(feat_a, feat_b, sample_mask):
    """Return the summed cosine distance and valid vector count.

    ``sample_mask`` selects batch entries.  For token features every token of
    a selected sample participates, so the count is ``selected_samples * T``.
    Empty selections return an autograd-connected zero.
    """
    if feat_a.shape != feat_b.shape:
        raise ValueError("alignment features must have identical shapes")
    if feat_a.dim() not in (2, 3):
        raise ValueError("alignment features must have shape (B, D) or (B, T, D)")
    if sample_mask is None:
        sample_mask = torch.ones(feat_a.size(0), device=feat_a.device, dtype=torch.bool)
    else:
        sample_mask = sample_mask.to(device=feat_a.device, dtype=torch.bool)
        if sample_mask.shape != (feat_a.size(0),):
            raise ValueError("sample_mask must have shape (B,)")

    if feat_a.dim() == 3:
        token_mask = sample_mask.unsqueeze(1).expand(-1, feat_a.size(1)).reshape(-1)
        flat_a = feat_a.reshape(-1, feat_a.size(-1))
        flat_b = feat_b.reshape(-1, feat_b.size(-1))
    else:
        token_mask = sample_mask
        flat_a = feat_a
        flat_b = feat_b

    valid_count = token_mask.sum(dtype=torch.long)
    if valid_count.item() == 0:
        return feat_a.sum() * 0.0, valid_count

    distances = 1.0 - F.cosine_similarity(flat_a[token_mask], flat_b[token_mask], dim=-1)
    return distances.sum(), valid_count


def distributed_masked_cosine_alignment(feat_a, feat_b, sample_mask, accelerator):
    """Compute DDP-correct masked alignment and globally meaningful metrics.

    Accelerate/DDP averages gradients across ranks.  Multiplying each rank's
    local sum by the world size before dividing by the all-rank valid count
    makes that averaged gradient equal the gradient of the global mean, even
    if a rank has no valid samples.
    """
    local_sum, local_count = masked_cosine_sum_and_count(feat_a, feat_b, sample_mask)
    global_count = accelerator.reduce(local_count.detach(), reduction="sum")
    global_sum = accelerator.reduce(local_sum.detach(), reduction="sum")

    if global_count.item() == 0:
        zero = feat_a.sum() * 0.0
        return AlignmentLossResult(zero, zero.detach(), global_count)

    backward_loss = local_sum * accelerator.num_processes / global_count.to(local_sum.dtype)
    global_mean = global_sum / global_count.to(global_sum.dtype)
    return AlignmentLossResult(backward_loss, global_mean, global_count)
