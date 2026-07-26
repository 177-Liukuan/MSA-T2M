"""Metrics for MSA-VAE's native SentenceT5 alignment space."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _validated_float_tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if value.ndim != ndim:
        raise ValueError(
            "{} must be {}D, got shape {}".format(
                name,
                ndim,
                tuple(value.shape),
            )
        )
    if any(size <= 0 for size in value.shape):
        raise ValueError("{} must not contain an empty dimension".format(name))
    tensor = value.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("{} contains non-finite values".format(name))
    return tensor


def _validated_global_inputs(
    motion_embeddings,
    text_embeddings,
    text_motion_indices,
):
    motion = _validated_float_tensor(
        "motion_embeddings",
        motion_embeddings,
        ndim=2,
    )
    text = _validated_float_tensor(
        "text_embeddings",
        text_embeddings,
        ndim=2,
    )
    if motion.shape[1] != text.shape[1]:
        raise ValueError(
            "motion and text embedding dimensions must match"
        )
    if not isinstance(text_motion_indices, torch.Tensor):
        raise TypeError("text_motion_indices must be a torch.Tensor")
    if text_motion_indices.ndim != 1:
        raise ValueError("text_motion_indices must be 1D")
    if text_motion_indices.shape[0] != text.shape[0]:
        raise ValueError(
            "text_motion_indices must contain one owner per text row"
        )
    if text_motion_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("text_motion_indices must use an integer dtype")
    owners = text_motion_indices.to(
        device=motion.device,
        dtype=torch.long,
    )
    if text.device != motion.device:
        text = text.to(device=motion.device)
    motion_count = motion.shape[0]
    if torch.any(owners < 0).item() or torch.any(
        owners >= motion_count
    ).item():
        raise ValueError("text owner index is outside the motion range")
    owner_counts = torch.bincount(owners, minlength=motion_count)
    if torch.any(owner_counts == 0).item():
        raise ValueError("every motion must own at least one text caption")
    motion_norms = torch.linalg.vector_norm(motion, dim=1)
    text_norms = torch.linalg.vector_norm(text, dim=1)
    if torch.any(motion_norms == 0).item():
        raise ValueError("motion_embeddings contains a zero-norm row")
    if torch.any(text_norms == 0).item():
        raise ValueError("text_embeddings contains a zero-norm row")
    return motion, text, owners, owner_counts


def calculate_motion_macro_cosine(
    motion_embeddings,
    text_embeddings,
    text_motion_indices,
):
    """Average caption cosine within each motion, then across motions."""

    motion, text, owners, owner_counts = _validated_global_inputs(
        motion_embeddings,
        text_embeddings,
        text_motion_indices,
    )
    motion = F.normalize(motion, dim=1)
    text = F.normalize(text, dim=1)
    caption_cosines = (
        text * motion.index_select(0, owners)
    ).sum(dim=1)
    motion_sums = torch.zeros(
        motion.shape[0],
        dtype=caption_cosines.dtype,
        device=caption_cosines.device,
    )
    motion_sums.index_add_(0, owners, caption_cosines)
    motion_means = motion_sums / owner_counts.to(
        dtype=caption_cosines.dtype
    )
    return float(motion_means.mean().item())


def calculate_masked_local_cosine(
    local_embeddings,
    local_targets,
    valid_mask,
):
    """Average valid token cosine within each motion, then across motions."""

    prediction = _validated_float_tensor(
        "local_embeddings",
        local_embeddings,
        ndim=3,
    )
    target = _validated_float_tensor(
        "local_targets",
        local_targets,
        ndim=3,
    )
    if prediction.shape != target.shape:
        raise ValueError(
            "local embedding and target shapes must match"
        )
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use bool dtype")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError(
            "valid_mask shape must match batch and token dimensions"
        )
    if valid_mask.device != prediction.device:
        mask = valid_mask.to(device=prediction.device)
    else:
        mask = valid_mask
    if target.device != prediction.device:
        target = target.to(device=prediction.device)
    token_counts = mask.sum(dim=1)
    if torch.any(token_counts == 0).item():
        raise ValueError("every motion must contain at least one valid token")
    valid_prediction = prediction[mask]
    valid_target = target[mask]
    prediction_norms = torch.linalg.vector_norm(
        valid_prediction,
        dim=1,
    )
    target_norms = torch.linalg.vector_norm(valid_target, dim=1)
    if torch.any(prediction_norms == 0).item():
        raise ValueError("local_embeddings contains a valid zero-norm row")
    if torch.any(target_norms == 0).item():
        raise ValueError("local_targets contains a valid zero-norm row")
    token_cosines = (
        F.normalize(valid_prediction, dim=1)
        * F.normalize(valid_target, dim=1)
    ).sum(dim=1)
    token_owners = (
        torch.arange(
            prediction.shape[0],
            device=prediction.device,
            dtype=torch.long,
        )[:, None]
        .expand_as(mask)[mask]
    )
    motion_sums = torch.zeros(
        prediction.shape[0],
        dtype=token_cosines.dtype,
        device=prediction.device,
    )
    motion_sums.index_add_(0, token_owners, token_cosines)
    motion_means = motion_sums / token_counts.to(
        dtype=token_cosines.dtype
    )
    return float(motion_means.mean().item())


def _retrieval_summary(ranks, prefix):
    ranks = ranks.to(dtype=torch.float32)
    metrics = {}
    for cutoff in (1, 2, 3, 5):
        metrics[
            "{}r{}_percent".format(prefix, cutoff)
        ] = float((ranks < cutoff).float().mean().item() * 100.0)
    metrics["{}medr".format(prefix)] = float(
        np.median(ranks.detach().cpu().numpy()) + 1.0
    )
    return metrics


def _multi_positive_ranks(
    similarity,
    owners,
) -> Tuple[torch.Tensor, torch.Tensor]:
    caption_count, motion_count = similarity.shape
    caption_rows = torch.arange(
        caption_count,
        device=similarity.device,
    )
    positive_scores = similarity[caption_rows, owners]
    tied = similarity == positive_scores[:, None]
    t2m_ranks = (
        (similarity > positive_scores[:, None]).sum(dim=1).float()
        + (tied.sum(dim=1).float() - 1.0) / 2.0
    )

    m2t_ranks = []
    for motion_index in range(motion_count):
        positive_mask = owners == motion_index
        column = similarity[:, motion_index]
        best_positive_score = column[positive_mask].max()
        negative_scores = column[~positive_mask]
        rank = (
            (negative_scores > best_positive_score).sum().float()
            + (negative_scores == best_positive_score).sum().float() / 2.0
        )
        m2t_ranks.append(rank)
    return t2m_ranks, torch.stack(m2t_ranks)


def calculate_msa_t5_retrieval(
    text_embeddings,
    motion_embeddings,
    text_motion_indices,
) -> Dict[str, float]:
    """Compute rectangular multi-positive retrieval in MSA's T5 space."""

    motion, text, owners, _ = _validated_global_inputs(
        motion_embeddings,
        text_embeddings,
        text_motion_indices,
    )
    similarity = F.normalize(text, dim=1) @ F.normalize(
        motion,
        dim=1,
    ).transpose(0, 1)
    t2m_ranks, m2t_ranks = _multi_positive_ranks(
        similarity,
        owners,
    )
    metrics = _retrieval_summary(t2m_ranks, "msa_t5_t2m_")
    metrics.update(_retrieval_summary(m2t_ranks, "msa_t5_m2t_"))
    return metrics


def shuffled_text_control(text_embeddings, seed):
    """Return a deterministic derangement of caption target rows."""

    text = _validated_float_tensor(
        "text_embeddings",
        text_embeddings,
        ndim=2,
    )
    row_count = text.shape[0]
    if row_count < 2:
        raise ValueError(
            "shuffled text control requires at least two caption rows"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    original = torch.arange(row_count)
    permutation = None
    for _ in range(256):
        candidate = torch.randperm(row_count, generator=generator)
        if torch.all(candidate != original).item():
            permutation = candidate
            break
    if permutation is None:
        offset = int(
            torch.randint(
                1,
                row_count,
                (1,),
                generator=generator,
            ).item()
        )
        permutation = torch.roll(original, shifts=offset)
    return text.index_select(0, permutation.to(device=text.device))
