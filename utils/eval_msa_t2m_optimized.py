from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.eval_trans import (
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
)


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


class OptimizedRAGEvalSampler:
    def __init__(
        self,
        rag_model,
        retriever,
        empty_text_emb,
        latent_dim=16,
        device=torch.device("cuda"),
        reference_end_latent=None,
        stop_threshold=3.0,
        enable_stopping=True,
        text_source="offline",
        text_lookup=None,
        text_encoder=None,
        text_embed_dim=768,
        disable_rag=False,
        use_random_topk_inference=False,
    ):
        self.rag_model = rag_model
        self.retriever = retriever
        self.empty_text_emb = empty_text_emb
        self.latent_dim = int(latent_dim)
        self.device = device
        self.reference_end_latent = reference_end_latent
        self.stop_threshold = float(stop_threshold)
        self.enable_stopping = bool(enable_stopping)
        self.text_source = text_source
        self.text_lookup = text_lookup
        self.text_encoder = text_encoder
        self.text_embed_dim = int(text_embed_dim)
        self.disable_rag = bool(disable_rag)
        self.use_random_topk_inference = bool(use_random_topk_inference)

    def eval(self):
        self.rag_model.eval()
        return self

    @torch.no_grad()
    def sample_batch_for_eval_CFG(
        self,
        text,
        lengths,
        unit_length=4,
        cfg=4.0,
    ):
        text_list = [text] if isinstance(text, str) else list(text)
        if len(text_list) == 0:
            raise ValueError("text batch must not be empty")

        if self.text_source == "online_t5":
            if self.text_encoder is None:
                raise RuntimeError(
                    "text_encoder is required when text_source=online_t5"
                )
            text_np = np.asarray(
                self.text_encoder.encode(text_list),
                dtype=np.float32,
            )
            text_emb = torch.from_numpy(text_np).float().to(self.device)
        else:
            if self.text_lookup is None:
                raise RuntimeError(
                    "text_lookup is required when text_source=offline"
                )
            text_emb = self.text_lookup.batch_lookup(text_list, self.device)

        if text_emb.ndim != 2 or text_emb.shape != (
            len(text_list),
            self.text_embed_dim,
        ):
            raise ValueError(
                "text embedding shape mismatch: got {}, expected [{}, {}]".format(
                    tuple(text_emb.shape),
                    len(text_list),
                    self.text_embed_dim,
                )
            )

        if self.disable_rag:
            top_hcls = None
            top_scores = None
        else:
            if self.retriever is None:
                raise RuntimeError("retriever is required when RAG is enabled")
            top_hcls, top_scores = self.retriever.retrieve(text_emb)
            if self.use_random_topk_inference and top_hcls.shape[1] > 1:
                batch_size, topk, _ = top_hcls.shape
                selected = torch.randint(
                    0,
                    topk,
                    (batch_size,),
                    device=top_hcls.device,
                )
                rows = torch.arange(batch_size, device=top_hcls.device)
                top_hcls = top_hcls[rows, selected].unsqueeze(1)
                top_scores = top_scores[rows, selected].unsqueeze(1)

        lengths_tensor = torch.as_tensor(
            lengths,
            dtype=torch.long,
            device=self.device,
        )
        if lengths_tensor.ndim != 1 or lengths_tensor.shape[0] != len(text_list):
            raise ValueError("lengths must have one value per caption")
        max_token_lengths = torch.clamp(
            lengths_tensor // int(unit_length),
            min=1,
        )

        return generate_latents_active_set(
            rag_model=self.rag_model,
            text_emb=text_emb,
            empty_text_emb=self.empty_text_emb,
            top_hcls=top_hcls,
            top_scores=top_scores,
            max_token_lengths=max_token_lengths,
            latent_dim=self.latent_dim,
            reference_end_latent=self.reference_end_latent,
            stop_threshold=self.stop_threshold,
            enable_stopping=self.enable_stopping,
            cfg_scale=cfg,
        )


@torch.no_grad()
def decode_equal_length_groups(
    decoder,
    latent_sequences,
    max_motion_length,
    motion_dim,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(latent_sequences) == 0:
        raise ValueError("latent_sequences must not be empty")

    groups: Dict[int, List[int]] = {}
    for index, latent in enumerate(latent_sequences):
        if latent.ndim != 3 or latent.shape[0] != 1:
            raise ValueError("each latent sequence must have shape [1, T, D]")
        groups.setdefault(int(latent.shape[1]), []).append(index)

    decoded_by_index = {}
    pred_lengths = torch.zeros(len(latent_sequences), dtype=torch.long)
    output_dtype = None
    output_device = None

    for indices in groups.values():
        grouped_latents = torch.cat(
            [latent_sequences[index] for index in indices],
            dim=0,
        )
        decoded = decoder.forward_decoder(grouped_latents)
        if decoded.ndim != 3 or decoded.shape[0] != len(indices):
            raise ValueError("decoder output must have shape [batch, frames, dim]")
        if decoded.shape[-1] != int(motion_dim):
            raise ValueError(
                "decoder motion dim mismatch: got {}, expected {}".format(
                    decoded.shape[-1],
                    motion_dim,
                )
            )
        output_dtype = decoded.dtype
        output_device = decoded.device
        for local_index, original_index in enumerate(indices):
            decoded_by_index[original_index] = decoded[local_index]
            pred_lengths[original_index] = min(
                int(decoded.shape[1]),
                int(max_motion_length),
            )

    motions = torch.zeros(
        len(latent_sequences),
        int(max_motion_length),
        int(motion_dim),
        dtype=output_dtype,
        device=output_device,
    )
    for index in range(len(latent_sequences)):
        decoded = decoded_by_index[index]
        copy_length = min(int(decoded.shape[0]), int(max_motion_length))
        motions[index, :copy_length] = decoded[:copy_length]

    return motions, pred_lengths


@torch.no_grad()
def evaluation_transformer_272_optimized(
    val_loader,
    net,
    trans,
    logger,
    evaluator,
    cfg=4.0,
    device=torch.device("cuda"),
    unit_length=4,
):
    textencoder, motionencoder = evaluator
    trans.eval()

    motion_annotation_list = []
    motion_pred_list = []
    r_precision_real = torch.zeros(3, device=device)
    r_precision_pred = torch.zeros(3, device=device)
    matching_score_real = torch.tensor(0.0, device=device)
    matching_score_pred = torch.tensor(0.0, device=device)
    sample_count = 0
    empty_fallback_count = 0

    for text, pose, motion_lengths in val_loader:
        batch_result = trans.sample_batch_for_eval_CFG(
            text,
            motion_lengths,
            unit_length=unit_length,
            cfg=cfg,
        )
        if len(batch_result.latents) != len(text):
            raise ValueError(
                "generated sample count {} does not match text batch {}".format(
                    len(batch_result.latents),
                    len(text),
                )
            )
        empty_fallback_count += batch_result.empty_fallback_count

        pred_pose_eval, pred_lengths = decode_equal_length_groups(
            net,
            batch_result.latents,
            max_motion_length=pose.shape[1],
            motion_dim=pose.shape[-1],
        )

        text_pred = textencoder(text).loc
        motion_pred = motionencoder(pred_pose_eval, pred_lengths).loc

        pose = pose.to(device).float()
        text_real = textencoder(text).loc
        motion_real = motionencoder(pose, motion_lengths).loc
        motion_annotation_list.append(motion_real)
        motion_pred_list.append(motion_pred)

        batch_r, batch_matching = calculate_R_precision(
            text_real.detach().cpu().numpy(),
            motion_real.detach().cpu().numpy(),
            top_k=3,
            sum_all=True,
        )
        r_precision_real += torch.as_tensor(
            batch_r,
            device=device,
            dtype=r_precision_real.dtype,
        )
        matching_score_real += float(batch_matching)

        batch_r, batch_matching = calculate_R_precision(
            text_pred.detach().cpu().numpy(),
            motion_pred.detach().cpu().numpy(),
            top_k=3,
            sum_all=True,
        )
        r_precision_pred += torch.as_tensor(
            batch_r,
            device=device,
            dtype=r_precision_pred.dtype,
        )
        matching_score_pred += float(batch_matching)
        sample_count += len(text)

    if sample_count == 0:
        raise ValueError("validation loader produced no samples")

    motion_annotation_np = (
        torch.cat(motion_annotation_list, dim=0).detach().cpu().numpy()
    )
    motion_pred_np = torch.cat(motion_pred_list, dim=0).detach().cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    pred_mu, pred_cov = calculate_activation_statistics(motion_pred_np)
    diversity_times = 300 if sample_count > 300 else 100
    diversity_real = calculate_diversity(
        motion_annotation_np,
        diversity_times,
    )
    diversity_pred = calculate_diversity(
        motion_pred_np,
        diversity_times,
    )

    r_precision_real = r_precision_real / sample_count
    r_precision_pred = r_precision_pred / sample_count
    matching_score_real = matching_score_real / sample_count
    matching_score_pred = matching_score_pred / sample_count
    fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

    logger.info(
        "--> \t Eval. :, FID. {:.4f}, Diversity Real. {:.4f}, "
        "Diversity Pred. {:.4f}, R_precision Real. {}, "
        "R_precision Pred. {}, MM-dist (matching_score) Real. {}, "
        "MM-dist (matching_score) Pred. {}, Empty latent fallbacks. {}".format(
            fid,
            diversity_real,
            diversity_pred,
            r_precision_real,
            r_precision_pred,
            matching_score_real,
            matching_score_pred,
            empty_fallback_count,
        )
    )

    return (
        fid,
        diversity_pred,
        r_precision_pred[0],
        r_precision_pred[1],
        r_precision_pred[2],
        matching_score_pred,
        logger,
    )
