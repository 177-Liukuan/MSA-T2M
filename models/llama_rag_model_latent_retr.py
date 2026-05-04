"""Local-RAG Latent Retrieval extension for LLaMARAG Wrapper.

Same Flamingo-style CA block architecture as llama_rag_model_mca.py, but the
cross-attention KV comes from RETRIEVED MOTION LATENTS (shape F x latent_dim)
rather than word-level T5 tokens.

Projection:  nn.Linear(latent_dim=16, model_dim=768, bias=False)
             (shared for all CA blocks)

Two variants (matching MCA counterparts):
  LLaMARAGLatentRetrWrapper         — gate-free, zero-init out_proj (Branch A)
  LLaMARAGLatentRetrGatedWrapper    — tanh-gated Flamingo (Branch B)

External forward signature:
    model(motion_latents, text_emb,
          top3_h_cls, top3_sim_scores,
          cfg_drop_mask, empty_text_emb,
          retr_latents, retr_latent_lens)
  where:
    retr_latents     (B, L_max, latent_dim) — retrieved & concatenated latents
    retr_latent_lens (B,)  int64            — valid frame counts (for padding mask)
"""

import torch
import torch.nn as nn

from models.llama_rag_model import LLaMARAGWrapper
from models.llama_rag_model_mca import (
    TextCrossAttentionBlock,
    GatedCrossAttentionBlock,
)


class LLaMARAGLatentRetrWrapper(LLaMARAGWrapper):
    """LLaMARAGWrapper + Flamingo-style XATTN-DENSE on retrieved motion latents.

    Parameters
    ----------
    base_model        : LLaMAHF instance
    model_dim         : Transformer hidden dim (default 768)
    retrieval_dim     : h_cls embedding dim    (default 768)
    disable_rag       : ablation flag – remove global h_cls prefix
    latent_dim        : motion latent dim      (default 16)
    ca_n_layers       : number of CA blocks to insert
    ca_every_n_layers : override insertion interval directly
    ca_n_head         : CA heads; None = same as backbone
    ff_mult           : FFN expansion factor in each CA block
    ca_block_cls      : block class (TextCrossAttentionBlock or Gated variant)
    disable_latent_retr : ablation – skip local latent CA entirely
    """

    def __init__(
        self,
        base_model,
        model_dim: int = 768,
        retrieval_dim: int = 768,
        disable_rag: bool = False,
        latent_dim: int = 16,
        ca_n_layers: int = 6,
        ca_every_n_layers: int = None,
        ca_n_head: int = None,
        ff_mult: int = 2,
        ca_block_cls=None,
        disable_latent_retr: bool = False,
        ca_insertion_mode: str = 'before_sa',
    ) -> None:
        # ca_insertion_mode:
#   'before_sa'    (A, default) — CA runs BEFORE the SA block  (original Flamingo order)
#   'after_sa'     (B)          — CA runs AFTER  the SA block  (text-first, retr-refine)
#   'late_after_sa'(C)          — CA runs AFTER SA, only in the second half of layers
#                                 (first n_total//2 layers are pure SA)
        super().__init__(base_model, model_dim, retrieval_dim, disable_rag)

        if ca_block_cls is None:
            ca_block_cls = TextCrossAttentionBlock

        self.disable_latent_retr = disable_latent_retr

        n_total = len(base_model.transformer.h)

        # Insertion schedule
        if ca_every_n_layers is not None:
            every_n = max(1, ca_every_n_layers)
        else:
            every_n = max(1, n_total // ca_n_layers)

        # Insertion schedule: for 'late_after_sa', CA is restricted to the second half.
        _ca_start = (n_total // 2) if ca_insertion_mode == 'late_after_sa' else 0
        self.ca_layer_indices = [
            idx for idx in range(_ca_start, n_total)
            if (idx - _ca_start + 1) % every_n == 0
        ]
        n_ca = len(self.ca_layer_indices)
        self.ca_every_n_layers = every_n
        self.ca_insertion_mode = ca_insertion_mode  # 'before_sa' | 'after_sa' | 'late_after_sa'

        if ca_n_head is None:
            ca_n_head = base_model.config.n_head

        # Project latent_dim → model_dim (shared by all CA blocks)
        self.latent_retr_proj = nn.Linear(latent_dim, model_dim, bias=False)

        # One CA block per insertion point
        self.ca_blocks = nn.ModuleList(
            [ca_block_cls(model_dim, ca_n_head, ff_mult) for _ in range(n_ca)]
        )

        # Learnable null KV for CFG dropout (1 token, model_dim)
        self.null_latent_kv = nn.Parameter(torch.zeros(1, 1, model_dim))

        # Fast lookup: layer_index -> ca_block_index
        self._ca_index_map = {li: ci for ci, li in enumerate(self.ca_layer_indices)}

    def extra_repr(self) -> str:
        n_total = len(self.base_model.transformer.h)
        return (
            f"ca_insertion_mode={self.ca_insertion_mode}, "
            f"ca_every_n_layers={self.ca_every_n_layers}, "
            f"n_ca_blocks={len(self.ca_blocks)}, "
            f"n_total_layers={n_total}, "
            f"ca_at_layers={self.ca_layer_indices}, "
            f"disable_latent_retr={self.disable_latent_retr}"
        )

    # ------------------------------------------------------------------
    #  Helper: project retrieved latents → CA KV + build padding mask
    # ------------------------------------------------------------------

    def _prepare_latent_kv(self, retr_latents, retr_latent_lens,
                            retr_drop_mask, bsz):
        """Return (latent_kv, latent_key_padding_mask) or (None, None).

        retr_drop_mask is INDEPENDENT of the text cfg_drop_mask so the model
        can learn to use each condition separately.
        """
        if retr_latents is None or self.disable_latent_retr:
            return None, None

        latent_kv = self.latent_retr_proj(retr_latents)   # (B, L, D)
        L = latent_kv.size(1)
        device = latent_kv.device

        # Padding mask: True = invalid (padded) position
        if retr_latent_lens is not None:
            idx = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
            latent_key_padding_mask = idx >= retr_latent_lens.unsqueeze(1)  # (B, L)
        else:
            latent_key_padding_mask = None

        # Independent retrieval dropout: replace dropped samples with null token
        if retr_drop_mask is not None and retr_drop_mask.any():
            null_kv = self.null_latent_kv.expand(bsz, L, -1)
            drop3d  = retr_drop_mask.view(-1, 1, 1)
            latent_kv = torch.where(drop3d, null_kv, latent_kv)
            if latent_key_padding_mask is not None:
                latent_key_padding_mask = latent_key_padding_mask.clone()
                latent_key_padding_mask[retr_drop_mask] = False

        return latent_kv, latent_key_padding_mask

    # ------------------------------------------------------------------
    #  Forward  (Flamingo insertion: CA block → LM block)
    # ------------------------------------------------------------------

    def forward(
        self,
        motion_latents,
        text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_drop_mask=None,         # controls text + h_cls condition
        empty_text_emb=None,
        retr_latents=None,          # (B, L_max, latent_dim)
        retr_latent_lens=None,      # (B,) int64
        retr_cfg_drop_mask=None,    # INDEPENDENT retrieval dropout mask;
                                    # None = fall back to cfg_drop_mask
    ):
        bsz = motion_latents.size(0)

        # Global condition prefix (h_cls + text emb)
        cond_tokens = self._build_condition_tokens(
            text_emb, top3_h_cls, top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )
        motion_tokens = self.base_model.transformer.wte(motion_latents.float())
        x = torch.cat([cond_tokens, motion_tokens], dim=1)

        # Use dedicated retrieval drop mask; fall back to text mask if not provided
        effective_retr_drop = (
            retr_cfg_drop_mask if retr_cfg_drop_mask is not None else cfg_drop_mask
        )
        # Prepare local CA KV
        latent_kv, latent_key_padding_mask = self._prepare_latent_kv(
            retr_latents, retr_latent_lens, effective_retr_drop, bsz
        )

        # Interleaved forward: insertion mode controls CA/SA ordering.
        for layer_idx, block in enumerate(self.base_model.transformer.h):
            if self.ca_insertion_mode == 'before_sa':
                # Mode A (default): CA → SA  (original Flamingo order)
                if latent_kv is not None and layer_idx in self._ca_index_map:
                    ca_block = self.ca_blocks[self._ca_index_map[layer_idx]]
                    x = ca_block(x, latent_kv, latent_key_padding_mask)
                x = block(x)
            else:
                # Mode B/C: SA → CA  (text self-attention first, then retrieval refinement)
                x = block(x)
                if latent_kv is not None and layer_idx in self._ca_index_map:
                    ca_block = self.ca_blocks[self._ca_index_map[layer_idx]]
                    x = ca_block(x, latent_kv, latent_key_padding_mask)

        x = self.base_model.transformer.ln_f(x)
        logits = self.base_model.out_proj(x)
        return logits

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_next_with_cfg(
        self,
        motion_prefix,
        text_emb,
        empty_text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_scale=4.0,
        cfg_scale_retr=1.0,
        temperature=1.0,
        retr_latents=None,
        retr_latent_lens=None,
    ):
        """Autoregressive next-token sampling with classifier-free guidance.

        When retrieved motion latents are present, uses **dual CFG** (3 forwards):
          z_guided = z_none
                   + cfg_scale      * (z_both - z_retr)   # text guidance given retrieval
                   + cfg_scale_retr * (z_retr - z_none)   # retrieval guidance over baseline

        When no retrieval is provided, falls back to standard 2-forward text CFG.

        Args:
            cfg_scale:      Text CFG scale  (s_text, default 4.0).
            cfg_scale_retr: Retrieval CFG scale (s_retr, default 1.0).
                            1.0 => mild retrieval prior; 0.0 => text CFG conditioned on retr
                            only; >1.0 => stronger retrieval amplification.
        """
        bsz = text_emb.shape[0]
        device = motion_prefix.device
        cond_mask   = torch.zeros(bsz, dtype=torch.bool, device=device)  # False = real
        uncond_mask = torch.ones(bsz, dtype=torch.bool, device=device)   # True  = null

        # Forward 1: f(text, retr) -- both conditions real
        z_both = self.forward(
            motion_prefix, text_emb, top3_h_cls, top3_sim_scores,
            cfg_drop_mask=cond_mask, empty_text_emb=empty_text_emb,
            retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
            retr_cfg_drop_mask=cond_mask,
        )[:, -1, :]

        # Forward 2: f(null_text, null_retr) -- unconditional baseline
        z_none = self.forward(
            motion_prefix, text_emb, top3_h_cls, top3_sim_scores,
            cfg_drop_mask=uncond_mask, empty_text_emb=empty_text_emb,
            retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
            retr_cfg_drop_mask=uncond_mask,
        )[:, -1, :]

        if retr_latents is not None:
            # Dual CFG: Forward 3 -- f(null_text, retr): null text, real retrieval.
            # Isolates the retrieval's marginal contribution separately from text.
            z_retr = self.forward(
                motion_prefix, text_emb, top3_h_cls, top3_sim_scores,
                cfg_drop_mask=uncond_mask, empty_text_emb=empty_text_emb,
                retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
                retr_cfg_drop_mask=cond_mask,   # keep real retrieval, null text
            )[:, -1, :]

            # Compose guidance in hidden space; diff_loss.sample(cfg=1.0) samples
            # directly from the pre-composed condition (B, D) without further CFG.
            z_guided = (z_none
                        + cfg_scale      * (z_both - z_retr)
                        + cfg_scale_retr * (z_retr - z_none))
            return self.base_model.diff_loss.sample(
                z_guided, temperature=temperature, cfg=1.0
            )

        # No retrieval -- standard 2-forward text CFG
        mix_hidden = torch.cat([z_both, z_none], dim=0)
        sampled = self.base_model.diff_loss.sample(
            mix_hidden, temperature=temperature, cfg=cfg_scale
        )
        sampled_cond, _ = (sampled.chunk(2, dim=0)
                           if cfg_scale != 1.0 else (sampled, None))
        return sampled_cond


# ---------------------------------------------------------------------------
#  Branch B: Flamingo gated (convenience subclass)
# ---------------------------------------------------------------------------

class LLaMARAGLatentRetrGatedWrapper(LLaMARAGLatentRetrWrapper):
    """Convenience subclass using GatedCrossAttentionBlock (Branch B)."""

    def __init__(self, *args, **kwargs):
        kwargs['ca_block_cls'] = GatedCrossAttentionBlock
        super().__init__(*args, **kwargs)
