"""RAG conditioning wrapper for MotionStreamer LLaMA backbone."""

import torch
import torch.nn as nn


class LLaMARAGWrapper(nn.Module):
    """Add retrieval-enhanced conditioning on top of the original LLaMAHF backbone.

    Condition sequence layout:
        RAG mode: [t_text, t_ret, motion_token_0, motion_token_1, ...]
        no-RAG ablation mode: [t_text, motion_token_0, motion_token_1, ...]
    """

    def __init__(self, base_model, model_dim=768, retrieval_dim=768, disable_rag=False):
        super().__init__()
        self.base_model = base_model
        # Ablation switch: disable retrieval token and fall back to text-only conditioning.
        self.disable_rag = bool(disable_rag)

        self.retrieval_embed = nn.Linear(retrieval_dim, model_dim)

        if retrieval_dim == model_dim:
            nn.init.eye_(self.retrieval_embed.weight)
            if self.retrieval_embed.bias is not None:
                nn.init.zeros_(self.retrieval_embed.bias)

        # Keep only a learnable null retrieval token for CFG branch.
        self.null_retrieval_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.num_condition_tokens = 1 if self.disable_rag else 2

    def _fuse_retrieval(self, top3_h_cls, top3_sim_scores):
        """Fuse Top-K retrieved h_cls vectors into one retrieval token.

        Args:
            top3_h_cls: [B, K, D_text]
            top3_sim_scores: [B, K]
        Returns:
            t_ret: [B, 1, D_model]
        """
        projected_h_cls = self.retrieval_embed(top3_h_cls)  # [B, K, model_dim]
        weights = torch.softmax(top3_sim_scores, dim=1).unsqueeze(-1)  # [B, K, 1]
        t_ret = (projected_h_cls * weights).sum(dim=1, keepdim=True)  # [B, 1, model_dim]
        return t_ret

    def _build_condition_tokens(
        self,
        text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_drop_mask=None,
        empty_text_emb=None,
    ):
        """Build condition tokens with optional joint CFG dropout.

        In full mode, build [t_text, t_ret].
        In no-RAG ablation mode, build [t_text] only.
        """
        bsz = text_emb.shape[0]

        t_text = self.base_model.transformer.cond_embed(text_emb).unsqueeze(1)  # [B, 1, D_model]
        t_ret = None

        if not self.disable_rag:
            if top3_h_cls is None or top3_sim_scores is None:
                raise ValueError('top3_h_cls and top3_sim_scores are required when disable_rag=False.')
            t_ret = self._fuse_retrieval(top3_h_cls, top3_sim_scores)  # [B, 1, D_model]

        if cfg_drop_mask is not None:
            if empty_text_emb is None:
                raise ValueError('empty_text_emb is required when cfg_drop_mask is provided.')
            if empty_text_emb.ndim == 1:
                empty_text_emb = empty_text_emb.unsqueeze(0).expand(bsz, -1)
            elif empty_text_emb.ndim == 2 and empty_text_emb.shape[0] == 1:
                empty_text_emb = empty_text_emb.expand(bsz, -1)

            empty_t_text = self.base_model.transformer.cond_embed(empty_text_emb).unsqueeze(1)

            mask = cfg_drop_mask.view(-1, 1, 1)
            t_text = torch.where(mask, empty_t_text, t_text)

            if not self.disable_rag:
                null_ret = self.null_retrieval_token.expand(bsz, -1, -1)
                t_ret = torch.where(mask, null_ret, t_ret)

        if self.disable_rag:
            cond_tokens = t_text  # [B, 1, D_model]
        else:
            cond_tokens = torch.cat([t_text, t_ret], dim=1)  # [B, 2, D_model]

        return cond_tokens

    def forward(
        self,
        motion_latents,
        text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_drop_mask=None,
        empty_text_emb=None,
    ):
        """Forward with condition tokens.

        Args:
            motion_latents: [B, T, latent_dim]
            text_emb: [B, D_text]
            top3_h_cls: [B, K, D_text], optional when disable_rag=True
            top3_sim_scores: [B, K], optional when disable_rag=True
        Returns:
            hidden_states: [B, T+2, D_model] in RAG mode, [B, T+1, D_model] in no-RAG mode
        """
        cond_tokens = self._build_condition_tokens(
            text_emb,
            top3_h_cls,
            top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )

        motion_tokens = self.base_model.transformer.wte(motion_latents.float())
        x = torch.cat([cond_tokens, motion_tokens], dim=1)

        for block in self.base_model.transformer.h:
            x = block(x)
        x = self.base_model.transformer.ln_f(x)
        logits = self.base_model.out_proj(x)
        return logits

    def motion_condition_slice(self, hidden_states, motion_len):
        """Map hidden states to diffusion conditions aligned with motion targets."""
        start = self.num_condition_tokens - 1
        end = start + motion_len
        return hidden_states[:, start:end, :]

    @torch.no_grad()
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
        """Sample one next latent token with explicit cond/uncond CFG setup."""
        bsz = text_emb.shape[0]
        cond_mask = torch.zeros(bsz, dtype=torch.bool, device=motion_prefix.device)
        uncond_mask = torch.ones(bsz, dtype=torch.bool, device=motion_prefix.device)

        cond_hidden = self.forward(
            motion_prefix,
            text_emb,
            top3_h_cls,
            top3_sim_scores,
            cfg_drop_mask=cond_mask,
            empty_text_emb=empty_text_emb,
        )[:, -1, :]

        uncond_hidden = self.forward(
            motion_prefix,
            text_emb,
            top3_h_cls,
            top3_sim_scores,
            cfg_drop_mask=uncond_mask,
            empty_text_emb=empty_text_emb,
        )[:, -1, :]

        mix_hidden = torch.cat([cond_hidden, uncond_hidden], dim=0)
        sampled = self.base_model.diff_loss.sample(mix_hidden, temperature=temperature, cfg=cfg_scale)
        if cfg_scale != 1.0:
            sampled_cond, _ = sampled.chunk(2, dim=0)
        else:
            sampled_cond = sampled

        return sampled_cond
