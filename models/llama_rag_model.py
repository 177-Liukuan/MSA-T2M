"""RAG conditioning wrapper for MotionStreamer LLaMA backbone."""

import torch
import torch.nn as nn


class LLaMARAGWrapper(nn.Module):
    """Add retrieval-enhanced conditioning on top of the original LLaMAHF backbone.

    Condition sequence layout:
        [t_text, t_ret, motion_token_0, motion_token_1, ...]
    """

    def __init__(self, base_model, text_dim=512, retrieval_dim=512, model_dim=768):
        super().__init__()
        self.base_model = base_model

        # Independent projection layers required by design.
        self.text_proj = nn.Linear(text_dim, model_dim, bias=False)
        self.retrieval_proj = nn.Linear(retrieval_dim, model_dim, bias=False)

        # Learnable null retrieval token for CFG unconditional branch.
        self.null_retrieval_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.num_condition_tokens = 2

    def _fuse_retrieval(self, top3_h_cls, top3_sim_scores):
        """Fuse Top-K retrieved h_cls vectors into one retrieval token.

        Args:
            top3_h_cls: [B, 3, 512]
            top3_sim_scores: [B, 3]
        Returns:
            t_ret: [B, 1, 768]
        """
        # 中文说明：先把检索到的 3 个 h_cls 向量映射到 Transformer 维度。
        retrieval_tokens = self.retrieval_proj(top3_h_cls)  # [B, 3, 768]

        # 中文说明：对相似度做 softmax，得到归一化权重，避免数值不稳定。
        weights = torch.softmax(top3_sim_scores, dim=1).unsqueeze(-1)  # [B, 3, 1]

        # 中文说明：使用加权求和，把 3 个候选先验压缩为单一检索 token。
        t_ret = (retrieval_tokens * weights).sum(dim=1, keepdim=True)  # [B, 1, 768]
        return t_ret

    def _build_condition_tokens(
        self,
        text_emb,
        top3_h_cls,
        top3_sim_scores,
        cfg_drop_mask=None,
        empty_text_emb=None,
    ):
        """Build [t_text, t_ret] condition tokens with optional joint CFG dropout."""
        bsz = text_emb.shape[0]

        t_text = self.text_proj(text_emb).unsqueeze(1)  # [B, 1, 768]
        t_ret = self._fuse_retrieval(top3_h_cls, top3_sim_scores)  # [B, 1, 768]

        if cfg_drop_mask is not None:
            if empty_text_emb is None:
                raise ValueError('empty_text_emb is required when cfg_drop_mask is provided.')

            if empty_text_emb.ndim == 1:
                empty_text_emb = empty_text_emb.unsqueeze(0).expand(bsz, -1)
            elif empty_text_emb.ndim == 2 and empty_text_emb.shape[0] == 1:
                empty_text_emb = empty_text_emb.expand(bsz, -1)

            empty_t_text = self.text_proj(empty_text_emb).unsqueeze(1)  # [B, 1, 768]
            null_ret = self.null_retrieval_token.expand(bsz, -1, -1)   # [B, 1, 768]

            # cfg_drop_mask: [B] -> [B,1,1]
            mask = cfg_drop_mask.view(-1, 1, 1)
            t_text = torch.where(mask, empty_t_text, t_text)
            t_ret = torch.where(mask, null_ret, t_ret)

        cond_tokens = torch.cat([t_text, t_ret], dim=1)  # [B, 2, 768]
        return cond_tokens

    def forward(
        self,
        motion_latents,
        text_emb,
        top3_h_cls,
        top3_sim_scores,
        cfg_drop_mask=None,
        empty_text_emb=None,
    ):
        """Forward with retrieval-augmented condition tokens.

        Args:
            motion_latents: [B, T, latent_dim]
            text_emb: [B, 512]
            top3_h_cls: [B, 3, 512]
            top3_sim_scores: [B, 3]
        Returns:
            hidden_states: [B, T+2, 768]
        """
        cond_tokens = self._build_condition_tokens(
            text_emb,
            top3_h_cls,
            top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )

        motion_tokens = self.base_model.transformer.wte(motion_latents.float())  # [B, T, 768]

        # 中文说明：最终序列严格按 [文本 token, 检索 token, 运动 token...] 拼接。
        # 注意这里输入给因果注意力的总长度是 2+T，因此底层 is_causal=True
        # 会自动生成对应大小的下三角掩码，确保每个位置只能看见过去位置，
        # 也即不会访问未来运动 token，从而避免信息泄露。
        x = torch.cat([cond_tokens, motion_tokens], dim=1)  # [B, 2+T, 768]

        for block in self.base_model.transformer.h:
            x = block(x)
        x = self.base_model.transformer.ln_f(x)
        logits = self.base_model.out_proj(x)
        return logits

    def motion_condition_slice(self, hidden_states, motion_len):
        """Map hidden states to diffusion conditions aligned with motion targets.

        With two condition tokens [t_text, t_ret], prediction alignment is:
            target motion token 0 <- hidden state at index 1 (t_ret)
            target motion token i <- hidden state at index i+1
        So we take hidden_states[:, 1:1+motion_len, :].
        """
        start = self.num_condition_tokens - 1
        end = start + motion_len
        return hidden_states[:, start:end, :]

    @torch.no_grad()
    def sample_next_with_cfg(
        self,
        motion_prefix,
        text_emb,
        top3_h_cls,
        top3_sim_scores,
        empty_text_emb,
        cfg_scale=4.0,
        temperature=1.0,
    ):
        """Sample one next latent token with explicit cond/uncond CFG setup.

        This method prepares the two branches to match:
            x0_hat = G(empty, null_ret) + s * (G(text, t_ret) - G(empty, null_ret))
        """
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
