"""RAG conditioning wrapper for MotionStreamer LLaMA backbone."""

import torch
import torch.nn as nn


class LocalRAGCrossAttn(nn.Module):
    """Lightweight cross-attention to aggregate local RAG mu frames into L_local tokens.

    Uses DETR-style learned position queries biased by text, attending over all K*T_max
    retrieved mu frames with per-position independent attention distributions.
    Produces L_local aggregated tokens in model space.
    """

    def __init__(self, model_dim=768, local_rag_dim=16, L_local=4, num_heads=4, add_selfatten=False):
        super().__init__()
        self.L_local = L_local
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.add_selfatten = bool(add_selfatten)

        # L_local learnable position queries (DETR-style)
        self.learned_queries = nn.Parameter(torch.randn(1, L_local, model_dim) * 0.02)
        # Text bias: maps text feat -> L_local query offsets
        self.text_to_query_bias = nn.Linear(model_dim, L_local * model_dim)
        # Project z latents (motion latents) (low-dim) into model space for K/V
        self.kv_proj = nn.Linear(local_rag_dim, model_dim)
        # Optional: TransformerEncoder for self-attention over z frames (captures inter-frame dependency)
        if self.add_selfatten:
            sa_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,  # Pre-LN for training stability
            )
            self.z_seq_encoder = nn.TransformerEncoder(sa_layer, num_layers=2)
        else:
            self.z_seq_encoder = None
        # Multi-head attention projections
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, text_feat, top_z_seqs, top_z_lens=None):
        """
        Args:
            text_feat:    [B, model_dim]  -- raw text embedding (before projection)
            top_z_seqs:  [B, K, T_max, local_rag_dim]  -- padded full z sequences
            top_z_lens:  [B, K]  -- valid frame counts per retrieved motion (masking)
        Returns:
            t_local: [B, L_local, model_dim]
        """
        B, K, T_max, D = top_z_seqs.shape

        # Build queries: learned base + text-conditional bias
        q_base = self.learned_queries.expand(B, -1, -1)                          # [B, L_local, D]
        text_bias = self.text_to_query_bias(text_feat).view(B, self.L_local, self.model_dim)
        queries = q_base + text_bias                                              # [B, L_local, D]

        # Padding mask: True = position should be ignored in attention
        if top_z_lens is not None:
            frame_idx = torch.arange(T_max, device=top_z_seqs.device)           # [T_max]
            valid = frame_idx[None, None, :] < top_z_lens.unsqueeze(-1)         # [B, K, T_max]
            attn_mask = ~valid.reshape(B, K * T_max)                             # [B, K*T_max]
        else:
            attn_mask = None

        # Build K/V from all retrieved frames flattened over K and T_max
        kv_input = top_z_seqs.reshape(B, K * T_max, D)                          # [B, K*T_max, D_z]
        kv = self.kv_proj(kv_input)                                               # [B, K*T_max, D_model]

        # Optional self-attention encoding: capture inter-frame dependencies before cross-attn
        if self.z_seq_encoder is not None:
            kv = self.z_seq_encoder(kv, src_key_padding_mask=attn_mask)         # [B, K*T_max, D_model]

        # Multi-head attention
        Q = self.q_proj(queries)   # [B, L_local, D]
        K_ = self.k_proj(kv)       # [B, K*T_max, D]
        V_ = self.v_proj(kv)       # [B, K*T_max, D]

        def split_heads(x):
            Bx, S, Dx = x.shape
            return x.view(Bx, S, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K_, V_ = split_heads(Q), split_heads(K_), split_heads(V_)
        # Q: [B, h, L_local, head_dim]   K_/V_: [B, h, K*T_max, head_dim]

        attn = torch.matmul(Q, K_.transpose(-2, -1)) * self.scale                # [B, h, L_local, K*T_max]
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[:, None, None, :], float('-inf'))  # broadcast over h, L_local
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V_)                                              # [B, h, L_local, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, self.L_local, self.model_dim)
        return self.out_proj(out)


class LLaMARAGWrapper(nn.Module):
    """Add retrieval-enhanced conditioning on top of the original LLaMAHF backbone.

    Condition sequence layout:
        RAG mode (no local):   [t_text, t_global, motion_0, motion_1, ...]
        RAG mode (with local): [t_text, t_global, t_local_0, ..., t_local_{L-1}, motion_0, ...]
        no-RAG ablation mode:  [t_text, motion_0, motion_1, ...]

    Local RAG tokens are produced by LocalRAGCrossAttn: L_local learned queries
    (biased by text) cross-attending over ALL K*T_max retrieved mu frames.
    """

    def __init__(
        self,
        base_model,
        model_dim=768,
        retrieval_dim=768,
        disable_rag=False,
        L_local=0,
        local_rag_dim=16,
        add_selfatten=False,
    ):
        super().__init__()
        self.base_model = base_model
        self.disable_rag = bool(disable_rag)
        self.L_local = int(L_local)

        self.retrieval_embed = nn.Linear(retrieval_dim, model_dim)

        if retrieval_dim == model_dim:
            nn.init.eye_(self.retrieval_embed.weight)
            if self.retrieval_embed.bias is not None:
                nn.init.zeros_(self.retrieval_embed.bias)

        # Learnable null retrieval token for CFG unconditional branch (global RAG).
        self.null_retrieval_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Local RAG: cross-attention aggregator from mu sequences.
        if self.L_local > 0 and not self.disable_rag:
            self.local_cross_attn = LocalRAGCrossAttn(
                model_dim, local_rag_dim, self.L_local, add_selfatten=add_selfatten
            )
            # Null local token for CFG dropout (shape matches L_local output tokens).
            self.null_local_rag_token = nn.Parameter(torch.zeros(1, self.L_local, model_dim))

        # num_condition_tokens counts all prefix tokens before motion sequence.
        if self.disable_rag:
            self.num_condition_tokens = 1                         # [t_text]
        elif self.L_local > 0:
            self.num_condition_tokens = 2 + self.L_local         # [t_text, t_global, t_local x L]
        else:
            self.num_condition_tokens = 2                         # [t_text, t_global]

    def _fuse_retrieval(self, top3_h_cls, top3_sim_scores):
        """Fuse Top-K retrieved h_cls vectors into one global retrieval token.

        Args:
            top3_h_cls:      [B, K, D_text]
            top3_sim_scores: [B, K]
        Returns:
            t_ret: [B, 1, D_model]
        """
        projected_h_cls = self.retrieval_embed(top3_h_cls)                       # [B, K, model_dim]
        weights = torch.softmax(top3_sim_scores, dim=1).unsqueeze(-1)            # [B, K, 1]
        t_ret = (projected_h_cls * weights).sum(dim=1, keepdim=True)             # [B, 1, model_dim]
        return t_ret

    def _build_condition_tokens(
        self,
        text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_drop_mask=None,
        empty_text_emb=None,
        top_z_seqs=None,
        top_z_lens=None,
    ):
        """Build condition tokens with optional joint CFG dropout.

        Layout:
            no-RAG:        [t_text]
            RAG+no-local:  [t_text, t_global]
            RAG+local:     [t_text, t_global, t_local_0, ..., t_local_{L-1}]

        All conditions share the same CFG dropout mask (joint dropout).

        Args:
            text_emb:        [B, D_text]  -- raw text embedding
            top_z_seqs:     [B, K, T_max, local_rag_dim]  -- padded z sequences
            top_z_lens:     [B, K]  -- valid frame counts per slot
        """
        bsz = text_emb.shape[0]

        t_text = self.base_model.transformer.cond_embed(text_emb).unsqueeze(1)  # [B, 1, D_model]
        t_global = None
        t_local = None

        if not self.disable_rag:
            if top3_h_cls is None or top3_sim_scores is None:
                raise ValueError('top3_h_cls and top3_sim_scores are required when disable_rag=False.')
            t_global = self._fuse_retrieval(top3_h_cls, top3_sim_scores)         # [B, 1, D_model]

            if self.L_local > 0:
                if top_z_seqs is None:
                    raise ValueError('top_z_seqs is required when L_local > 0.')
                # Cross-attend over K*T_max retrieved mu frames using text-biased queries.
                t_local = self.local_cross_attn(
                    text_emb, top_z_seqs, top_z_lens
                )                                                                 # [B, L_local, D_model]

        if cfg_drop_mask is not None:
            if empty_text_emb is None:
                raise ValueError('empty_text_emb is required when cfg_drop_mask is provided.')
            if empty_text_emb.ndim == 1:
                empty_text_emb = empty_text_emb.unsqueeze(0).expand(bsz, -1)
            elif empty_text_emb.ndim == 2 and empty_text_emb.shape[0] == 1:
                empty_text_emb = empty_text_emb.expand(bsz, -1)

            empty_t_text = self.base_model.transformer.cond_embed(empty_text_emb).unsqueeze(1)

            mask = cfg_drop_mask.view(-1, 1, 1)                                  # [B, 1, 1]
            t_text = torch.where(mask, empty_t_text, t_text)

            if not self.disable_rag:
                null_global = self.null_retrieval_token.expand(bsz, -1, -1)
                t_global = torch.where(mask, null_global, t_global)

                if self.L_local > 0:
                    null_local = self.null_local_rag_token.expand(bsz, -1, -1)
                    t_local = torch.where(mask, null_local, t_local)

        if self.disable_rag:
            cond_tokens = t_text                                                   # [B, 1, D]
        elif self.L_local > 0:
            cond_tokens = torch.cat([t_text, t_global, t_local], dim=1)           # [B, 2+L, D]
        else:
            cond_tokens = torch.cat([t_text, t_global], dim=1)                    # [B, 2, D]

        return cond_tokens

    def forward(
        self,
        motion_latents,
        text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_drop_mask=None,
        empty_text_emb=None,
        top_z_seqs=None,
        top_z_lens=None,
    ):
        """Forward with condition tokens.

        Args:
            motion_latents:  [B, T, latent_dim]
            text_emb:        [B, D_text]
            top3_h_cls:      [B, K, D_text], optional when disable_rag=True
            top3_sim_scores: [B, K], optional when disable_rag=True
            top_z_seqs:     [B, K, T_max, local_rag_dim], optional when L_local=0
            top_z_lens:     [B, K] int tensor, valid frame counts
        Returns:
            hidden_states: [B, T+num_condition_tokens, D_model]
        """
        cond_tokens = self._build_condition_tokens(
            text_emb,
            top3_h_cls,
            top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
            top_z_seqs=top_z_seqs,
            top_z_lens=top_z_lens,
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
        top_z_seqs=None,
        top_z_lens=None,
        cfg_scale=4.0,
        temperature=1.0,
    ):
        """Sample one next latent token using a single batched cond+uncond forward pass.

        Optimization: concatenate inputs along batch dimension and run a single forward
        pass instead of two separate passes. This halves transformer FLOPs per AR step.
        """
        bsz = text_emb.shape[0]

        # Build combined mask: first bsz = conditioned (False), second bsz = unconditioned (True)
        double_mask = torch.cat([
            torch.zeros(bsz, dtype=torch.bool, device=motion_prefix.device),
            torch.ones(bsz, dtype=torch.bool, device=motion_prefix.device),
        ])

        def _double(x):
            return torch.cat([x, x], dim=0) if x is not None else None

        # Single forward pass with doubled batch: [2*bsz, ...]
        mix_hidden = self.forward(
            _double(motion_prefix),
            _double(text_emb),
            _double(top3_h_cls),
            _double(top3_sim_scores),
            cfg_drop_mask=double_mask,
            empty_text_emb=empty_text_emb,
            top_z_seqs=_double(top_z_seqs),
            top_z_lens=_double(top_z_lens),
        )[:, -1, :]  # [2*bsz, D_model]: first bsz=cond, second bsz=uncond

        sampled = self.base_model.diff_loss.sample(mix_hidden, temperature=temperature, cfg=cfg_scale)
        if cfg_scale != 1.0:
            sampled_cond, _ = sampled.chunk(2, dim=0)
        else:
            sampled_cond = sampled

        return sampled_cond
