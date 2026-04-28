"""Multi-Text Cross-Attention (MCA) extension for LLaMARAG Wrapper.

Two cross-attention block variants are provided; choose via the wrapper class:

  ┌────────────────────────────────────────────────────────────────────────┐
  │ Branch A: Gate-free   (TextCrossAttentionBlock)                        │
  │   Use: LLaMARAGMultiTextCAWrapper  (default)                          │
  │   Output projections zero-initialised → residual starts at 0 and      │
  │   grows freely with gradients. Avoids gate self-suppression when       │
  │   backbone is frozen.                                                  │
  │   x = x + out_proj_zero(CA(RMSNorm(x), K, V))                        │
  │   x = x + ff_out_zero(FFN(RMSNorm(x)))                               │
  └────────────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Branch B: Flamingo gated  (GatedCrossAttentionBlock)                   │
  │   Use: LLaMARAGMultiTextCAGatedWrapper                                 │
  │   Dual tanh gates (attn_gate, ff_gate) initialised to 0.              │
  │   Compatible with net_Iter*.pth checkpoints that contain these keys.  │
  │   x = x + tanh(attn_gate) * CA(RMSNorm(x), K, V)                    │
  │   x = x + tanh(ff_gate)   * FFN(RMSNorm(x))                         │
  └────────────────────────────────────────────────────────────────────────┘

Common design:
  - Insertion schedule: CA blocks at fixed intervals (every ca_every_n_layers).
  - Insertion order: CA block runs BEFORE the corresponding LM block.
  - Q/K/V inside nn.MultiheadAttention use default Kaiming-uniform init.
  - out_proj / ff_out_proj: zero-init (Branch A) or gate-controlled (Branch B).

None of the original files (llama_model.py, llama_rag_model.py,
train_t2m_rag.py, msa_gen_motion.py …) are touched.
"""

import math
import torch
import torch.nn as nn

from models.llama_rag_model import LLaMARAGWrapper
from models.llama_model import RMSNorm



# ---------------------------------------------------------------------------
#  Flamingo Gated Cross-Attention + Dense block  (Branch B)
# ---------------------------------------------------------------------------

class GatedCrossAttentionBlock(nn.Module):
    """Flamingo-style GATED XATTN-DENSE block with dual tanh gates.

    Sub-layer 1: x = x + tanh(attn_gate) * CA(RMSNorm(x), K, V)
    Sub-layer 2: x = x + tanh(ff_gate)   * FFN(RMSNorm(x))

    Both gates are nn.Parameter initialised to 0 → tanh(0) = 0, so the block
    contributes nothing at training start, identical to the zero-init approach
    in Branch A.  The difference is that gates are scalar multipliers that can
    be suppressed by gradient pressure (observed when backbone is frozen).

    Checkpoint compatibility: these are the ``attn_gate`` / ``ff_gate`` keys
    stored in net_Iter*.pth checkpoints produced by the Flamingo training run.
    """

    def __init__(self, n_embd: int, n_head: int, ff_mult: int = 2) -> None:
        super().__init__()

        # --- Cross-attention sub-layer ---
        self.norm_q = RMSNorm(n_embd)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            batch_first=True,
            bias=False,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))   # gate 1: tanh(0)=0

        # --- Feed-forward / Dense sub-layer ---
        # ff_mult=2: inner_dim=1536, lighter than backbone SwiGLU (inner=2048).
        inner_dim = int(n_embd * ff_mult)
        self.norm_ff = RMSNorm(n_embd)
        self.ff_in_proj = nn.Linear(n_embd, inner_dim, bias=False)
        self.ff_act = nn.GELU()
        self.ff_out_proj = nn.Linear(inner_dim, n_embd, bias=False)
        self.ff_gate = nn.Parameter(torch.zeros(1))     # gate 2: tanh(0)=0

    def forward(
        self,
        x: torch.Tensor,
        text_kv: torch.Tensor,
        text_key_padding_mask=None,
    ) -> torch.Tensor:
        # Sub-layer 1: gated cross-attention
        q = self.norm_q(x)
        ca_out, _ = self.cross_attn(
            query=q,
            key=text_kv,
            value=text_kv,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_gate.tanh() * ca_out

        # Sub-layer 2: gated feed-forward
        x = x + self.ff_gate.tanh() * self.ff_out_proj(
            self.ff_act(self.ff_in_proj(self.norm_ff(x)))
        )
        return x


# ---------------------------------------------------------------------------
#  Plain Cross-Attention + Dense block  (gate-free, zero-init out_proj)
# ---------------------------------------------------------------------------

class TextCrossAttentionBlock(nn.Module):
    """Plain XATTN-DENSE block without gating.

    Zero contribution at training start is achieved by zero-initialising the
    output projections (out_proj, ff_out_proj) rather than with explicit gate
    parameters.  This allows gradients to grow the contribution freely without
    the self-suppression that tanh-gates exhibit when the backbone is frozen.

    Sub-layer 1 – Plain Cross-Attention
      Q = RMSNorm(x)
      K, V = text_kv  (projected word-level T5 token embeddings)
      x = x + out_proj_zero(CA(Q, K, V))      # out_proj initialised to zeros

    Sub-layer 2 – Feed-Forward (Dense)
      x = x + ff_out_proj_zero(FFN(RMSNorm(x)))  # ff_out_proj initialised to zeros

    Q/K/V projections inside nn.MultiheadAttention use default Kaiming/Xavier init.
    """

    def __init__(self, n_embd: int, n_head: int, ff_mult: int = 2) -> None:
        super().__init__()

        # --- Cross-attention sub-layer ---
        self.norm_q = RMSNorm(n_embd)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            batch_first=True,
            bias=False,
        )
        # Zero-init MHA's internal out_proj directly: avoids a redundant
        # extra (n_embd×n_embd) matmul while achieving the same
        # "zero contribution at training start" guarantee.
        nn.init.zeros_(self.cross_attn.out_proj.weight)

        # --- Feed-forward / Dense sub-layer ---
        # ff_mult=2: inner_dim=1536, lighter than backbone SwiGLU (inner=2048).
        # ff_out_proj zero-init → zero contribution at training start.
        inner_dim = int(n_embd * ff_mult)
        self.norm_ff = RMSNorm(n_embd)
        self.ff_in_proj = nn.Linear(n_embd, inner_dim, bias=False)   # Kaiming (default)
        self.ff_act = nn.GELU()
        self.ff_out_proj = nn.Linear(inner_dim, n_embd, bias=False)
        nn.init.zeros_(self.ff_out_proj.weight)                       # zero-init

    def forward(
        self,
        x: torch.Tensor,              # (B, T, D)  – motion + prefix tokens
        text_kv: torch.Tensor,        # (B, S, D)  – projected text tokens
        text_key_padding_mask=None,   # (B, S) bool – True = pad position
    ) -> torch.Tensor:
        # Sub-layer 1: plain cross-attention; MHA internal out_proj is zero-init
        # so contribution starts at 0 without a redundant extra projection.
        q = self.norm_q(x)
        ca_out, _ = self.cross_attn(
            query=q,
            key=text_kv,
            value=text_kv,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        x = x + ca_out

        # Sub-layer 2: FFN with zero-init output projection
        x = x + self.ff_out_proj(self.ff_act(self.ff_in_proj(self.norm_ff(x))))
        return x


# ---------------------------------------------------------------------------
#  Extended RAG wrapper with MCA  (Flamingo-style)
# ---------------------------------------------------------------------------

class LLaMARAGMultiTextCAWrapper(LLaMARAGWrapper):
    """LLaMARAGWrapper + Flamingo-style GATED XATTN-DENSE injection.

    Parameters
    ----------
    base_model        : LLaMAHF instance
    model_dim         : Transformer hidden dim
    retrieval_dim     : h_cls embedding dim
    disable_rag       : ablation – removes retrieval prefix token
    text_token_dim    : T5 encoder output dim (default 1024 for sentence-t5-xxl)
    ca_n_layers       : **total** number of CA blocks to insert (default 6).
                        Internally computes ca_every_n_layers = n_total // ca_n_layers.
                        E.g. for a 32-layer backbone: 32//6 = 5, inserting before
                        layers {4,9,14,19,24,29}.
    ca_every_n_layers : override the interval directly (mutually exclusive with
                        ca_n_layers; takes precedence if provided).
    ca_n_head         : number of CA heads; None → same as backbone
    ff_mult           : FFN expansion factor in each CA block (default 4)
    """

    def __init__(
        self,
        base_model,
        model_dim: int = 768,
        retrieval_dim: int = 768,
        disable_rag: bool = False,
        text_token_dim: int = 1024,
        ca_n_layers: int = 6,
        ca_every_n_layers: int = None,
        ca_n_head: int = None,
        ff_mult: int = 2,
        ca_block_cls=None,
    ) -> None:
        super().__init__(base_model, model_dim, retrieval_dim, disable_rag)
        if ca_block_cls is None:
            ca_block_cls = TextCrossAttentionBlock

        n_total = len(base_model.transformer.h)

        # --- Determine insertion interval (Flamingo-style) ---
        if ca_every_n_layers is not None:
            every_n = max(1, ca_every_n_layers)
        else:
            every_n = max(1, n_total // ca_n_layers)

        # Collect layer indices WHERE we insert a CA block (before that LM block)
        # Flamingo: (layer_idx + 1) % every_n == 0
        self.ca_layer_indices = [
            idx for idx in range(n_total) if (idx + 1) % every_n == 0
        ]
        n_ca = len(self.ca_layer_indices)
        self.ca_every_n_layers = every_n   # stored for repr / logging

        if ca_n_head is None:
            ca_n_head = base_model.config.n_head

        # Shared linear: T5-token dim → model_dim
        self.text_token_proj = nn.Linear(text_token_dim, model_dim, bias=False)

        # One CA block per insertion point (type determined by ca_block_cls)
        self.ca_blocks = nn.ModuleList(
            [ca_block_cls(model_dim, ca_n_head, ff_mult) for _ in range(n_ca)]
        )

        # Learnable null KV for CFG dropout
        self.null_text_kv = nn.Parameter(torch.zeros(1, 1, model_dim))

        # Build fast lookup: layer_index → ca_block_index
        self._ca_index_map = {li: ci for ci, li in enumerate(self.ca_layer_indices)}

    def extra_repr(self) -> str:
        n_total = len(self.base_model.transformer.h)
        return (
            f"ca_every_n_layers={self.ca_every_n_layers}, "
            f"n_ca_blocks={len(self.ca_blocks)}, "
            f"n_total_layers={n_total}, "
            f"ca_at_layers={self.ca_layer_indices}"
        )

    # ------------------------------------------------------------------
    #  Helper: build projected text KV + padding mask
    # ------------------------------------------------------------------

    def _prepare_text_kv(
        self,
        text_tokens,        # (B, S, text_token_dim) float32 | None
        text_token_lens,    # (B,) int64 | None
        cfg_drop_mask,      # (B,) bool   | None
        bsz: int,
    ):
        """Return (text_kv, text_key_padding_mask) or (None, None)."""
        if text_tokens is None:
            return None, None

        text_kv = self.text_token_proj(text_tokens)   # (B, S, D)
        S = text_kv.size(1)
        device = text_kv.device

        # Padding mask: True = invalid position
        if text_token_lens is not None:
            idx = torch.arange(S, device=device).unsqueeze(0)
            text_key_padding_mask = idx >= text_token_lens.unsqueeze(1)
        else:
            text_key_padding_mask = None

        # CFG dropout: replace dropped samples' KVs with null token
        if cfg_drop_mask is not None and cfg_drop_mask.any():
            null_kv = self.null_text_kv.expand(bsz, S, -1)
            drop3d = cfg_drop_mask.view(-1, 1, 1)
            text_kv = torch.where(drop3d, null_kv, text_kv)
            if text_key_padding_mask is not None:
                text_key_padding_mask = text_key_padding_mask.clone()
                text_key_padding_mask[cfg_drop_mask] = False

        return text_kv, text_key_padding_mask

    # ------------------------------------------------------------------
    #  Forward  (Flamingo insertion order: CA → LM block)
    # ------------------------------------------------------------------

    def forward(
        self,
        motion_latents,
        text_emb,
        top3_h_cls=None,
        top3_sim_scores=None,
        cfg_drop_mask=None,
        empty_text_emb=None,
        text_tokens=None,          # (B, S, text_token_dim)
        text_token_lens=None,      # (B,)
    ):
        """Forward with Flamingo-style interleaved GATED XATTN-DENSE.

        When text_tokens=None the CA path is fully skipped → identical to base.
        """
        bsz = motion_latents.size(0)

        # ---- Build prefix condition tokens (inherited, unchanged) ----
        cond_tokens = self._build_condition_tokens(
            text_emb, top3_h_cls, top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )
        motion_tokens = self.base_model.transformer.wte(motion_latents.float())
        x = torch.cat([cond_tokens, motion_tokens], dim=1)

        # ---- Prepare cross-attention KV ----
        text_kv, text_key_padding_mask = self._prepare_text_kv(
            text_tokens, text_token_lens, cfg_drop_mask, bsz
        )

        # ---- Flamingo-style interleaved forward ----
        #  For each selected layer:  GATED_XATTN_DENSE(x) → LM_block(x)
        #  For other layers:         LM_block(x)  only
        for layer_idx, block in enumerate(self.base_model.transformer.h):
            if text_kv is not None and layer_idx in self._ca_index_map:
                ca_block = self.ca_blocks[self._ca_index_map[layer_idx]]
                x = ca_block(x, text_kv, text_key_padding_mask)
            x = block(x)

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
        temperature=1.0,
        text_tokens=None,
        text_token_lens=None,
    ):
        """Sample one next latent token with synergistic CFG."""
        bsz = text_emb.shape[0]
        device = motion_prefix.device
        cond_mask   = torch.zeros(bsz, dtype=torch.bool, device=device)
        uncond_mask = torch.ones(bsz, dtype=torch.bool, device=device)

        cond_hidden = self.forward(
            motion_prefix, text_emb, top3_h_cls, top3_sim_scores,
            cfg_drop_mask=cond_mask, empty_text_emb=empty_text_emb,
            text_tokens=text_tokens, text_token_lens=text_token_lens,
        )[:, -1, :]

        uncond_hidden = self.forward(
            motion_prefix, text_emb, top3_h_cls, top3_sim_scores,
            cfg_drop_mask=uncond_mask, empty_text_emb=empty_text_emb,
            text_tokens=text_tokens, text_token_lens=text_token_lens,
        )[:, -1, :]

        mix_hidden = torch.cat([cond_hidden, uncond_hidden], dim=0)
        sampled = self.base_model.diff_loss.sample(
            mix_hidden, temperature=temperature, cfg=cfg_scale
        )
        sampled_cond, _ = sampled.chunk(2, dim=0) if cfg_scale != 1.0 else (sampled, None)
        return sampled_cond


# ---------------------------------------------------------------------------
#  Checkpoint compatibility helper
# ---------------------------------------------------------------------------

def load_mca_checkpoint_compat(model, state_dict: dict) -> None:
    """Load a state_dict from the OLD single-gate MCA checkpoint into the
    new dual-gate model.

    Mapping applied:
      old  ca_blocks.{i}.gate   →  new  ca_blocks.{i}.attn_gate
      (ff_gate / ff / norm_ff are newly initialised to zeros / defaults)

    Usage
    -----
        ckpt = torch.load('net_Iter100000.pth', map_location='cpu')
        load_mca_checkpoint_compat(rag_model, ckpt['rag'])
    """
    new_sd = {}
    for k, v in state_dict.items():
        # rename single gate → attn_gate
        if k.startswith('ca_blocks.') and k.endswith('.gate'):
            new_k = k[:-len('.gate')] + '.attn_gate'
        else:
            new_k = k
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    # Expected missing: ff_gate, ff.*, norm_ff.* for each CA block
    # All new params remain at their zero / default init values
    newly_added = [k for k in missing if any(
        s in k for s in ('ff_gate', 'ff.', 'norm_ff')
    )]
    truly_missing = [k for k in missing if k not in newly_added]
    if truly_missing:
        print(f'[load_mca_checkpoint_compat] WARN truly missing keys: {truly_missing}')
    if unexpected:
        print(f'[load_mca_checkpoint_compat] WARN unexpected keys: {unexpected}')
    print(
        f'[load_mca_checkpoint_compat] OK – '
        f'{len(newly_added)} new Flamingo-style params initialised to zero.'
    )


# ---------------------------------------------------------------------------
#  Branch B wrapper: Flamingo gated  (convenience subclass)
# ---------------------------------------------------------------------------

class LLaMARAGMultiTextCAGatedWrapper(LLaMARAGMultiTextCAWrapper):
    """Convenience subclass that uses GatedCrossAttentionBlock (Flamingo Branch B).

    Drop-in replacement for LLaMARAGMultiTextCAWrapper; all arguments are
    identical.  Checkpoints produced by the Flamingo training run (containing
    ``attn_gate`` / ``ff_gate`` keys) load correctly via strict=False.

    Example
    -------
        rag_model = LLaMARAGMultiTextCAGatedWrapper(
            base_model=base_model,
            model_dim=config.n_embd,
            ca_n_layers=6,
        )
        ckpt = torch.load('net_Iter090000.pth', map_location='cpu')
        rag_model.load_state_dict(ckpt['rag'], strict=False)
    """

    def __init__(self, *args, **kwargs):
        kwargs['ca_block_cls'] = GatedCrossAttentionBlock
        super().__init__(*args, **kwargs)
