"""
MSA-VAE: Multi-Scale Semantic Alignment VAE

Architecture:
  Bottom layer: Causal 1D CNN VAE - local physical representation with temporal causal property
  Top layer:    Transformer AE    - global semantic aggregation via [CLS] token

  Forward flow (dual-track decoupling):
    Motion -> CausalEncoder -> mu, logvar -> z_local (reparameterize)
    Physical track:  z_local  -> CausalDecoder -> x_recon (robust noisy decoding)
    Semantic track:  mu_local -> TransformerEncoder([CLS]) -> h_cls
                     h_cls    -> TransformerDecoder -> mu_recon
                     Loss: ||mu_recon - mu_local||^2

  Projection heads (Semantic alignment, based on deterministic mu):
    global_proj: h_cls  -> Semantic text space (global alignment)
    local_proj:  mu_i   -> Semantic text space (local alignment)
"""

import math
import torch
import torch.nn as nn
from models.causal_cnn import CausalEncoder, CausalDecoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLatentEncoder(nn.Module):
    """
    Transformer Encoder sitting on top of local latent token sequence.

    Prepends a learnable [CLS] token, adds positional encoding, then runs
    standard TransformerEncoder layers.  The [CLS] output (h_cls) captures
    global spatio-temporal dynamics of the full motion.
    """

    def __init__(self, latent_dim, d_model=512, nhead=8, num_layers=4,
                 ff_size=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(latent_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_size,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, z_local, key_padding_mask=None):
        """
        Args:
            z_local: (B, T', latent_dim) - local latent tokens from CNN encoder
            key_padding_mask: (B, T') bool - True for **padded** (invalid) positions,
                              None when all positions are valid (fixed-length batch).
        Returns:
            h_cls: (B, d_model) - global CLS representation
            h_seq: (B, 1+T', d_model) - full encoder output ([CLS] + tokens)
        """
        bs = z_local.size(0)
        z_proj = self.input_proj(z_local)                          # (B, T', d_model)
        cls_tokens = self.cls_token.expand(bs, -1, -1)             # (B, 1, d_model)
        z_seq = torch.cat([cls_tokens, z_proj], dim=1)             # (B, 1+T', d_model)
        z_seq = self.pos_encoder(z_seq)

        # Extend mask: [CLS] is always valid -> prepend False column
        if key_padding_mask is not None:
            cls_mask = torch.zeros(bs, 1, dtype=torch.bool, device=z_local.device)
            src_key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        else:
            src_key_padding_mask = None

        h_seq = self.transformer_encoder(z_seq, src_key_padding_mask=src_key_padding_mask)
        h_cls = h_seq[:, 0, :]                                    # (B, d_model)
        return h_cls, h_seq


class TransformerLatentDecoder(nn.Module):
    """
    Transformer Decoder that reconstructs the local latent sequence from h_cls.

    Uses h_cls as cross-attention memory and learnable positional queries as
    target to decode back a sequence of local latent tokens.
    Constraint: ||z_local - z_recon||^2  ensures h_cls is "generatively complete".
    """

    def __init__(self, latent_dim, d_model=512, nhead=8, num_layers=4,
                 ff_size=1024, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_size,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(self, h_cls, seq_len, tgt_key_padding_mask=None):
        """
        Args:
            h_cls:   (B, d_model)
            seq_len: int - number of local latent tokens to reconstruct
            tgt_key_padding_mask: (B, seq_len) bool - True for padded positions,
                                  None when all positions are valid.
        Returns:
            z_recon: (B, seq_len, latent_dim)
        """
        bs = h_cls.size(0)
        # Zero-initialized queries with positional encoding
        time_queries = torch.zeros(bs, seq_len, self.d_model, device=h_cls.device)
        time_queries = self.pos_encoder(time_queries)

        memory = h_cls.unsqueeze(1)                               # (B, 1, d_model)
        h_decoded = self.transformer_decoder(
            tgt=time_queries, memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        z_recon = self.output_proj(h_decoded)                     # (B, seq_len, latent_dim)
        return z_recon


# ---------------------------------------------------------------------------
#  MSA-VAE: full model combining Causal CNN VAE + Transformer AE
# ---------------------------------------------------------------------------

class MSA_VAE(nn.Module):
    """
    Multi-Scale Semantic Alignment VAE.

    Bottom: Causal 1D CNN VAE   -> z_local  (local, temporally causal latents)
    Top   : Transformer AE      -> h_cls    (global semantic aggregation)
    Heads : global_proj / local_proj -> CLIP-aligned features (for later use)
    """

    def __init__(
        self,
        # --- Causal CNN VAE ---
        input_emb_width=272,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        latent_dim=16,
        clip_range=None,
        # --- Transformer AE ---
        trans_d_model=512,
        trans_nhead=8,
        trans_enc_layers=4,
        trans_dec_layers=4,
        trans_ff_size=1024,
        trans_dropout=0.1,
        # --- CLIP alignment ---
        clip_dim=512,
        # --- Ablation ---
        disable_decoupling=False,
    ):
        super().__init__()
        if clip_range is None:
            clip_range = [-30, 20]

        self.latent_dim = latent_dim
        self.trans_d_model = trans_d_model
        self.stride_t = stride_t
        self.down_t = down_t
        self.disable_decoupling = bool(disable_decoupling)

        # ========== Bottom layer: Causal CNN VAE ==========
        cnn_width = hidden_size  # match original Causal_HumanTAE convention
        self.cnn_encoder = CausalEncoder(
            input_emb_width, hidden_size, down_t, stride_t, cnn_width,
            depth, dilation_growth_rate,
            activation=activation, norm=norm,
            latent_dim=latent_dim, clip_range=clip_range,
        )
        self.decode_proj = nn.Linear(latent_dim, cnn_width)
        self.cnn_decoder = CausalDecoder(
            input_emb_width, hidden_size, down_t, stride_t, cnn_width,
            depth, dilation_growth_rate,
            activation=activation, norm=norm,
        )

        # ========== Top layer: Transformer AE ==========
        self.trans_encoder = TransformerLatentEncoder(
            latent_dim=latent_dim, d_model=trans_d_model, nhead=trans_nhead,
            num_layers=trans_enc_layers, ff_size=trans_ff_size, dropout=trans_dropout,
        )
        self.trans_decoder = TransformerLatentDecoder(
            latent_dim=latent_dim, d_model=trans_d_model, nhead=trans_nhead,
            num_layers=trans_dec_layers, ff_size=trans_ff_size, dropout=trans_dropout,
        )

        # ========== CLIP alignment projection heads ==========
        # Global: [CLS] -> CLIP text space
        # trans_d_model == clip_dim: Identity (force direct CLIP space collapse)
        # otherwise: single Linear only, no nonlinearity
        if trans_d_model == clip_dim:
            self.global_proj = nn.Identity()
        else:
            self.global_proj = nn.Linear(trans_d_model, clip_dim)

        # Local: each z_i -> CLIP text space
        # latent_dim == clip_dim: Identity
        # otherwise: single Linear only, no nonlinearity
        if latent_dim == clip_dim:
            self.local_proj = nn.Identity()
        else:
            self.local_proj = nn.Linear(latent_dim, clip_dim)

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------
    def preprocess(self, x):
        """(B, T, 272) -> (B, 272, T)"""
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        """(B, 272, T) -> (B, T, 272)"""
        return x.permute(0, 2, 1)

    @staticmethod
    def lengths_to_mask(lengths, max_len):
        """Convert lengths -> padding mask (True = padded/invalid).
        Args:
            lengths: (B,) int tensor - valid length per sample
            max_len: int
        Returns:
            mask: (B, max_len) bool - True for padded positions
        """
        idx = torch.arange(max_len, device=lengths.device)
        return idx.unsqueeze(0) >= lengths.unsqueeze(1)

    # ------------------------------------------------------------------
    #  Encoding
    # ------------------------------------------------------------------
    def encode_cnn(self, x):
        """Raw motion -> local latent tokens via Causal CNN.
        Returns:
            z_local: (B, T', latent_dim)
            mu, logvar: same shape
        """
        x_in = self.preprocess(x)
        z_local, mu, logvar = self.cnn_encoder(x_in)
        return z_local, mu, logvar

    def encode_cnn_stats(self, x):
        """Raw motion -> deterministic posterior parameters."""
        x_in = self.preprocess(x)
        return self.cnn_encoder.encode_stats(x_in)

    def encode_transformer(self, z_local, key_padding_mask=None):
        """Local latents -> global [CLS] via Transformer Encoder.
        Args:
            z_local: (B, T', latent_dim)
            key_padding_mask: (B, T') bool - True for padded positions.
        Returns:
            h_cls: (B, d_model)
            h_seq: (B, 1+T', d_model)
        """
        h_cls, h_seq = self.trans_encoder(z_local, key_padding_mask=key_padding_mask)
        return h_cls, h_seq

    def encode(self, x, key_padding_mask=None):
        """Full encode: CNN + Transformer.
        Args:
            x: (B, T, 272)
            key_padding_mask: (B, T') bool - True for padded **latent** positions.
                              T' = T / stride^down_t.  None for fixed-length input.
        Returns:
            z_local, mu, logvar, h_cls
        """
        z_local, mu, logvar = self.encode_cnn(x)
        # Semantic track: Transformer receives deterministic mu (not noisy z_local)
        h_cls, _ = self.encode_transformer(mu, key_padding_mask=key_padding_mask)
        return z_local, mu, logvar, h_cls

    # ------------------------------------------------------------------
    #  Decoding
    # ------------------------------------------------------------------
    def decode_transformer(self, h_cls, seq_len, tgt_key_padding_mask=None):
        """Reconstruct local latent sequence from [CLS]."""
        return self.trans_decoder(h_cls, seq_len, tgt_key_padding_mask=tgt_key_padding_mask)

    def decode_cnn(self, z):
        """Local latents -> reconstructed motion via Causal CNN Decoder."""
        z_proj = self.decode_proj(z)
        x_decoder = self.cnn_decoder(z_proj)
        return self.postprocess(x_decoder)

    def forward_decoder(self, z):
        """Decode from given latent tokens (for inference / generation)."""
        return self.decode_cnn(z)

    def _latent_padding_mask(self, lengths, max_len):
        if lengths is None:
            return None
        latent_lengths = lengths.long()
        for _ in range(self.down_t):
            latent_lengths = torch.div(
                latent_lengths,
                self.stride_t,
                rounding_mode='floor',
            )
        return self.lengths_to_mask(latent_lengths, max_len)

    def forward_semantic(self, x, lengths=None):
        """Run deterministic CNN statistics and the semantic hierarchy only."""
        mu, logvar = self.encode_cnn_stats(x)
        key_padding_mask = self._latent_padding_mask(lengths, mu.size(1))
        h_cls, _ = self.encode_transformer(
            mu, key_padding_mask=key_padding_mask
        )
        mu_recon = self.decode_transformer(
            h_cls,
            seq_len=mu.size(1),
            tgt_key_padding_mask=key_padding_mask,
        )
        return {
            'mu': mu,
            'logvar': logvar,
            'h_cls': h_cls,
            'mu_recon': mu_recon,
            'trans_latent_target': mu,
            'clip_global_feat': self.global_proj(h_cls),
            'clip_local_feat': self.local_proj(mu),
        }

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x, lengths=None):
        """
        Full forward pass.

        Args:
            x: (B, T, 272) - input motion (may be zero-padded at the end)
            lengths: (B,) int tensor - actual motion length per sample.
                     None when all samples have the same length (no padding).

        Returns dict with keys:
            x_recon          - reconstructed motion         (B, T, 272)
            mu, logvar       - CNN VAE distribution params  (B, T', latent_dim)
            h_cls            - global [CLS] representation  (B, d_model)
            z_local          - noisy local latents (for CNN decoder)    (B, T', latent_dim)
            mu_recon         - Transformer-decoded deterministic mu     (B, T', latent_dim)
            clip_global_feat - projected global feature     (B, clip_dim)
            clip_local_feat  - projected local features     (B, T', clip_dim)
        """
        # --- Bottom: Causal CNN VAE ---
        # z_local is the reparameterized sample z = mu + eps * exp(0.5 * logvar) from CNN VAE.
        z_local, mu, logvar = self.encode_cnn(x)

        # --- Build padding mask for latent tokens ---
        key_padding_mask = self._latent_padding_mask(
            lengths, z_local.size(1)
        )

        # --- Semantic track: Transformer AE input/target switch for decoupling ablation ---
        trans_latent_target = z_local if self.disable_decoupling else mu
        h_cls, _ = self.encode_transformer(trans_latent_target, key_padding_mask=key_padding_mask)
        mu_recon = self.decode_transformer(h_cls, seq_len=trans_latent_target.size(1),
                                           tgt_key_padding_mask=key_padding_mask)

        # --- Physical track: CNN Decoder receives noisy z_local ---
        x_recon = self.decode_cnn(z_local)

        # --- CLIP projection heads (based on deterministic mu) ---
        clip_global_feat = self.global_proj(h_cls)
        clip_local_feat = self.local_proj(mu)

        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'h_cls': h_cls,
            'z_local': z_local,
            'mu_recon': mu_recon,
            'trans_latent_target': trans_latent_target,
            'clip_global_feat': clip_global_feat,
            'clip_local_feat': clip_local_feat,
        }


# ---------------------------------------------------------------------------
#  Convenience wrapper (matches Causal_HumanTAE interface)
# ---------------------------------------------------------------------------

class MSA_HumanVAE(nn.Module):
    """
    Wrapper around MSA_VAE, analogous to Causal_HumanTAE.
    Provides both the full dict-based forward() and backward-compatible helpers.
    """

    def __init__(
        self,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        latent_dim=16,
        clip_range=None,
        # Transformer AE hyper-params
        trans_d_model=512,
        trans_nhead=8,
        trans_enc_layers=4,
        trans_dec_layers=4,
        trans_ff_size=1024,
        trans_dropout=0.1,
        clip_dim=512,
        disable_decoupling=False,
    ):
        super().__init__()
        self.msa_vae = MSA_VAE(
            input_emb_width=272,
            hidden_size=hidden_size,
            down_t=down_t,
            stride_t=stride_t,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            latent_dim=latent_dim,
            clip_range=clip_range,
            trans_d_model=trans_d_model,
            trans_nhead=trans_nhead,
            trans_enc_layers=trans_enc_layers,
            trans_dec_layers=trans_dec_layers,
            trans_ff_size=trans_ff_size,
            trans_dropout=trans_dropout,
            clip_dim=clip_dim,
            disable_decoupling=disable_decoupling,
        )

    def encode(self, x, key_padding_mask=None):
        """Returns (z_local, mu, logvar, h_cls)."""
        return self.msa_vae.encode(x, key_padding_mask=key_padding_mask)

    def forward(self, x, lengths=None, semantic_only=False):
        """Returns full output dict from MSA_VAE."""
        if semantic_only:
            return self.msa_vae.forward_semantic(x, lengths=lengths)
        return self.msa_vae(x, lengths=lengths)

    def forward_decoder(self, z):
        """Decode from given latent tokens (for inference / generation)."""
        return self.msa_vae.forward_decoder(z)
