"""
SAE-v1: Semantic Alignment Encoder, Phase 1.

Key change vs Causal TAE:
  The encoder's first CausalConv1d uses pad_mode='replicate' instead of
  zero-padding, which eliminates the startup jitter in the first few frames
  of decoded motion sequences (as described in MoLingo).

All other layers remain identical to Causal_TAE.
"""

import torch
import torch.nn as nn
from models.resnet import CausalResnet1D


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D causal convolution with configurable left-padding mode.

    Args:
        pad_mode: 'zero' (original TAE behaviour) or 'replicate' (SAE-v1 fix).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, pad_mode='zero'):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
        )

    def forward(self, x):
        if self.pad_mode == 'zero':
            x = nn.functional.pad(x, (self.pad, 0))
        else:
            x = nn.functional.pad(x, (self.pad, 0), mode=self.pad_mode)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SAEEncoder(nn.Module):
    """Causal encoder for SAE-v1.

    Identical to CausalEncoder in causal_cnn.py except:
      - The very first CausalConv1d (input projection) uses pad_mode='replicate'
        to eliminate zero-padding artefacts at sequence start.
      - All subsequent CausalConv1d layers still use pad_mode='zero'.
    """

    def __init__(self,
                 input_emb_width=272,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range=None):
        super().__init__()
        self.clip_range = clip_range if clip_range is not None else []
        self.proj = nn.Linear(width, latent_dim * 2)

        blocks = []
        filter_t = stride_t * 2

        # ---- First conv: replicate padding (SAE-v1 fix) ----
        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1,
                                   pad_mode='replicate'))
        blocks.append(nn.ReLU())

        # ---- Downsampling blocks: zero padding (unchanged) ----
        for _ in range(down_t):
            block = nn.Sequential(
                CausalConv1d(width, width, filter_t, stride_t, 1,
                             pad_mode='zero'),
                CausalResnet1D(width, depth, dilation_growth_rate,
                               activation=activation, norm=norm),
            )
            blocks.append(block)

        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1,
                                   pad_mode='zero'))
        self.model = nn.Sequential(*blocks)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.model(x)
        x = x.transpose(1, 2)          # (B, T', C)
        x = self.proj(x)               # (B, T', latent_dim*2)
        mu, logvar = x.chunk(2, dim=2)
        logvar = torch.clamp(logvar,
                             self.clip_range[0] if self.clip_range else -30,
                             self.clip_range[1] if self.clip_range else 20)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# ---------------------------------------------------------------------------
# Decoder (unchanged from Causal TAE)
# ---------------------------------------------------------------------------

class SAEDecoder(nn.Module):
    """Causal decoder — identical to CausalDecoder in causal_cnn.py."""

    def __init__(self,
                 input_emb_width=272,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t = stride_t * 2
        blocks.append(CausalConv1d(hidden_size, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for _ in range(down_t):
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate,
                               reverse_dilation=True,
                               activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)

        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.model(z)


# ---------------------------------------------------------------------------
# Top-level SAE-v1 modules
# ---------------------------------------------------------------------------

class SAE_v1(nn.Module):
    """SAE-v1 (replicate-padding encoder + causal decoder)."""

    def __init__(self,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 width=1024,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range=None):
        super().__init__()
        self.decode_proj = nn.Linear(latent_dim, width)
        self.encoder = SAEEncoder(
            272, hidden_size, down_t, stride_t, width, depth,
            dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=latent_dim,
            clip_range=clip_range if clip_range is not None else [],
        )
        self.decoder = SAEDecoder(
            272, hidden_size, down_t, stride_t, width, depth,
            dilation_growth_rate, activation=activation, norm=norm,
        )

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def encode(self, x):
        x_in = self.preprocess(x)
        z, mu, logvar = self.encoder(x_in)
        z = self.postprocess(z)
        z = z.contiguous().view(-1, z.shape[-1])
        return z, mu, logvar

    def forward(self, x):
        x_in = self.preprocess(x)
        z, mu, logvar = self.encoder(x_in)
        z_proj = self.decode_proj(z)
        x_dec = self.decoder(z_proj)
        x_out = self.postprocess(x_dec)
        return x_out, mu, logvar

    def forward_decoder(self, z):
        z_proj = self.decode_proj(z)
        x_dec = self.decoder(z_proj)
        return self.postprocess(x_dec)


class SAE_HumanV1(nn.Module):
    """Thin wrapper matching the Causal_HumanTAE interface for drop-in use."""

    def __init__(self,
                 hidden_size=1024,
                 down_t=2,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range=None):
        super().__init__()
        self.tae = SAE_v1(
            hidden_size, down_t, stride_t, hidden_size, depth,
            dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=latent_dim,
            clip_range=clip_range if clip_range is not None else [],
        )

    def encode(self, x):
        return self.tae.encode(x)

    def forward(self, x):
        return self.tae(x)

    def forward_decoder(self, z):
        return self.tae.forward_decoder(z)
