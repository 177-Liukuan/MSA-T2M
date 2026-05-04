"""
TAE-GAN-v1: Decoder-Only Adversarial Fine-Tuning of Causal TAE.

Strategy
--------
- Encoder     : FROZEN  (pretrained TAE weights, latent space preserved)
- Decoder     : TRAINABLE
- Discriminator: TRAINABLE (trained from scratch)

Loss breakdown (after disc_start warmup):
  L_G = L_recon + L_KL + w_adv * L_adv_G + w_fm * L_fm
  L_D = L_adv_D  (hinge loss)

where:
  L_recon : reconstruction loss (delegated to ReConsLoss in trainer)
  L_KL    : KL divergence
  L_adv_G : generator hinge loss  = -mean(D(fake))
  L_fm    : feature matching loss  (L1 on discriminator internals)
  L_adv_D : discriminator hinge loss

Adaptive weight (VideoVAE+ / VQGAN style):
  w_adv = ||∂L_recon/∂θ_last|| / (||∂L_adv_G/∂θ_last|| + ε) · disc_weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tae import Causal_HumanTAE
from models.motion_discriminator import MotionPatchDiscriminator, hinge_d_loss
from models.motion_dynamics import extract_dynamics_features, feature_matching_loss, DISC_IN_CHANNELS


class TAEGANV1(nn.Module):

    def __init__(
        self,
        # ── TAE backbone ──────────────────────────────────────────────────
        hidden_size: int = 1024,
        down_t: int = 2,
        stride_t: int = 2,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        latent_dim: int = 16,
        clip_range: list = None,
        # ── GAN hyper-params ──────────────────────────────────────────────
        disc_start: int = 10000,      # warm-up steps before GAN loss activates
        disc_weight: float = 0.5,     # adaptive weight scale for adv loss
        fm_weight: float = 10.0,      # feature matching weight
        disc_ndf: int = 64,           # discriminator base channels
        disc_n_layers: int = 3,       # discriminator depth
    ):
        super().__init__()

        if clip_range is None:
            clip_range = [-30, 20]

        # ── Build TAE backbone ────────────────────────────────────────────
        self.tae = Causal_HumanTAE(
            hidden_size=hidden_size,
            down_t=down_t,
            stride_t=stride_t,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            latent_dim=latent_dim,
            clip_range=clip_range,
        )

        # ── Build discriminator ───────────────────────────────────────────
        self.discriminator = MotionPatchDiscriminator(
            in_channels=DISC_IN_CHANNELS,
            ndf=disc_ndf,
            n_layers=disc_n_layers,
        )

        # ── GAN hyper-params ──────────────────────────────────────────────
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.fm_weight = fm_weight

        # ── Freeze encoder ────────────────────────────────────────────────
        self._freeze_encoder()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _freeze_encoder(self):
        """Freeze the TAE encoder so only the decoder is fine-tuned."""
        for param in self.tae.tae.encoder.parameters():
            param.requires_grad_(False)

    def load_tae_checkpoint(self, ckpt_path: str, strict: bool = False):
        """
        Load a pretrained TAE checkpoint into self.tae.

        Supports both:
          - plain state_dict (e.g. torch.save(net.state_dict(), path))
          - {'net': state_dict} (e.g. torch.save({'net': net.state_dict()}, path))
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('net', ckpt)
        missing, unexpected = self.tae.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    def get_last_decoder_layer(self) -> torch.Tensor:
        """
        Return the weight tensor of the final decoder conv layer.
        Used for adaptive weight computation (gradient norm ratio).
        The CausalDecoder.model ends with:  [..., ReLU, CausalConv1d(width→272)]
        That last CausalConv1d.conv.weight is what we reference.
        """
        return self.tae.tae.decoder.model[-1].conv.weight

    # ─────────────────────────────────────────────────────────────────────────
    # Forward helpers (delegate to TAE)
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, x):
        return self.tae.encode(x)

    def forward(self, x):
        """Standard reconstruction forward — used in eval and base loss."""
        return self.tae(x)

    def forward_decoder(self, z):
        return self.tae.forward_decoder(z)

    # ─────────────────────────────────────────────────────────────────────────
    # GAN loss interface
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive discriminator weight (VQGAN / VideoVAE+ style):
          w = ||∇_last(L_recon)|| / (||∇_last(L_adv)|| + ε) * disc_weight

        Both autograd.grad calls use retain_graph=True so the main graph
        is preserved for the subsequent accelerator.backward(loss_G) call.
        """
        last_layer = self.get_last_decoder_layer()
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads   = torch.autograd.grad(g_loss,   last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def generator_step(
        self,
        motion_pred: torch.Tensor,   # [B, T, 272]  (normalised, in graph)
        nll_loss: torch.Tensor,      # scalar recon loss (in graph, retain_graph needed)
        mean: torch.Tensor,          # [272] normalisation mean (on device)
        std: torch.Tensor,           # [272] normalisation std  (on device)
        global_step: int,
    ):
        """
        Compute generator-side GAN losses.

        Returns:
            g_loss      : adversarial generator loss (None before disc_start)
            fm_loss     : feature matching loss       (None before disc_start)
            d_weight    : adaptive weight scalar      (None before disc_start)
            log         : dict of scalar logging values
        """
        if global_step < self.disc_start:
            return None, None, None, {}

        # Denormalise (differentiable — motion_pred stays in graph)
        pred_denorm = motion_pred * std + mean                   # [B, T, 272]
        feats_fake  = extract_dynamics_features(pred_denorm)     # [B, 134, T]

        # Generator forward through discriminator
        logits_fake, feats_fake_list = self.discriminator(feats_fake, return_features=True)
        g_loss = -torch.mean(logits_fake)

        # Feature matching: run discriminator on real in no_grad to get targets
        # (real feats are passed in separately from the trainer to avoid double alloc)
        # Returns just g_loss here; FM is computed in trainer after real feats obtained.

        # Adaptive weight
        try:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss)
        except RuntimeError:
            # Happens during eval (no graph) — safe to ignore
            d_weight = torch.tensor(0.0, device=motion_pred.device)

        log = {
            'g_loss':    g_loss.detach(),
            'd_weight':  d_weight.detach() if isinstance(d_weight, torch.Tensor) else d_weight,
        }
        return g_loss, feats_fake_list, d_weight, log

    def discriminator_step(
        self,
        motion_gt: torch.Tensor,    # [B, T, 272]  normalised GT
        motion_pred: torch.Tensor,  # [B, T, 272]  normalised pred (detached)
        mean: torch.Tensor,
        std: torch.Tensor,
        global_step: int,
    ):
        """
        Compute discriminator loss.

        Returns:
            d_loss : scalar (None before disc_start)
            log    : dict
        """
        if global_step < self.disc_start:
            return None, {}

        with torch.no_grad():
            gt_denorm   = motion_gt.detach()   * std + mean
            pred_denorm = motion_pred.detach() * std + mean

        feats_real = extract_dynamics_features(gt_denorm)
        feats_fake = extract_dynamics_features(pred_denorm)

        logits_real = self.discriminator(feats_real)
        logits_fake = self.discriminator(feats_fake)

        d_loss = hinge_d_loss(logits_real, logits_fake)

        log = {
            'd_loss':       d_loss.detach(),
            'logits_real':  logits_real.detach().mean(),
            'logits_fake':  logits_fake.detach().mean(),
        }
        return d_loss, log

    def real_dynamics_features(
        self,
        motion_gt: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Extract dynamics features from GT motion (detached, for FM loss targets)."""
        with torch.no_grad():
            gt_denorm = motion_gt.detach() * std + mean
            return extract_dynamics_features(gt_denorm)
