"""
Temporal PatchGAN Discriminator for Motion Sequences.
Design: lightweight Conv1d stack, returns patch-level logits + intermediate features.
Reference style: VideoVAE+ NLayerDiscriminator3D (adapted to 1D temporal).
"""

import torch
import torch.nn as nn


class MotionPatchDiscriminator(nn.Module):
    """
    Temporal PatchGAN discriminator.

    Input : [B, C_in, T]  — time-series dynamics features
    Output: [B, 1, T']    — patch-level real/fake logits (+ optional feature list)

    Kept intentionally lightweight to ensure stable GAN training:
      - GroupNorm (works with small batch sizes, unlike BatchNorm)
      - LeakyReLU(0.2)
      - No global pooling → patch output preserves temporal structure
    """

    def __init__(self, in_channels: int = 132, ndf: int = 64, n_layers: int = 3,
                 use_spectral_norm: bool = True):
        super().__init__()

        kw = 4      # kernel width
        padw = 1

        def _sn(layer):
            """Optionally wrap a conv with spectral normalisation."""
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # Build layer list so we can return intermediate features
        layers = []
        # First conv — no norm
        layers.append(_sn(nn.Conv1d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.append(_sn(nn.Conv1d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)))
            layers.append(nn.GroupNorm(min(32, nf), nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Penultimate conv — stride 1, increase channels
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.append(_sn(nn.Conv1d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)))
        layers.append(nn.GroupNorm(min(32, nf), nf))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final output conv — 1 channel logits
        layers.append(_sn(nn.Conv1d(nf, 1, kernel_size=kw, stride=1, padding=padw)))

        self.layers = nn.ModuleList(layers)

        # Weight initialisation (same style as VideoVAE+ weights_init)
        # Note: spectral_norm wraps the original weight; still accessible via .weight_orig
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x              : [B, C_in, T]
            return_features: if True, also return list of intermediate activations
        Returns:
            logits         : [B, 1, T']
            features       : list of intermediate tensors (only when return_features=True)
        """
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        # Last element is the logits; preceding ones are intermediate features
        if return_features:
            return features[-1], features[:-1]
        return features[-1]


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Standard hinge adversarial loss for the discriminator."""
    loss_real = torch.mean(nn.functional.relu(1.0 - logits_real))
    loss_fake = torch.mean(nn.functional.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)
