import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.diffusion import create_diffusion


class BaseGenerativeHead(nn.Module):
    """Unified interface for token-level generative heads.

    Subclasses must preserve the interface contract:
    - forward(target, z, mask) -> (loss_scalar, pred_xstart)
    - sample(z, temperature, cfg) -> sampled_latent
    """

    def forward(self, target, z, mask=None):
        raise NotImplementedError

    def sample(self, z, temperature=1.0, cfg=1.0):
        raise NotImplementedError


class DDPMHead(BaseGenerativeHead):
    """Original DDPM head kept intact for backward compatibility."""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        learn_sigma=False,
    ):
        super().__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        pred_xstart = loss_dict["pred_xstart"]
        if mask is not None:
            mask = mask.to(loss.dtype)
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss.mean(), pred_xstart

    def sample(self, z, temperature=1.0, cfg=1.0):
        if cfg != 1.0:
            if z.shape[0] % 2 != 0:
                raise ValueError("CFG sampling requires even batch size (cond+uncond).")
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=z.device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=z.device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
        )

        return sampled_token_latent

    def sample_additive_cfg(self, z_nn, z_tn, z_tr, s_t=6.0, s_r=2.0, temperature=1.0):
        """3-forward additive noise-space CFG for DDPM.

        All 3 condition vectors are in-distribution:
          z_nn: forward(null_text, null_retr)  -- ~1%  of training steps
          z_tn: forward(real_text, null_retr)  -- ~9%  of training steps
          z_tr: forward(real_text, real_retr)  -- ~81% of training steps

        Guided noise estimate:
          eps_guided = eps_nn + s_t*(eps_tn - eps_nn) + s_r*(eps_tr - eps_tn)
        """
        B = z_nn.shape[0]
        device = z_nn.device
        z_triple = torch.cat([z_nn, z_tn, z_tr], dim=0)  # (3B, D)

        _B, _s_t, _s_r = B, float(s_t), float(s_r)

        def _model_additive(x, t, c):
            x3 = x.repeat(3, 1)
            t3 = t.repeat(3)
            eps3 = self.net(x3, t3, c)
            eps_nn_b, eps_tn_b, eps_tr_b = torch.split(eps3, _B, dim=0)
            return eps_nn_b + _s_t * (eps_tn_b - eps_nn_b) + _s_r * (eps_tr_b - eps_tn_b)

        noise = torch.randn(B, self.in_channels, device=device) * float(temperature)
        model_kwargs = dict(c=z_triple)
        return self.gen_diffusion.p_sample_loop(
            _model_additive, noise.shape, noise,
            clip_denoised=False, model_kwargs=model_kwargs,
            progress=False, temperature=temperature,
        )


class RectifiedFlowHead(BaseGenerativeHead):
    """Rectified Flow head using velocity prediction in latent space.

    Training path:
        eps ~ N(0, I), t ~ U(0, 1)
        zt = (1 - t) * z0 + t * eps
        v_target = eps - z0
        loss = MSE(v_pred(zt, t, cond), v_target)

    Sampling path:
        Start from Gaussian noise at t=1 and integrate ODE to t=0 with Euler.
    """

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_flow_steps=20,
        flow_solver="euler",
        grad_checkpointing=False,
        rf_time_sampling="uniform",
        rf_loss_type="mse",
    ):
        super().__init__()
        self.in_channels = target_channels
        self.num_flow_steps = int(num_flow_steps)
        self.flow_solver = str(flow_solver).lower()
        self.rf_time_sampling = str(rf_time_sampling).lower()
        self.rf_loss_type = str(rf_loss_type).lower()

        if self.rf_time_sampling != "uniform":
            raise ValueError(f"Unsupported rf_time_sampling: {self.rf_time_sampling}")
        if self.rf_loss_type != "mse":
            raise ValueError(f"Unsupported rf_loss_type: {self.rf_loss_type}")

        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

    @staticmethod
    def _assert_shapes(target, z):
        if target.ndim != 2 or z.ndim != 2:
            raise ValueError(f"Expected rank-2 tensors target/z, got {target.shape} and {z.shape}")
        if target.shape[0] != z.shape[0]:
            raise ValueError(f"Batch mismatch: target {target.shape[0]} vs z {z.shape[0]}")

    def forward(self, target, z, mask=None):
        self._assert_shapes(target, z)

        n = target.shape[0]
        t = torch.rand(n, device=target.device, dtype=target.dtype)
        t_view = t[:, None]

        eps = torch.randn_like(target)
        zt = (1.0 - t_view) * target + t_view * eps
        v_target = eps - target

        v_pred = self.net(zt, t, z)
        per_token_loss = (v_pred - v_target).pow(2).mean(dim=-1)

        if mask is not None:
            mask = mask.to(per_token_loss.dtype)
            loss = (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = per_token_loss.mean()

        # Recover x_start estimate from velocity field: z0 = zt - t * v.
        pred_xstart = zt - t_view * v_pred
        return loss, pred_xstart

    def _euler_step(self, x, z_cond, t_scalar, dt):
        t_batch = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=x.dtype)
        v = self.net(x, t_batch, z_cond)
        return x + dt * v

    def sample(self, z, temperature=1.0, cfg=1.0):
        if self.flow_solver != "euler":
            raise ValueError(f"Unsupported flow solver: {self.flow_solver}. Only 'euler' is implemented.")

        steps = max(1, int(self.num_flow_steps))
        dt = -1.0 / float(steps)

        if cfg != 1.0:
            if z.shape[0] % 2 != 0:
                raise ValueError("CFG sampling requires even batch size (cond+uncond).")

            half = z.shape[0] // 2
            x_half = torch.randn(half, self.in_channels, device=z.device, dtype=z.dtype) * float(temperature)

            for i in range(steps):
                t_now = 1.0 - float(i) / float(steps)
                x_pair = torch.cat([x_half, x_half], dim=0)
                t_pair = torch.full((x_pair.shape[0],), t_now, device=z.device, dtype=z.dtype)
                v_pair = self.net(x_pair, t_pair, z)
                v_cond, v_uncond = torch.split(v_pair, half, dim=0)
                v_guided = v_uncond + cfg * (v_cond - v_uncond)
                x_half = x_half + dt * v_guided

            sampled = torch.cat([x_half, x_half], dim=0)
            return sampled

        x = torch.randn(z.shape[0], self.in_channels, device=z.device, dtype=z.dtype) * float(temperature)
        for i in range(steps):
            t_now = 1.0 - float(i) / float(steps)
            x = self._euler_step(x=x, z_cond=z, t_scalar=t_now, dt=dt)
        return x

    def sample_additive_cfg(self, z_nn, z_tn, z_tr, s_t=6.0, s_r=2.0, temperature=1.0):
        """3-forward additive velocity-space CFG for Rectified Flow.

        All 3 condition vectors are in-distribution:
          z_nn: forward(null_text, null_retr)  -- ~1%  of training steps
          z_tn: forward(real_text, null_retr)  -- ~9%  of training steps
          z_tr: forward(real_text, real_retr)  -- ~81% of training steps

        Guided velocity:
          v_guided = v_nn + s_t*(v_tn - v_nn) + s_r*(v_tr - v_tn)
        """
        if self.flow_solver != "euler":
            raise ValueError("Only euler solver supported for sample_additive_cfg.")

        steps = max(1, int(self.num_flow_steps))
        dt = -1.0 / float(steps)
        B = z_nn.shape[0]
        z_triple = torch.cat([z_nn, z_tn, z_tr], dim=0)  # (3B, D)

        x = torch.randn(B, self.in_channels, device=z_nn.device, dtype=z_nn.dtype) * float(temperature)
        for i in range(steps):
            t_now = 1.0 - float(i) / float(steps)
            x3 = x.repeat(3, 1)
            t3 = torch.full((3 * B,), t_now, device=x.device, dtype=x.dtype)
            v3 = self.net(x3, t3, z_triple)
            v_nn, v_tn, v_tr = torch.split(v3, B, dim=0)
            v_guided = v_nn + float(s_t) * (v_tn - v_nn) + float(s_r) * (v_tr - v_tn)
            x = x + dt * v_guided
        return x


class DiffLoss(nn.Module):
    """Compatibility wrapper for DDPM and Rectified Flow generative heads."""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        learn_sigma=False,
        flow_type="ddpm",
        num_flow_steps=20,
        flow_solver="euler",
        rf_time_sampling="uniform",
        rf_loss_type="mse",
    ):
        super().__init__()
        self.in_channels = target_channels
        self.flow_type = str(flow_type).lower()

        if self.flow_type == "ddpm":
            self.head = DDPMHead(
                target_channels=target_channels,
                z_channels=z_channels,
                depth=depth,
                width=width,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                learn_sigma=learn_sigma,
            )
            # Preserve historical attributes for legacy callsites.
            self.train_diffusion = self.head.train_diffusion
            self.gen_diffusion = self.head.gen_diffusion
        elif self.flow_type == "rectified_flow":
            self.head = RectifiedFlowHead(
                target_channels=target_channels,
                z_channels=z_channels,
                depth=depth,
                width=width,
                num_flow_steps=num_flow_steps,
                flow_solver=flow_solver,
                grad_checkpointing=grad_checkpointing,
                rf_time_sampling=rf_time_sampling,
                rf_loss_type=rf_loss_type,
            )
        else:
            raise ValueError(f"Unsupported flow_type: {self.flow_type}")

        # Keep attribute name stable for any external access.
        self.net = self.head.net

    def forward(self, target, z, mask=None):
        return self.head.forward(target=target, z=z, mask=mask)

    def sample(self, z, temperature=1.0, cfg=1.0):
        return self.head.sample(z=z, temperature=temperature, cfg=cfg)

    def sample_additive_cfg(self, z_nn, z_tn, z_tr, s_t=6.0, s_r=2.0, temperature=1.0):
        """Delegate 3-forward additive CFG to the active head.

        Args:
            z_nn: condition (null_text, null_retr)  shape (B, D)
            z_tn: condition (real_text, null_retr)  shape (B, D)
            z_tr: condition (real_text, real_retr)  shape (B, D)
            s_t:  text guidance scale      (recommended 5-7)
            s_r:  retrieval guidance scale (recommended 1.5-2.5)
        Returns:
            sampled latent of shape (B, target_channels)
        """
        return self.head.sample_additive_cfg(
            z_nn=z_nn, z_tn=z_tn, z_tr=z_tr,
            s_t=s_t, s_r=s_r, temperature=temperature,
        )



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """

        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, grad_checkpointing=False):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP.
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers.
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = x.float()

        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
