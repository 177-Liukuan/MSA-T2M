"""Train MotionStreamer stage-2 with global + local RAG conditioning.

Extends train_t2m_rag.py with local RAG prefix tokens:
  L_local uniformly-sampled mu latents per retrieved motion,
  weighted-summed over K retrievals, prepended before motion tokens.

New args vs train_t2m_rag.py:
  --z_latent_dir    path to t2m_latents_msa_vae/<exp>/
  --L_local         number of local RAG prefix tokens (default 4)
  --local_rag_dim   dim of mu latents (default 16)
"""

import os
import sys
import json
import argparse
import torch
import warnings
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from accelerate import Accelerator

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
from humanml3d_272 import dataset_msa_rag_local
import options.option_transformer as option_trans
import utils.utils_model as utils_model

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_iters - warmup_iters, eta_min=min_lr
        )

    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter):
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step(current_iter - self.warmup_iters)

    def state_dict(self):
        return {'warmup_iters': self.warmup_iters, 'total_iters': self.total_iters, 'min_lr': self.min_lr}

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters = state_dict['total_iters']
        self.min_lr = state_dict['min_lr']


def parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--text_latent_dir', type=str, default='./humanml3d_272/text_latents_t5')
    extra_parser.add_argument('--hcls_dir', type=str, default='./humanml3d_272/h_cls_latents_msa_vae/exp')
    extra_parser.add_argument('--z_latent_dir', type=str, default='./humanml3d_272/t2m_latents_msa_vae/exp')
    extra_parser.add_argument('--empty_text_path', type=str, default='./humanml3d_272/text_latents_t5/empty_text_embedding.npy')
    extra_parser.add_argument('--retrieval_topk', type=int, default=3)
    extra_parser.add_argument('--L_local', type=int, default=4, help='Number of local RAG prefix tokens.')
    extra_parser.add_argument('--local_rag_dim', type=int, default=16, help='Dim of z latents.')
    extra_parser.add_argument('--cfg_dropout_prob', type=float, default=0.1)
    extra_parser.add_argument('--num_workers', type=int, default=4)
    extra_parser.add_argument('--text_embed_dim', type=int, default=768)
    extra_parser.add_argument('--disable_rag', action='store_true', default=False)
    extra_parser.add_argument('--ema_decay', type=float, default=0.9999)
    extra_parser.add_argument('--ema_update_every', type=int, default=1)
    extra_parser.add_argument('--disable_ema', action='store_true', default=False)
    extra_parser.add_argument('--add_selfatten', action='store_true', default=False,
                              help='Use TransformerEncoder to encode z frames before cross-attention.')

    custom_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    for attr in [
        'text_latent_dir', 'hcls_dir', 'z_latent_dir', 'empty_text_path',
        'retrieval_topk', 'L_local', 'local_rag_dim', 'cfg_dropout_prob',
        'num_workers', 'text_embed_dim', 'disable_rag', 'ema_decay', 'ema_update_every',
        'add_selfatten',
    ]:
        setattr(args, attr, getattr(custom_args, attr))
    args.use_ema = not custom_args.disable_ema

    return args


def init_ema_state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def update_ema_state_dict(ema_state, model, decay):
    one_minus = 1.0 - decay
    for k, v in model.state_dict().items():
        if not torch.is_tensor(v):
            continue
        if k not in ema_state:
            ema_state[k] = v.detach().clone()
            continue
        if torch.is_floating_point(v):
            ema_state[k].mul_(decay).add_(v.detach(), alpha=one_minus)
        else:
            ema_state[k].copy_(v.detach())


def lengths_to_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


def estimate_lengths_from_padded_latents(m_tokens):
    valid = m_tokens.abs().sum(dim=-1) > 0
    lens = valid.long().sum(dim=1)
    return torch.clamp(lens, min=2)


def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    step = torch.tensor(step, dtype=torch.float32)
    total_steps = torch.tensor(total_steps, dtype=torch.float32)
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor


def replace_with_pred(latents, pred_xstart, step, total_steps):
    decay_factor = cosine_decay(step, total_steps).to(latents.device)
    bsz, seq_len, _ = latents.shape
    num_replace = int(seq_len * decay_factor)
    replace_indices = torch.randperm(seq_len, device=latents.device)[:num_replace]
    replace_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=latents.device)
    replace_mask[:, replace_indices] = 1
    updated = latents.clone()
    updated[replace_mask] = pred_xstart[replace_mask]
    return updated


def get_core_model(model):
    return model.module if hasattr(model, 'module') else model


def forward_loss_withmask_2_forward(
    latents,
    rag_model,
    m_lens,
    text_emb,
    top3_h_cls,
    top3_sim_scores,
    top_z_seqs,
    top_z_lens,
    step,
    total_steps,
    cfg_drop_mask,
    empty_text_emb,
    diffmlps_batch_mul=4,
):
    """Two-forward training with global+local RAG condition and diffusion loss."""
    core_model = get_core_model(rag_model)

    bsz, seq_len, _ = latents.shape
    mask = lengths_to_mask(m_lens, seq_len).reshape(bsz * seq_len).repeat(diffmlps_batch_mul)

    # First forward: no_grad, build pseudo x_start.
    with torch.no_grad():
        conditions = rag_model(
            motion_latents=latents,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
            top_z_seqs=top_z_seqs,
            top_z_lens=top_z_lens,
        )
        z = core_model.motion_condition_slice(conditions, seq_len)
        target = latents.clone().detach().reshape(bsz * seq_len, -1)
        z = z.reshape(bsz * seq_len, -1)
        _, pred_xstart = core_model.base_model.diff_loss(target=target, z=z)

    pred_xstart = pred_xstart.clone().detach().reshape(bsz, seq_len, -1)

    # Second forward: replace part of input with predicted x_start, compute grad.
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)
    updated_conditions = rag_model(
        motion_latents=updated_latents,
        text_emb=text_emb,
        top3_h_cls=top3_h_cls,
        top3_sim_scores=top3_sim_scores,
        cfg_drop_mask=cfg_drop_mask,
        empty_text_emb=empty_text_emb,
        top_z_seqs=top_z_seqs,
        top_z_lens=top_z_lens,
    )
    updated_z = core_model.motion_condition_slice(updated_conditions, seq_len)

    updated_target = latents.clone().detach().reshape(bsz * seq_len, -1).repeat(diffmlps_batch_mul, 1)
    updated_z = updated_z.reshape(bsz * seq_len, -1).repeat(diffmlps_batch_mul, 1)
    updated_target = updated_target[mask]
    updated_z = updated_z[mask]

    updated_loss, _ = core_model.base_model.diff_loss(target=updated_target, z=updated_z)
    return updated_loss


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision='bf16')
    comp_device = accelerator.device

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info(f'L_local={args.L_local}, block_size={78 + args.L_local}')

    train_loader = dataset_msa_rag_local.DATALoader(
        args.dataname,
        is_test=False,
        batch_size=args.batch_size,
        motion_latent_dir=args.latent_dir,
        text_latent_dir=args.text_latent_dir,
        hcls_dir=args.hcls_dir,
        z_latent_dir=args.z_latent_dir,
        topk=args.retrieval_topk,
        L_local=args.L_local,
        local_rag_dim=args.local_rag_dim,
        num_workers=args.num_workers,
        text_embed_dim=args.text_embed_dim,
    )

    # Load empty text embedding for CFG unconditional branch.
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        for p in [
            os.path.join(args.text_latent_dir, 'empty_text_embedding.npy'),
            os.path.join(args.text_latent_dir, 'empty_cfg_text_t5.npy'),
        ]:
            if os.path.exists(p):
                empty_text_path = p
                break

    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(f'Cannot find empty CFG text embedding: {args.empty_text_path}')

    import numpy as _np
    empty_text_emb = torch.from_numpy(_np.load(empty_text_path).astype('float32')).reshape(-1)
    if empty_text_emb.shape[0] != args.text_embed_dim:
        raise ValueError(
            f'empty text embedding dim should be {args.text_embed_dim}, got {empty_text_emb.shape[0]}'
        )

    # Build backbone + RAG wrapper with local RAG.
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78 + args.L_local   # extend block_size for local prefix tokens

    base_model = LLaMAHF(
        config,
        args.num_diffusion_head_layers,
        args.latent_dim,
        comp_device,
        generative_head_type=args.generative_head_type,
        num_flow_steps=args.num_flow_steps,
        flow_solver=args.flow_solver,
        rf_time_sampling=args.rf_time_sampling,
        rf_loss_type=args.rf_loss_type,
    )
    rag_model = LLaMARAGWrapper(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=args.disable_rag,
        L_local=args.L_local,
        local_rag_dim=args.local_rag_dim,
        add_selfatten=args.add_selfatten,
    )

    # Freeze non-trainable attention scale params to avoid DDP unused-param errors.
    frozen_scale_params = 0
    for n, p in rag_model.named_parameters():
        if '.attn.scale' in n:
            p.requires_grad = False
            frozen_scale_params += 1
    if frozen_scale_params > 0:
        logger.info(f'Frozen non-trainable attention scale params: {frozen_scale_params}')

    ema_base_state = None
    ema_rag_state = None

    if args.resume_trans is not None:
        logger.info(f'Loading checkpoint from {args.resume_trans}')
        ckpt = torch.load(args.resume_trans, map_location='cpu')

        def strip_module(sd):
            return {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}

        if 'trans' in ckpt:
            rag_model.base_model.load_state_dict(strip_module(ckpt['trans']), strict=False)
        if 'rag' in ckpt:
            rag_model.load_state_dict(strip_module(ckpt['rag']), strict=False)
        if args.use_ema:
            if 'trans_ema' in ckpt:
                ema_base_state = strip_module(ckpt['trans_ema'])
            if 'rag_ema' in ckpt:
                ema_rag_state = strip_module(ckpt['rag_ema'])

    rag_model.train()
    rag_model.to(comp_device)
    empty_text_emb = empty_text_emb.to(comp_device)

    optimizer = utils_model.initial_optim(
        args.decay_option, args.lr, args.weight_decay, rag_model, args.optimizer
    )
    scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter // 10, args.total_iter)

    rag_model, optimizer, train_loader = accelerator.prepare(rag_model, optimizer, train_loader)
    train_loader_iter = dataset_msa_rag_local.cycle(train_loader)

    ema_enabled = bool(args.use_ema)
    unwrapped_model = accelerator.unwrap_model(rag_model)
    if ema_enabled:
        if ema_base_state is None:
            ema_base_state = init_ema_state_dict(unwrapped_model.base_model)
        if ema_rag_state is None:
            ema_rag_state = init_ema_state_dict(unwrapped_model)
        if accelerator.is_main_process:
            logger.info(f'EMA enabled: decay={args.ema_decay}')

    train_forward_model = (
        rag_model.module if (args.num_gpus > 1 and hasattr(rag_model, 'module')) else rag_model
    )

    nb_iter, avg_loss = 0, 0.0
    print_iter = 100
    save_iter = 10000

    logger.info('Start training RAG+Local MotionStreamer...')

    while nb_iter <= args.total_iter:
        text_emb, top3_h_cls, top3_sim_scores, top_z_seqs, top_z_lens, m_tokens = next(train_loader_iter)

        text_emb       = text_emb.to(comp_device)
        top3_h_cls     = top3_h_cls.to(comp_device)
        top3_sim_scores = top3_sim_scores.to(comp_device)
        top_z_seqs      = top_z_seqs.to(comp_device)   # [B, K, T_max, 16]
        top_z_lens      = top_z_lens.to(comp_device)   # [B, K]
        m_tokens       = m_tokens.to(comp_device)

        m_tokens_len = estimate_lengths_from_padded_latents(m_tokens)
        input_latent = m_tokens[:, :-1]
        m_lens = torch.clamp(m_tokens_len, min=1, max=input_latent.shape[1])

        # Joint CFG dropout: all three conditions dropped together.
        cfg_drop_mask = torch.rand(text_emb.shape[0], device=comp_device) < args.cfg_dropout_prob

        loss = forward_loss_withmask_2_forward(
            latents=input_latent,
            rag_model=train_forward_model,
            m_lens=m_lens,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            top_z_seqs=top_z_seqs,
            top_z_lens=top_z_lens,
            step=nb_iter,
            total_steps=args.total_iter,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
            diffmlps_batch_mul=4,
        )

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(nb_iter)

        if ema_enabled and ((nb_iter + 1) % max(1, args.ema_update_every) == 0):
            update_ema_state_dict(ema_base_state, unwrapped_model.base_model, args.ema_decay)
            update_ema_state_dict(ema_rag_state, unwrapped_model, args.ema_decay)

        avg_loss += loss.item()
        nb_iter += 1

        if nb_iter % print_iter == 0 and accelerator.is_main_process:
            avg_loss /= print_iter
            writer.add_scalar('./Loss/train', avg_loss, nb_iter)
            writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
            writer.add_scalar('./CFG/drop_ratio', cfg_drop_mask.float().mean().item(), nb_iter)
            logger.info(f'Train. Iter {nb_iter} : loss {avg_loss:.5f}')
            avg_loss = 0.0

        if nb_iter % save_iter == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.out_dir, f'net_Iter{nb_iter:06d}.pth')
            payload = {
                'trans': accelerator.unwrap_model(rag_model).base_model.state_dict(),
                'rag': accelerator.unwrap_model(rag_model).state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': nb_iter,
                'generative_head_type': args.generative_head_type,
                'use_ema': ema_enabled,
                'ema_decay': args.ema_decay,
                'L_local': args.L_local,
                'add_selfatten': args.add_selfatten,
            }
            if ema_enabled:
                payload['trans_ema'] = ema_base_state
                payload['rag_ema'] = ema_rag_state
            torch.save(payload, ckpt_path)
            logger.info(f'Checkpoint saved to: {ckpt_path}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
        logger.info('Training finished.')
    accelerator.end_training()


if __name__ == '__main__':
    main()
