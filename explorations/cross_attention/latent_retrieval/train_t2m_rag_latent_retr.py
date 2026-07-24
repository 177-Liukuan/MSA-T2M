"""Train MotionStreamer stage-2 with global h_cls RAG + local motion-latent retrieval CA.

Extends train_t2m_rag.py:
1. Uses LLaMARAGLatentRetrWrapper (Flamingo CA on retrieved motion latents).
2. Uses dataset_msa_rag_latent_retr.DATALoader (per-caption text-to-text retrieval).
3. Passes retr_latents / retr_latent_lens through the two-forward training loop.
4. Supports offline library cache (--library_cache_dir) for fast startup.
5. ALL original files unchanged.
"""

import os
import sys
import argparse
import json
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from accelerate import Accelerator

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
from models.llama_rag_model_latent_retr import (
    LLaMARAGLatentRetrWrapper,
    LLaMARAGLatentRetrGatedWrapper,
)
from humanml3d_272 import dataset_msa_rag
from humanml3d_272 import dataset_msa_rag_latent_retr
import options.option_transformer as option_trans
import utils.utils_model as utils_model

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ---------------------------------------------------------------------------
#  LR scheduler (same as MCA variant)
# ---------------------------------------------------------------------------

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
        return {'warmup_iters': self.warmup_iters,
                'total_iters': self.total_iters, 'min_lr': self.min_lr}

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters  = state_dict['total_iters']
        self.min_lr       = state_dict['min_lr']


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--text_latent_dir', type=str,
                              default='./humanml3d_272/text_latents_t5')
    extra_parser.add_argument('--hcls_dir', type=str,
                              default='./humanml3d_272/h_cls_latents_msa_vae/exp')
    extra_parser.add_argument('--empty_text_path', type=str,
                              default='./humanml3d_272/text_latents_t5/empty_text_embedding.npy')
    extra_parser.add_argument('--retrieval_topk', type=int, default=3)
    extra_parser.add_argument('--latent_retr_topk', type=int, default=3,
                              help='Number of retrieved motion latents (local RAG).')
    extra_parser.add_argument('--latent_dim', type=int, default=16,
                              help='Motion latent dim for latent_retr_proj.')
    extra_parser.add_argument('--cfg_dropout_prob', type=float, default=0.1)
    extra_parser.add_argument('--retr_cfg_drop_prob', type=float, default=0.1,
                              help='Independent dropout prob for retrieved motion latents. '
                                   'Decoupled from text CFG dropout so the model learns '
                                   'to use each condition independently.')
    extra_parser.add_argument('--num_workers', type=int, default=0)
    extra_parser.add_argument('--text_embed_dim', type=int, default=768)
    extra_parser.add_argument('--disable_rag', action='store_true', default=False,
                              help='Ablation: disable global h_cls retrieval prefix.')
    extra_parser.add_argument('--disable_latent_retr', action='store_true', default=False,
                              help='Ablation: disable local motion-latent CA.')
    extra_parser.add_argument('--ca_every_n_layers', type=int, default=4,
                              help='Insert one cross-attention block every N transformer layers. '
                                   'E.g. 4 inserts at layers [3,7,11] for a 12-layer backbone.')
    extra_parser.add_argument('--ca_n_head', type=int, default=0,
                              help='CA heads (0 = same as backbone).')
    extra_parser.add_argument('--ca_insertion_mode', type=str, default='before_sa',
                              choices=['before_sa', 'after_sa', 'late_after_sa'],
                              help='CA insertion order relative to SA block. '
                                   'before_sa (A, default): CA→SA (original Flamingo). '
                                   'after_sa  (B): SA→CA across all layers. '
                                   'late_after_sa (C): first n//2 layers pure SA, '
                                   'then SA→CA every ca_every_n_layers.')
    extra_parser.add_argument('--ema_decay', type=float, default=0.9999)
    extra_parser.add_argument('--ema_update_every', type=int, default=1)
    extra_parser.add_argument('--disable_ema', action='store_true', default=False)
    extra_parser.add_argument('--freeze_backbone', action='store_true', default=False,
                              help='Flamingo-style: freeze LLaMA backbone, train CA only.')
    extra_parser.add_argument('--use_gated_ca', action='store_true', default=False,
                              help='Use GatedCrossAttentionBlock (Branch B) instead of gate-free.')
    extra_parser.add_argument('--library_cache_dir', type=str, default=None,
                              help='Dir for pre-built local-RAG library cache files. '
                                   'Build with build_latent_retr_library.py. '
                                   'If None, library is rebuilt from scratch each run.')
    extra_parser.add_argument('--precomputed_retr_dir', type=str, default=None,
                              help='Dir with pre-computed top-k retrieval results '
                                   '(one .npy per motion ID, shape (num_caps, topk)). '
                                   'Build with precompute_latent_retr_lookup.py. '
                                   'Eliminates per-sample CPU matmul → ~1.5-2× faster training.')
    custom_args, remaining = extra_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if 'ipykernel' in sys.modules:
        args = option_trans.get_args_parser()
    else:
        try:
            args = option_trans.get_args_parser()
        except SystemExit:
            args = extra_parser.parse_args([])

    args.text_latent_dir    = custom_args.text_latent_dir
    args.hcls_dir           = custom_args.hcls_dir
    args.empty_text_path    = custom_args.empty_text_path
    args.retrieval_topk     = custom_args.retrieval_topk
    args.latent_retr_topk   = custom_args.latent_retr_topk
    args.latent_dim         = custom_args.latent_dim
    args.cfg_dropout_prob   = custom_args.cfg_dropout_prob
    args.retr_cfg_drop_prob = custom_args.retr_cfg_drop_prob
    args.num_workers        = custom_args.num_workers
    args.text_embed_dim     = custom_args.text_embed_dim
    args.disable_rag        = custom_args.disable_rag
    args.disable_latent_retr = custom_args.disable_latent_retr
    args.ca_every_n_layers  = custom_args.ca_every_n_layers
    args.ca_n_head          = custom_args.ca_n_head
    args.ca_insertion_mode  = custom_args.ca_insertion_mode
    args.ema_decay          = custom_args.ema_decay
    args.ema_update_every   = custom_args.ema_update_every
    args.use_ema            = not custom_args.disable_ema
    args.freeze_backbone    = custom_args.freeze_backbone
    args.use_gated_ca       = custom_args.use_gated_ca
    args.library_cache_dir  = custom_args.library_cache_dir
    args.precomputed_retr_dir = custom_args.precomputed_retr_dir
    return args


# ---------------------------------------------------------------------------
#  EMA helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Training utilities
# ---------------------------------------------------------------------------

def lengths_to_mask(lengths, max_len):
    return (torch.arange(max_len, device=lengths.device)
            .expand(len(lengths), max_len) < lengths.unsqueeze(1))


def estimate_lengths_from_padded_latents(m_tokens):
    valid = m_tokens.abs().sum(dim=-1) > 0
    lens  = valid.long().sum(dim=1)
    return torch.clamp(lens, min=2)


def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    import math
    step        = min(step, total_steps)
    cosine_val  = 0.5 * (1 + math.cos(math.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_val


def replace_with_pred(latents, pred_xstart, step, total_steps):
    decay_factor = cosine_decay(step, total_steps)
    bsz, seq_len, _ = latents.shape
    num_replace = int(seq_len * decay_factor)
    replace_indices = torch.randperm(seq_len, device=latents.device)[:num_replace]
    replace_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=latents.device)
    replace_mask[:, replace_indices] = True
    updated = latents.clone()
    updated[replace_mask] = pred_xstart[replace_mask]
    return updated


def get_core_model(model):
    return model.module if hasattr(model, 'module') else model


def forward_loss_withmask_2_forward(
    latents, rag_model, m_lens,
    text_emb, top3_h_cls, top3_sim_scores,
    step, total_steps,
    cfg_drop_mask, empty_text_emb,
    diffmlps_batch_mul=4,
    retr_latents=None,
    retr_latent_lens=None,
    retr_cfg_drop_mask=None,   # independent retrieval dropout mask
):
    core_model = get_core_model(rag_model)
    bsz, seq_len, _ = latents.shape
    mask = (lengths_to_mask(m_lens, seq_len)
            .reshape(bsz * seq_len)
            .repeat(diffmlps_batch_mul))

    with torch.no_grad():
        conditions = rag_model(
            motion_latents=latents,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
            retr_latents=retr_latents,
            retr_latent_lens=retr_latent_lens,
            retr_cfg_drop_mask=retr_cfg_drop_mask,
        )
        z = core_model.motion_condition_slice(conditions, seq_len)
        target = latents.clone().detach().reshape(bsz * seq_len, -1)
        z = z.reshape(bsz * seq_len, -1)
        _, pred_xstart = core_model.base_model.diff_loss(target=target, z=z)

    pred_xstart   = pred_xstart.clone().detach().reshape(bsz, seq_len, -1)
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)

    updated_conditions = rag_model(
        motion_latents=updated_latents,
        text_emb=text_emb,
        top3_h_cls=top3_h_cls,
        top3_sim_scores=top3_sim_scores,
        cfg_drop_mask=cfg_drop_mask,
        empty_text_emb=empty_text_emb,
        retr_latents=retr_latents,
        retr_latent_lens=retr_latent_lens,
        retr_cfg_drop_mask=retr_cfg_drop_mask,
    )
    updated_z = core_model.motion_condition_slice(updated_conditions, seq_len)

    updated_target = (latents.clone().detach()
                      .reshape(bsz * seq_len, -1)
                      .repeat(diffmlps_batch_mul, 1))
    updated_z = updated_z.reshape(bsz * seq_len, -1).repeat(diffmlps_batch_mul, 1)

    updated_target = updated_target[mask]
    updated_z      = updated_z[mask]

    updated_loss, _ = core_model.base_model.diff_loss(
        target=updated_target, z=updated_z
    )
    return updated_loss


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator()
    comp_device = accelerator.device

    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info(f'Generative head type: {args.generative_head_type}')

    # ---- DataLoader ----
    train_loader = dataset_msa_rag_latent_retr.DATALoader(
        dataset_name=args.dataname,
        is_test=False,
        batch_size=args.batch_size,
        motion_latent_dir=args.latent_dir,
        text_latent_dir=args.text_latent_dir,
        hcls_dir=args.hcls_dir,
        topk=args.retrieval_topk,
        latent_retr_topk=args.latent_retr_topk,
        num_workers=args.num_workers,
        text_embed_dim=args.text_embed_dim,
        library_cache_dir=getattr(args, 'library_cache_dir', None),
        precomputed_retr_dir=getattr(args, 'precomputed_retr_dir', None),
    )

    # ---- Empty text embedding for CFG ----
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        for fallback in [
            os.path.join(args.text_latent_dir, 'empty_text_embedding.npy'),
            os.path.join(args.text_latent_dir, 'empty_cfg_text_t5.npy'),
        ]:
            if os.path.exists(fallback):
                empty_text_path = fallback
                break
    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(f'Cannot find empty CFG text embedding: {empty_text_path}')
    import numpy as _np
    empty_text_emb = torch.from_numpy(
        _np.load(empty_text_path).astype('float32')
    ).reshape(-1)

    # ---- Backbone + RAG wrapper ----
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(
        config,
        args.num_diffusion_head_layers,
        args.latent_dim,
        comp_device,
        generative_head_type=args.generative_head_type,
        num_flow_steps=getattr(args, 'num_flow_steps', 100),
        flow_solver=getattr(args, 'flow_solver', 'euler'),
        rf_time_sampling=getattr(args, 'rf_time_sampling', 'uniform'),
        rf_loss_type=getattr(args, 'rf_loss_type', 'simple'),
    )
    ca_n_head = getattr(args, 'ca_n_head', 0) or None
    _WrapperCls = (LLaMARAGLatentRetrGatedWrapper
                   if getattr(args, 'use_gated_ca', False)
                   else LLaMARAGLatentRetrWrapper)
    rag_model = _WrapperCls(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=args.disable_rag,
        latent_dim=getattr(args, 'latent_dim', 16),
        ca_every_n_layers=max(1, getattr(args, 'ca_every_n_layers', 4)),
        ca_n_head=ca_n_head,
        disable_latent_retr=getattr(args, 'disable_latent_retr', False),
        ca_insertion_mode=getattr(args, 'ca_insertion_mode', 'before_sa'),
    )

    # Freeze non-trainable attention scale params (avoids DDP unused-param error)
    for n, p in rag_model.named_parameters():
        if '.attn.scale' in n:
            p.requires_grad = False

    # Optional Flamingo-style backbone freeze
    if getattr(args, 'freeze_backbone', False):
        frozen = 0
        for n, p in rag_model.named_parameters():
            if any(k in n for k in [
                'base_model.transformer.h.',
                'base_model.transformer.wte.',
                'base_model.transformer.ln_f.',
            ]):
                p.requires_grad = False
                frozen += 1
        trainable = [n for n, p in rag_model.named_parameters() if p.requires_grad]
        logger.info(f'[Flamingo-style] Frozen {frozen} backbone param tensors. '
                    f'Trainable: {len(trainable)}')

    ema_base_state = None
    ema_rag_state  = None

    if args.resume_trans is not None:
        logger.info(f'Loading checkpoint from {args.resume_trans}')
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        if 'trans' in ckpt:
            sd = {(k.split('.', 1)[1] if k.split('.')[0] == 'module' else k): v
                  for k, v in ckpt['trans'].items()}
            rag_model.base_model.load_state_dict(sd, strict=False)
        if 'rag' in ckpt:
            sd = {(k.split('.', 1)[1] if k.split('.')[0] == 'module' else k): v
                  for k, v in ckpt['rag'].items()}
            rag_model.load_state_dict(sd, strict=False)
        if args.use_ema:
            if 'trans_ema' in ckpt:
                ema_base_state = {
                    (k.split('.', 1)[1] if k.split('.')[0] == 'module' else k): v
                    for k, v in ckpt['trans_ema'].items()
                }
            if 'rag_ema' in ckpt:
                ema_rag_state = {
                    (k.split('.', 1)[1] if k.split('.')[0] == 'module' else k): v
                    for k, v in ckpt['rag_ema'].items()
                }

    rag_model.train()
    rag_model.to(comp_device)
    empty_text_emb = empty_text_emb.to(comp_device)

    optimizer = utils_model.initial_optim(
        args.decay_option, args.lr, args.weight_decay, rag_model, args.optimizer
    )
    scheduler = WarmupCosineDecayScheduler(
        optimizer, args.total_iter // 10, args.total_iter
    )

    rag_model, optimizer, train_loader = accelerator.prepare(
        rag_model, optimizer, train_loader
    )
    train_loader_iter = dataset_msa_rag.cycle(train_loader)

    ema_enabled    = bool(args.use_ema)
    unwrapped_model = accelerator.unwrap_model(rag_model)
    if ema_enabled:
        if ema_base_state is None:
            ema_base_state = init_ema_state_dict(unwrapped_model.base_model)
        if ema_rag_state is None:
            ema_rag_state = init_ema_state_dict(unwrapped_model)

    train_forward_model = (
        rag_model.module
        if (getattr(args, 'num_gpus', 1) > 1 and hasattr(rag_model, 'module'))
        else rag_model
    )

    nb_iter, avg_loss, avg_loss_ddpm, avg_loss_rf = 0, 0.0, 0.0, 0.0
    print_iter = 100
    save_iter  = 10000

    msg = ('Start training latent-retrieval RAG MotionStreamer'
           + (' (disable_latent_retr ablation)' if args.disable_latent_retr else ''))
    logger.info(msg)

    while nb_iter <= args.total_iter:
        batch = next(train_loader_iter)
        # Batch tuple from dataset_msa_rag_latent_retr:
        # [0] text_emb, [1] top3_h_cls, [2] top3_sim_scores,
        # [3] motion_latents, [4] retr_latents, [5] retr_latent_lens
        text_emb        = batch[0].to(comp_device)
        top3_h_cls      = batch[1].to(comp_device)
        top3_sim_scores = batch[2].to(comp_device)
        m_tokens        = batch[3].to(comp_device)
        retr_latents    = batch[4].to(comp_device)
        retr_latent_lens = batch[5].to(comp_device)

        m_tokens_len = estimate_lengths_from_padded_latents(m_tokens)
        input_latent = m_tokens[:, :-1]
        m_lens = torch.clamp(m_tokens_len, min=1, max=input_latent.shape[1])

        cfg_drop_mask = (torch.rand(text_emb.shape[0], device=comp_device)
                         < args.cfg_dropout_prob)
        # Independent retrieval dropout: model learns each condition separately
        retr_cfg_drop_mask = (torch.rand(text_emb.shape[0], device=comp_device)
                              < getattr(args, 'retr_cfg_drop_prob', 0.1))

        loss = forward_loss_withmask_2_forward(
            latents=input_latent,
            rag_model=train_forward_model,
            m_lens=m_lens,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            step=nb_iter,
            total_steps=args.total_iter,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
            diffmlps_batch_mul=4,
            retr_latents=retr_latents,
            retr_latent_lens=retr_latent_lens,
            retr_cfg_drop_mask=retr_cfg_drop_mask,
        )

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(rag_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(nb_iter)

        if ema_enabled and ((nb_iter + 1) % max(1, args.ema_update_every) == 0):
            update_ema_state_dict(ema_base_state, unwrapped_model.base_model, args.ema_decay)
            update_ema_state_dict(ema_rag_state, unwrapped_model, args.ema_decay)

        avg_loss += loss.item()
        if args.generative_head_type == 'ddpm':
            avg_loss_ddpm += loss.item()
        else:
            avg_loss_rf += loss.item()
        nb_iter += 1

        if nb_iter % print_iter == 0 and accelerator.is_main_process:
            avg_loss     /= print_iter
            avg_loss_ddpm /= print_iter
            avg_loss_rf  /= print_iter
            writer.add_scalar('./Loss/total', avg_loss, nb_iter)
            writer.add_scalar('./Loss/ddpm', avg_loss_ddpm, nb_iter)
            writer.add_scalar('./Loss/rf', avg_loss_rf, nb_iter)
            writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
            logger.info(
                f'Train. Iter {nb_iter} : '
                f'loss_total {avg_loss:.5f} | '
                f'loss_ddpm {avg_loss_ddpm:.5f} | '
                f'loss_rf {avg_loss_rf:.5f}'
            )
            avg_loss = avg_loss_ddpm = avg_loss_rf = 0.0

        if nb_iter % save_iter == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.out_dir, f'net_Iter{nb_iter:06d}.pth')
            payload = {
                'trans':     accelerator.unwrap_model(rag_model).base_model.state_dict(),
                'rag':       accelerator.unwrap_model(rag_model).state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter':      nb_iter,
                'generative_head_type': args.generative_head_type,
                'use_ema':   ema_enabled,
                'ema_decay': args.ema_decay,
            }
            if ema_enabled:
                payload['trans_ema'] = ema_base_state
                payload['rag_ema']   = ema_rag_state
            torch.save(payload, ckpt_path)
            logger.info(f'Checkpoint saved to: {ckpt_path}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
        logger.info('Training finished.')
    accelerator.end_training()


if __name__ == '__main__':
    main()
