"""Train MotionStreamer stage-2 with RAG-guided conditioning.

Core design:
1. Offline-only features (numpy.load): text_emb + h_cls + motion_latents
2. Retrieval fusion: Top-3 h_cls -> weighted single retrieval token
3. Joint CFG dropout (text + retrieval) with null retrieval parameter
4. Keep original Two-Forward diffusion training strategy
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
from models.rag_training import (
    RAGTwoForwardLoss,
    estimate_lengths_from_padded_latents,
    get_rag_model,
)
from humanml3d_272 import dataset_msa_rag
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
            optimizer,
            T_max=total_iters - warmup_iters,
            eta_min=min_lr,
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
        return {
            'warmup_iters': self.warmup_iters,
            'total_iters': self.total_iters,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters = state_dict['total_iters']
        self.min_lr = state_dict['min_lr']


def parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--text_latent_dir', type=str, default='./humanml3d_272/text_latents_t5')
    extra_parser.add_argument('--hcls_dir', type=str, default='./humanml3d_272/h_cls_latents_msa_vae/exp')
    extra_parser.add_argument('--empty_text_path', type=str, default='./humanml3d_272/text_latents_t5/empty_text_embedding.npy')
    extra_parser.add_argument('--retrieval_topk', type=int, default=3)
    extra_parser.add_argument('--cfg_dropout_prob', type=float, default=0.1)
    extra_parser.add_argument('--num_workers', type=int, default=4)
    extra_parser.add_argument('--text_embed_dim', type=int, default=768)
    extra_parser.add_argument(
        '--cache_mode',
        choices=('reference', 'packed'),
        default='reference',
    )
    extra_parser.add_argument('--cache_dir', type=str, default=None)
    extra_parser.add_argument('--disable_rag', action='store_true', default=False, help='Ablation: disable retrieval token and use text-only conditioning.')
    extra_parser.add_argument('--ema_decay', type=float, default=0.9999)
    extra_parser.add_argument('--ema_update_every', type=int, default=1)
    extra_parser.add_argument('--disable_ema', action='store_true', default=False)

    custom_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    args.text_latent_dir = custom_args.text_latent_dir
    args.hcls_dir = custom_args.hcls_dir
    args.empty_text_path = custom_args.empty_text_path
    args.retrieval_topk = custom_args.retrieval_topk
    args.cfg_dropout_prob = custom_args.cfg_dropout_prob
    args.num_workers = custom_args.num_workers
    args.text_embed_dim = custom_args.text_embed_dim
    args.cache_mode = custom_args.cache_mode
    args.cache_dir = custom_args.cache_dir
    args.disable_rag = custom_args.disable_rag
    args.ema_decay = custom_args.ema_decay
    args.ema_update_every = custom_args.ema_update_every
    args.use_ema = not custom_args.disable_ema

    return args




def init_ema_state_dict(model):
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.detach().clone()
    return state


def update_ema_state_dict(ema_state, model, decay):
    model_state = model.state_dict()
    one_minus = 1.0 - decay
    for k, v in model_state.items():
        if not torch.is_tensor(v):
            continue
        if k not in ema_state:
            ema_state[k] = v.detach().clone()
            continue
        if torch.is_floating_point(v):
            ema_state[k].mul_(decay).add_(v.detach(), alpha=one_minus)
        else:
            ema_state[k].copy_(v.detach())

def build_checkpoint_payload(
    training_model,
    accelerator,
    optimizer,
    scheduler,
    iteration,
    generative_head_type,
    ema_enabled,
    ema_decay,
    ema_base_state=None,
    ema_rag_state=None,
):
    research_model = get_rag_model(accelerator.unwrap_model(training_model))
    payload = {
        'trans': research_model.base_model.state_dict(),
        'rag': research_model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': iteration,
        'generative_head_type': generative_head_type,
        'use_ema': ema_enabled,
        'ema_decay': ema_decay,
    }
    if ema_enabled:
        payload['trans_ema'] = ema_base_state
        payload['rag_ema'] = ema_rag_state
    return payload


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    # Enforce bf16-compatible accelerate runtime.
    accelerator = Accelerator(mixed_precision='bf16')
    comp_device = accelerator.device

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info(f'Generative head type: {args.generative_head_type}')

    train_loader = dataset_msa_rag.DATALoader(
        args.dataname,
        is_test=False,
        batch_size=args.batch_size,
        motion_latent_dir=args.latent_dir,
        text_latent_dir=args.text_latent_dir,
        hcls_dir=args.hcls_dir,
        topk=args.retrieval_topk,
        num_workers=args.num_workers,
        text_embed_dim=args.text_embed_dim,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
    )

    # Load precomputed empty text embedding for CFG unconditional branch.
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        fallback_candidates = [
            os.path.join(args.text_latent_dir, 'empty_text_embedding.npy'),
            os.path.join(args.text_latent_dir, 'empty_cfg_text_t5.npy'),
            os.path.join(args.text_latent_dir, 'empty_cfg_text_clip.npy'),
        ]
        for p in fallback_candidates:
            if os.path.exists(p):
                empty_text_path = p
                break

    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(
            f'Cannot find empty CFG text embedding. checked: {args.empty_text_path} and defaults under {args.text_latent_dir}'
        )

    empty_text_emb = torch.from_numpy(__import__('numpy').load(empty_text_path).astype('float32')).reshape(-1)
    if empty_text_emb.shape[0] != args.text_embed_dim:
        raise ValueError(
            f'empty text embedding dim should be {args.text_embed_dim}, got {empty_text_emb.shape[0]} from {empty_text_path}'
        )

    # Backbone + RAG wrapper
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
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
    rag_model = LLaMARAGWrapper(base_model=base_model, model_dim=config.n_embd, disable_rag=args.disable_rag)

    # attn.scale is used via .item() in attention; it will not receive gradients.
    # Freeze it explicitly to avoid DDP unused-parameter reduction errors.
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

        if 'trans' in ckpt:
            new_ckpt_trans = {}
            for key, value in ckpt['trans'].items():
                if key.split('.')[0] == 'module':
                    new_key = '.'.join(key.split('.')[1:])
                else:
                    new_key = key
                new_ckpt_trans[new_key] = value
            rag_model.base_model.load_state_dict(new_ckpt_trans, strict=False)

        if 'rag' in ckpt:
            new_ckpt_rag = {}
            for key, value in ckpt['rag'].items():
                if key.split('.')[0] == 'module':
                    new_key = '.'.join(key.split('.')[1:])
                else:
                    new_key = key
                new_ckpt_rag[new_key] = value
            rag_model.load_state_dict(new_ckpt_rag, strict=False)

        if args.use_ema:
            if 'trans_ema' in ckpt:
                ema_base_state = {('.'.join(k.split('.')[1:]) if k.split('.')[0] == 'module' else k): v for k, v in ckpt['trans_ema'].items()}
            if 'rag_ema' in ckpt:
                ema_rag_state = {('.'.join(k.split('.')[1:]) if k.split('.')[0] == 'module' else k): v for k, v in ckpt['rag_ema'].items()}

    rag_model.train()
    rag_model.to(comp_device)
    empty_text_emb = empty_text_emb.to(comp_device)
    training_model = RAGTwoForwardLoss(rag_model, diffmlps_batch_mul=4)

    optimizer = utils_model.initial_optim(
        args.decay_option,
        args.lr,
        args.weight_decay,
        training_model,
        args.optimizer,
    )
    scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter // 10, args.total_iter)

    training_model, optimizer, train_loader = accelerator.prepare(
        training_model, optimizer, train_loader
    )
    train_loader_iter = dataset_msa_rag.cycle(train_loader)

    ema_enabled = bool(args.use_ema)
    research_model = get_rag_model(accelerator.unwrap_model(training_model))
    if ema_enabled:
        if ema_base_state is None:
            ema_base_state = init_ema_state_dict(research_model.base_model)
        if ema_rag_state is None:
            ema_rag_state = init_ema_state_dict(research_model)
        if accelerator.is_main_process:
            logger.info(f'EMA enabled: decay={args.ema_decay}, update_every={args.ema_update_every}')

    nb_iter, avg_loss, avg_loss_ddpm, avg_loss_rf = 0, 0.0, 0.0, 0.0
    print_iter = 100
    save_iter = 10000

    logger.info('Start training no-RAG ablation MotionStreamer...' if args.disable_rag else 'Start training RAG-guided MotionStreamer...')

    while nb_iter <= args.total_iter:
        text_emb, top3_h_cls, top3_sim_scores, m_tokens = next(train_loader_iter)

        text_emb = text_emb.to(comp_device, non_blocking=True)
        top3_h_cls = top3_h_cls.to(comp_device, non_blocking=True)
        if text_emb.shape[-1] != args.text_embed_dim:
            raise ValueError(f'text_emb dim mismatch: got {text_emb.shape[-1]}, expected {args.text_embed_dim}')
        if top3_h_cls.shape[-1] != args.text_embed_dim:
            raise ValueError(f'top3_h_cls dim mismatch: got {top3_h_cls.shape[-1]}, expected {args.text_embed_dim}')
        top3_sim_scores = top3_sim_scores.to(comp_device, non_blocking=True)
        m_tokens = m_tokens.to(comp_device, non_blocking=True)

        # Estimate valid lengths from zero padding, then align with input_latent = m_tokens[:, :-1].
        m_tokens_len = estimate_lengths_from_padded_latents(m_tokens)
        input_latent = m_tokens[:, :-1]
        m_lens = torch.clamp(m_tokens_len, min=1, max=input_latent.shape[1])

        # Joint CFG dropout: drop text and retrieval together for 10% samples.
        cfg_drop_mask = torch.rand(text_emb.shape[0], device=comp_device) < args.cfg_dropout_prob

        loss = training_model(
            latents=input_latent,
            m_lens=m_lens,
            text_emb=text_emb,
            top3_h_cls=top3_h_cls,
            top3_sim_scores=top3_sim_scores,
            step=nb_iter,
            total_steps=args.total_iter,
            cfg_drop_mask=cfg_drop_mask,
            empty_text_emb=empty_text_emb,
        )

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(nb_iter)

        if ema_enabled and ((nb_iter + 1) % max(1, args.ema_update_every) == 0):
            update_ema_state_dict(ema_base_state, research_model.base_model, args.ema_decay)
            update_ema_state_dict(ema_rag_state, research_model, args.ema_decay)

        loss_value = loss.item()
        avg_loss += loss_value
        if args.generative_head_type == 'ddpm':
            avg_loss_ddpm += loss_value
        else:
            avg_loss_rf += loss_value
        nb_iter += 1

        if nb_iter % print_iter == 0:
            if accelerator.is_main_process:
                avg_loss = avg_loss / print_iter
                avg_loss_ddpm = avg_loss_ddpm / print_iter
                avg_loss_rf = avg_loss_rf / print_iter
                writer.add_scalar('./Loss/total', avg_loss, nb_iter)
                writer.add_scalar('./Loss/ddpm', avg_loss_ddpm, nb_iter)
                writer.add_scalar('./Loss/rf', avg_loss_rf, nb_iter)
                writer.add_scalar('./Loss/train', avg_loss, nb_iter)
                writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
                writer.add_scalar('./CFG/drop_ratio', cfg_drop_mask.float().mean().item(), nb_iter)
                logger.info(f'Train. Iter {nb_iter} : loss_total {avg_loss:.5f} | loss_ddpm {avg_loss_ddpm:.5f} | loss_rf {avg_loss_rf:.5f}')
            avg_loss = 0.0
            avg_loss_ddpm = 0.0
            avg_loss_rf = 0.0

        if nb_iter % save_iter == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.out_dir, f'net_Iter{nb_iter:06d}.pth')
            payload = build_checkpoint_payload(
                training_model=training_model,
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                iteration=nb_iter,
                generative_head_type=args.generative_head_type,
                ema_enabled=ema_enabled,
                ema_decay=args.ema_decay,
                ema_base_state=ema_base_state,
                ema_rag_state=ema_rag_state,
            )
            torch.save(payload, ckpt_path)
            logger.info(f'Checkpoint saved to: {ckpt_path}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
        logger.info('Training finished.')
    accelerator.end_training()


if __name__ == '__main__':
    main()
