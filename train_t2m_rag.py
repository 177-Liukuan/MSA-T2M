"""Train MotionStreamer stage-2 with RAG-guided conditioning.

Core design:
1. Offline-only features (numpy.load): text_emb + h_cls + motion_latents
2. Retrieval fusion: Top-3 h_cls -> weighted single retrieval token
3. Joint CFG dropout (text + retrieval) with null retrieval parameter
4. Keep original Two-Forward diffusion training strategy
"""

import os
import math
import json
import torch
import warnings
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from accelerate import Accelerator

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
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
    args = option_trans.get_args_parser()

    # RAG-specific args
    if not hasattr(args, 'text_latent_dir'):
        args.text_latent_dir = './humanml3d_272/text_latents_clip'
    if not hasattr(args, 'hcls_dir'):
        args.hcls_dir = './humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000_α0'
    if not hasattr(args, 'empty_text_path'):
        args.empty_text_path = './humanml3d_272/text_latents_clip/empty_cfg_text.npy'
    if not hasattr(args, 'retrieval_topk'):
        args.retrieval_topk = 3
    if not hasattr(args, 'cfg_dropout_prob'):
        args.cfg_dropout_prob = 0.1
    if not hasattr(args, 'num_workers'):
        args.num_workers = 0

    return args


def lengths_to_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


def estimate_lengths_from_padded_latents(m_tokens):
    """Estimate valid lengths from zero-padded latent sequences."""
    valid = m_tokens.abs().sum(dim=-1) > 0
    lens = valid.long().sum(dim=1)
    lens = torch.clamp(lens, min=2)
    return lens


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

    updated_latents = latents.clone()
    updated_latents[replace_mask] = pred_xstart[replace_mask]
    return updated_latents


def get_core_model(model):
    """Return the underlying model for both single-GPU and DDP cases."""
    return model.module if hasattr(model, 'module') else model


def forward_loss_withmask_2_forward(
    latents,
    rag_model,
    m_lens,
    text_emb,
    top3_h_cls,
    top3_sim_scores,
    step,
    total_steps,
    cfg_drop_mask,
    empty_text_emb,
    diffmlps_batch_mul=4,
):
    """Two-forward training with RAG condition and diffusion denoising loss."""
    core_model = get_core_model(rag_model)

    # First forward with [text_token, retrieval_token, motion_tokens...]
    conditions = rag_model(
        motion_latents=latents,
        text_emb=text_emb,
        top3_h_cls=top3_h_cls,
        top3_sim_scores=top3_sim_scores,
        cfg_drop_mask=cfg_drop_mask,
        empty_text_emb=empty_text_emb,
    )

    bsz, seq_len, _ = latents.shape
    mask = lengths_to_mask(m_lens, seq_len).reshape(bsz * seq_len).repeat(diffmlps_batch_mul)

    # 关键对齐：双条件 token 时，motion[0] 对应 hidden index=1。
    z = core_model.motion_condition_slice(conditions, seq_len)

    target = latents.clone().detach().reshape(bsz * seq_len, -1)
    z = z.reshape(bsz * seq_len, -1)

    with torch.no_grad():
        _, pred_xstart = core_model.base_model.diff_loss(target=target, z=z)

    pred_xstart = pred_xstart.clone().detach().reshape(bsz, seq_len, -1)

    # Second forward: replace part of input with predicted x_start
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)
    updated_conditions = rag_model(
        motion_latents=updated_latents,
        text_emb=text_emb,
        top3_h_cls=top3_h_cls,
        top3_sim_scores=top3_sim_scores,
        cfg_drop_mask=cfg_drop_mask,
        empty_text_emb=empty_text_emb,
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

    # Enforce bf16-compatible accelerate runtime.
    accelerator = Accelerator(mixed_precision='bf16')
    comp_device = accelerator.device

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    train_loader = dataset_msa_rag.DATALoader(
        args.dataname,
        is_test=False,
        batch_size=args.batch_size,
        motion_latent_dir=args.latent_dir,
        text_latent_dir=args.text_latent_dir,
        hcls_dir=args.hcls_dir,
        topk=args.retrieval_topk,
        num_workers=args.num_workers,
    )

    # Load precomputed empty text embedding for CFG unconditional branch.
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        fallback_path = os.path.join(args.text_latent_dir, 'empty_cfg_text_clip.npy')
        if os.path.exists(fallback_path):
            empty_text_path = fallback_path
        else:
            raise FileNotFoundError(
                f'Cannot find empty CFG text embedding. checked: {args.empty_text_path} and {fallback_path}'
            )

    empty_text_emb = torch.from_numpy(__import__('numpy').load(empty_text_path).astype('float32')).reshape(-1)
    if empty_text_emb.shape[0] != 512:
        raise ValueError(f'empty text embedding dim should be 512, got {empty_text_emb.shape[0]}')

    # Backbone + RAG wrapper
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    rag_model = LLaMARAGWrapper(base_model=base_model, text_dim=512, retrieval_dim=512, model_dim=config.n_embd)

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

    rag_model.train()
    rag_model.to(comp_device)
    empty_text_emb = empty_text_emb.to(comp_device)

    optimizer = utils_model.initial_optim(
        args.decay_option,
        args.lr,
        args.weight_decay,
        rag_model,
        args.optimizer,
    )
    scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter // 10, args.total_iter)

    rag_model, optimizer, train_loader = accelerator.prepare(rag_model, optimizer, train_loader)
    train_loader_iter = dataset_msa_rag.cycle(train_loader)

    nb_iter, avg_loss = 0, 0.0
    print_iter = 100
    save_iter = 10000

    logger.info('Start training RAG-guided MotionStreamer...')

    while nb_iter <= args.total_iter:
        text_emb, top3_h_cls, top3_sim_scores, m_tokens = next(train_loader_iter)

        text_emb = text_emb.to(comp_device)
        top3_h_cls = top3_h_cls.to(comp_device)
        top3_sim_scores = top3_sim_scores.to(comp_device)
        m_tokens = m_tokens.to(comp_device)

        # Estimate valid lengths from zero padding, then align with input_latent = m_tokens[:, :-1].
        m_tokens_len = estimate_lengths_from_padded_latents(m_tokens)
        input_latent = m_tokens[:, :-1]
        m_lens = torch.clamp(m_tokens_len, min=1, max=input_latent.shape[1])

        # Joint CFG dropout: drop text and retrieval together for 10% samples.
        cfg_drop_mask = torch.rand(text_emb.shape[0], device=comp_device) < args.cfg_dropout_prob

        if args.num_gpus > 1:
            loss = forward_loss_withmask_2_forward(
                latents=input_latent,
                rag_model=rag_model.module,
                m_lens=m_lens,
                text_emb=text_emb,
                top3_h_cls=top3_h_cls,
                top3_sim_scores=top3_sim_scores,
                step=nb_iter,
                total_steps=args.total_iter,
                cfg_drop_mask=cfg_drop_mask,
                empty_text_emb=empty_text_emb,
                diffmlps_batch_mul=4,
            )
        else:
            loss = forward_loss_withmask_2_forward(
                latents=input_latent,
                rag_model=rag_model,
                m_lens=m_lens,
                text_emb=text_emb,
                top3_h_cls=top3_h_cls,
                top3_sim_scores=top3_sim_scores,
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

        avg_loss += loss.item()
        nb_iter += 1

        if nb_iter % print_iter == 0:
            if accelerator.is_main_process:
                avg_loss = avg_loss / print_iter
                writer.add_scalar('./Loss/train', avg_loss, nb_iter)
                writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
                writer.add_scalar('./CFG/drop_ratio', cfg_drop_mask.float().mean().item(), nb_iter)
                logger.info(f'Train. Iter {nb_iter} : Loss {avg_loss:.5f}')
            avg_loss = 0.0

        if nb_iter % save_iter == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.out_dir, 'latest.pth')
            torch.save(
                {
                    'trans': accelerator.unwrap_model(rag_model).base_model.state_dict(),
                    'rag': accelerator.unwrap_model(rag_model).state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': nb_iter,
                },
                ckpt_path,
            )
            logger.info(f'Checkpoint saved to: {ckpt_path}')

    accelerator.wait_for_everyone()
    logger.info('Training finished.')


if __name__ == '__main__':
    main()
