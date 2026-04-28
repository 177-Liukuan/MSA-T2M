"""
build_rag_db.py
===============
使用训练好的 Q-Former 对 HumanML3D 训练集的所有运动序列提取全局 RAG Token，
并构建全局 RAG 检索数据库（global_rag DB）。

数据库结构
----------
  global_rag/
    <exp_name>/          # 与 Q-Former checkpoint 所在实验目录保持一致
      keys.npy           # [N, text_emb_dim]  文本 embedding（T5 sentence embedding）
      values.npy         # [N, num_queries * query_dim]  全局 RAG Token（展平）
      values_mean.npy    # [N, query_dim]  全局 RAG Token（mean pool）
      meta.json          # 每个条目的元信息（name, caption, motion_path 等）
      args.json          # 构建时的所有参数

使用示例
--------
  python build_rag_db.py \\
      --qformer-ckpt Experiments/QFormer_t2m_272_v1/net_best_r1.pth \\
      --tae-ckpt Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth \\
      --split train
"""

import os
import sys
import json
import argparse
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.tae import Causal_HumanTAE
from models.qformer_for_motion import MotionQFormer
from train_qformer_rag import TAEFeatureExtractor, MotionCaptionDataset


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 参数
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='构建全局 RAG DB')

    # 模型检查点
    p.add_argument('--qformer-ckpt', required=True, type=str,
                   help='Q-Former checkpoint 路径')
    p.add_argument('--tae-ckpt',     required=True, type=str,
                   help='Causal TAE checkpoint 路径')

    # 数据
    p.add_argument('--dataname',    default='t2m_272', type=str)
    p.add_argument('--data-root',   default='./humanml3d_272', type=str)
    p.add_argument('--text-latent-dir', default='./humanml3d_272/text_latents_t5', type=str)
    p.add_argument('--split',       default='train', type=str,
                   choices=['train', 'val', 'test'])
    p.add_argument('--max-motion-len', default=300, type=int)
    p.add_argument('--batch-size',  default=32, type=int)
    p.add_argument('--num-workers', default=4,  type=int)

    # TAE 架构参数（与训练时一致）
    p.add_argument('--down-t',      default=2,   type=int)
    p.add_argument('--depth',       default=3,   type=int)
    p.add_argument('--dilation-growth-rate', default=3, type=int)
    p.add_argument('--hidden-size', default=1024, type=int)
    p.add_argument('--latent-dim',  default=16,  type=int)

    # Q-Former 架构参数（与训练时一致）
    p.add_argument('--num-queries', default=4,   type=int)
    p.add_argument('--query-dim',   default=768, type=int)
    p.add_argument('--motion-dim',  default=1024, type=int)
    p.add_argument('--num-layers',  default=6,   type=int)
    p.add_argument('--num-heads',   default=8,   type=int)
    p.add_argument('--dropout',     default=0.0, type=float)   # 推理不用 dropout
    p.add_argument('--text-emb-dim',default=768, type=int)
    p.add_argument('--t5-model-path', default='sentencet5-xxl/', type=str,
                   help='本地 sentence-T5 模型目录（用于 MTM/MTG tokenizer）')
    p.add_argument('--max-text-len',default=64,  type=int)

    # 输出
    p.add_argument('--db-root',     default='global_rag', type=str,
                   help='RAG DB 根目录')
    p.add_argument('--pool-method', default='flatten',
                   choices=['flatten', 'mean'],
                   help='全局 RAG Token 的池化方式')
    p.add_argument('--device',      default='cuda', type=str)
    p.add_argument('--seed',        default=42, type=int)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 推理专用数据集（需要保留所有 caption，并返回所有 T5 embeddings）
# ═══════════════════════════════════════════════════════════════════════════════

class MotionCaptionInferDataset(Dataset):
    """
    每个 motion 返回所有 caption 及其 T5 embedding，
    用于构建 RAG DB（每条 caption 作为一个独立条目）。
    """

    def __init__(self, data_root, text_latent_dir, split='train',
                 max_motion_len=300):
        self.motion_dir      = pjoin(data_root, 'motion_data')
        self.text_dir        = pjoin(data_root, 'texts')
        self.text_latent_dir = text_latent_dir
        self.max_motion_len  = max_motion_len

        mean = np.load(pjoin(data_root, 'mean_std', 'Mean.npy')).astype(np.float32)
        std  = np.load(pjoin(data_root, 'mean_std', 'Std.npy' )).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.mean = mean
        self.std  = std

        split_file = pjoin(data_root, 'split', f'{split}.txt')
        with cs.open(split_file, 'r') as f:
            id_list = [l.strip() for l in f.readlines()]

        # 展开为 (name, motion_path, caption, text_emb) 条目
        self.items = []
        for name in tqdm(id_list, desc=f'Indexing {split}'):
            motion_path  = pjoin(self.motion_dir, name + '.npy')
            latent_path  = pjoin(text_latent_dir, name + '.npy')
            caption_path = pjoin(self.text_dir,   name + '.txt')
            if not all(os.path.exists(p) for p in [motion_path, latent_path, caption_path]):
                continue

            motion = np.load(motion_path).astype(np.float32)
            if motion.shape[0] < 16:
                continue

            embs = np.load(latent_path).astype(np.float32)  # [N_cap, 768]

            captions = []
            with cs.open(caption_path, 'r') as f:
                for line in f.readlines():
                    cap = line.strip().split('#')[0]
                    if cap:
                        captions.append(cap)
            if not captions:
                continue

            for cap_idx, caption in enumerate(captions):
                emb_idx  = min(cap_idx, embs.shape[0] - 1)
                text_emb = embs[emb_idx]
                self.items.append({
                    'name':         name,
                    'motion_path':  motion_path,
                    'caption':      caption,
                    'text_emb':     text_emb,
                    'cap_idx':      cap_idx,
                })

        print(f'[InferDataset] {split} items (motion x caption): {len(self.items)}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        motion = np.load(item['motion_path']).astype(np.float32)
        motion = (motion - self.mean) / self.std
        if motion.shape[0] > self.max_motion_len:
            motion = motion[:self.max_motion_len]
        T = motion.shape[0]
        return (
            torch.from_numpy(motion),           # [T, 272]
            torch.from_numpy(item['text_emb']), # [768]
            item['caption'],
            item['name'],
            item['cap_idx'],
            T,
        )


def collate_infer(batch):
    batch.sort(key=lambda x: x[5], reverse=True)
    motions, text_embs, captions, names, cap_idxs, lengths = zip(*batch)
    max_T = max(lengths)
    B     = len(motions)
    D     = motions[0].shape[-1]

    motion_padded = torch.zeros(B, max_T, D)
    motion_mask   = torch.ones(B, max_T, dtype=torch.bool)
    for i, (m, l) in enumerate(zip(motions, lengths)):
        motion_padded[i, :l] = m
        motion_mask  [i, :l] = False

    return (
        motion_padded,
        torch.stack(text_embs),
        list(captions),
        list(names),
        list(cap_idxs),
        list(lengths),
        motion_mask,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # ── 确定输出目录（与 qformer checkpoint 实验目录一致）────────────────────
    # args.qformer_ckpt 形如 Experiments/QFormer_t2m_272_v1/net_best_r1.pth
    # 取倒数第 2 段作为 exp_name
    exp_name = os.path.basename(os.path.dirname(os.path.abspath(args.qformer_ckpt)))
    db_dir   = pjoin(args.db_root, exp_name)
    os.makedirs(db_dir, exist_ok=True)
    print(f'RAG DB 输出目录: {db_dir}')

    # ── 保存构建参数 ──────────────────────────────────────────────────────────
    with open(pjoin(db_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # ── 加载 TAE（冻结）──────────────────────────────────────────────────────
    clip_range = [-30, 20]
    tae = Causal_HumanTAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        latent_dim=args.latent_dim,
        clip_range=clip_range,
    )
    ckpt = torch.load(args.tae_ckpt, map_location='cpu')
    state = ckpt['net'] if 'net' in ckpt else ckpt
    tae.load_state_dict(state, strict=True)
    tae.to(device)
    tae.eval()
    print(f'TAE 已加载: {args.tae_ckpt}')

    tae_extractor = TAEFeatureExtractor(tae)
    tae_extractor.to(device)

    # ── 加载 Q-Former ─────────────────────────────────────────────────────────
    qformer = MotionQFormer(
        num_queries   = args.num_queries,
        query_dim     = args.query_dim,
        motion_dim    = args.motion_dim,
        num_layers    = args.num_layers,
        num_heads     = args.num_heads,
        dropout       = args.dropout,
        t5_model_path = args.t5_model_path,
        text_emb_dim  = args.text_emb_dim,
        max_text_len  = args.max_text_len,
    )
    ckpt_q = torch.load(args.qformer_ckpt, map_location='cpu')
    state_q = ckpt_q['qformer'] if 'qformer' in ckpt_q else ckpt_q
    qformer.load_state_dict(state_q, strict=True)
    qformer.to(device)
    qformer.eval()
    print(f'Q-Former 已加载: {args.qformer_ckpt}')

    # ── 数据集 ────────────────────────────────────────────────────────────────
    dataset = MotionCaptionInferDataset(
        args.data_root, args.text_latent_dir,
        split=args.split, max_motion_len=args.max_motion_len)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_infer, drop_last=False)

    # ── 提取特征，构建 DB ─────────────────────────────────────────────────────
    all_keys      = []   # text embedding [N, D_text]
    all_values    = []   # flattened RAG token [N, Nq*Dq]
    all_values_mp = []   # mean-pooled RAG token [N, Dq]
    meta_list     = []

    print('正在提取 Q-Former 特征...')
    with torch.no_grad():
        for batch in tqdm(loader):
            (motion_padded, text_emb, captions,
             names, cap_idxs, lengths, motion_mask) = batch

            motion_padded = motion_padded.to(device)
            text_emb      = text_emb.to(device)
            motion_mask   = motion_mask.to(device)

            # 提取 TAE 中间特征
            feat, feat_mask = tae_extractor.extract(motion_padded, motion_mask)

            # 提取 Q-Former 全局 RAG Token
            Z_out = qformer.forward_features(feat, feat_mask)  # [B, Nq, Dq]

            B = Z_out.size(0)

            # 文本 embedding（已预计算，直接使用）
            keys = F.normalize(text_emb, dim=-1).cpu().numpy()  # [B, 768]

            # 值：展平 or mean-pool
            values_flat = Z_out.view(B, -1).cpu().numpy()        # [B, Nq*Dq]
            values_mean = Z_out.mean(dim=1).cpu().numpy()        # [B, Dq]

            all_keys.append(keys)
            all_values.append(values_flat)
            all_values_mp.append(values_mean)

            for i in range(B):
                meta_list.append({
                    'name':    names[i],
                    'caption': captions[i],
                    'cap_idx': cap_idxs[i],
                    'length':  lengths[i],
                })

    # ── 拼接并保存 ────────────────────────────────────────────────────────────
    all_keys      = np.concatenate(all_keys,      axis=0)
    all_values    = np.concatenate(all_values,    axis=0)
    all_values_mp = np.concatenate(all_values_mp, axis=0)

    print(f'\nDB 统计:')
    print(f'  条目数     : {len(all_keys)}')
    print(f'  keys 形状  : {all_keys.shape}   (text embedding)')
    print(f'  values 形状: {all_values.shape}  (RAG token flattened)')
    print(f'  values_mean: {all_values_mp.shape} (RAG token mean-pool)')

    np.save(pjoin(db_dir, 'keys.npy'),        all_keys)
    np.save(pjoin(db_dir, 'values.npy'),       all_values)
    np.save(pjoin(db_dir, 'values_mean.npy'),  all_values_mp)

    with open(pjoin(db_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f'\n全局 RAG DB 已保存至: {db_dir}')
    print('文件清单:')
    for fname in sorted(os.listdir(db_dir)):
        fpath = pjoin(db_dir, fname)
        size  = os.path.getsize(fpath) / 1024 / 1024
        print(f'  {fname:<25s}  {size:.2f} MB')


if __name__ == '__main__':
    main()
