"""
train_qformer_rag.py
====================
训练 Q-Former 实现运动-文本跨模态对齐，用于全局 RAG Token 生成。

损失函数
--------
  L_total = L_MTC + L_MTM + L_MTG

运行示例
--------
  accelerate launch --num_processes 4 train_qformer_rag.py \\
      --tae-ckpt Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth \\
      --exp-name QFormer_t2m_272_v1 \\
      --batch-size 64
"""

import os
import sys
import json
import math
import random
import argparse
import warnings
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─── 项目路径 ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.utils_model as utils_model
from models.tae import Causal_HumanTAE
from models.qformer_for_motion import MotionQFormer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 参数
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='Q-Former RAG 训练')

    # 数据
    p.add_argument('--dataname',    default='t2m_272', type=str)
    p.add_argument('--data-root',   default='./humanml3d_272', type=str)
    p.add_argument('--text-latent-dir', default='./humanml3d_272/text_latents_t5', type=str,
                   help='预计算的 T5 sentence embedding 目录 (npy, shape [N_cap, 768])')
    p.add_argument('--max-motion-len', default=300, type=int)
    p.add_argument('--batch-size',  default=64,  type=int)
    p.add_argument('--num-workers', default=4,   type=int)

    # TAE 编码器
    p.add_argument('--tae-ckpt',    required=True, type=str,
                   help='Causal TAE 检查点路径')
    p.add_argument('--down-t',      default=2,   type=int)
    p.add_argument('--depth',       default=3,   type=int)
    p.add_argument('--dilation-growth-rate', default=3, type=int)
    p.add_argument('--hidden-size', default=1024, type=int)
    p.add_argument('--latent-dim',  default=16,  type=int)

    # Q-Former 超参
    p.add_argument('--num-queries', default=4,   type=int)
    p.add_argument('--query-dim',   default=768, type=int)
    p.add_argument('--motion-dim',  default=1024, type=int)
    p.add_argument('--num-layers',  default=6,   type=int)
    p.add_argument('--num-heads',   default=8,   type=int)
    p.add_argument('--dropout',     default=0.1, type=float)
    p.add_argument('--queue-size',  default=4096, type=int,
                   help='Negative Queue 大小（0=禁用）。推荐 4096，有效负样本数=batch+queue')
    p.add_argument('--text-emb-dim',default=768, type=int,
                   help='T5 sentence embedding 维度（sentence-T5-large=768）')
    p.add_argument('--t5-model-path', default='sentencet5-xxl/', type=str,
                   help='本地 sentence-T5 模型目录（用于 MTM/MTG tokenizer）')
    p.add_argument('--max-text-len',default=64,  type=int)

    # 优化器
    p.add_argument('--lr',          default=1e-4, type=float)
    p.add_argument('--weight-decay',default=1e-4, type=float)
    p.add_argument('--total-iter',  default=500000, type=int)
    p.add_argument('--warm-up-iter',default=2000,  type=int)
    p.add_argument('--lr-scheduler',default=[200000, 400000], nargs='+', type=int)
    p.add_argument('--gamma',       default=0.1,  type=float)
    p.add_argument('--lr-sched-type', default='cosine', type=str,
                   choices=['multistep', 'cosine'],
                   help='LR 调度策略: cosine=余弦退火, multistep=分段衰减')
    p.add_argument('--lr-eta-min-ratio', default=0.01, type=float,
                   help='CosineAnnealingLR 最小 lr = lr * ratio')
    p.add_argument('--early-stop-patience', default=0, type=int,
                   help='早停 patience（评估次数），0=禁用。如 --early-stop-patience 6 表示连续 6 次 eval 无改善即停止')

    # 损失权重
    p.add_argument('--w-mtc',  default=1.0, type=float)
    p.add_argument('--w-mtm',  default=1.0, type=float)
    p.add_argument('--w-mtg',  default=1.0, type=float)
    p.add_argument('--mtm-neg-ratio', default=0.5, type=float,
                   help='负样本比例（MTM hard negative mining）')

    # 输出 & 日志
    p.add_argument('--out-dir',    default='Experiments', type=str)
    p.add_argument('--exp-name',   default='QFormer_t2m_272', type=str)
    p.add_argument('--print-iter', default=100,   type=int)
    p.add_argument('--eval-iter',  default=5000,  type=int)
    p.add_argument('--save-iter',  default=50000, type=int)
    p.add_argument('--resume-pth', default=None,  type=str)
    p.add_argument('--seed',       default=42,    type=int)
    p.add_argument('--num-gpus',   default=1,     type=int)
    p.add_argument('--grad-accum', default=1,     type=int,
                   help='梯度累积步数（等效 batch = batch_size × grad_accum）')
    p.add_argument('--t5-cache-batch', default=16, type=int,
                   help='T5 caption 缓存构建时的 batch size（4090: 16，H100: 256）')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════════════════

class MotionCaptionDataset(Dataset):
    """
    返回 (motion_npy, text_embedding, caption_str, motion_id)

    motion_npy : np.ndarray [T, 272]（已标准化）
    text_embedding : np.ndarray [768]（随机选一条 caption 的 T5 embedding）
    caption_str : str（与 text_embedding 对应的原始 caption）
    """

    def __init__(self, data_root, text_latent_dir, split='train',
                 max_motion_len=300, deterministic=False):
        self.motion_dir    = pjoin(data_root, 'motion_data')
        self.text_dir      = pjoin(data_root, 'texts')
        self.text_latent_dir = text_latent_dir
        self.max_motion_len  = max_motion_len
        self.deterministic   = deterministic  # val 时固定选第 0 条 caption，消除评估噪声

        mean = np.load(pjoin(data_root, 'mean_std', 'Mean.npy')).astype(np.float32)
        std  = np.load(pjoin(data_root, 'mean_std', 'Std.npy' )).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.mean = mean
        self.std  = std

        split_file = pjoin(data_root, 'split', f'{split}.txt')
        with cs.open(split_file, 'r') as f:
            id_list = [l.strip() for l in f.readlines()]

        self.data = []
        for name in tqdm(id_list, desc=f'Loading {split} split'):
            motion_path  = pjoin(self.motion_dir,  name + '.npy')
            latent_path  = pjoin(text_latent_dir,  name + '.npy')
            caption_path = pjoin(self.text_dir,    name + '.txt')
            if not (os.path.exists(motion_path) and
                    os.path.exists(latent_path) and
                    os.path.exists(caption_path)):
                continue
            motion = np.load(motion_path).astype(np.float32)
            if motion.shape[0] < 16:   # 太短则跳过
                continue
            # 读取所有 caption
            captions = []
            with cs.open(caption_path, 'r') as f:
                for line in f.readlines():
                    cap = line.strip().split('#')[0]
                    if cap:
                        captions.append(cap)
            if not captions:
                continue
            self.data.append({
                'name': name,
                'motion': motion,        # cached in RAM — avoids NFS re-read each epoch
                'motion_path': motion_path,
                'latent_path': latent_path,
                'captions': captions,
            })
        print(f'[Dataset] {split} samples: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion']   # from RAM cache (loaded once in __init__), no NFS I/O
        # 标准化
        motion = (motion - self.mean) / self.std
        # 截断
        if motion.shape[0] > self.max_motion_len:
            start = random.randint(0, motion.shape[0] - self.max_motion_len)
            motion = motion[start: start + self.max_motion_len]
        T = motion.shape[0]

        # 选 caption：训练时随机，val/test 时确定性（idx % n_cap），消除评估噪声
        n_cap = len(item['captions'])
        if self.deterministic:
            cap_idx = idx % n_cap
        else:
            cap_idx = random.randint(0, n_cap - 1)
        caption = item['captions'][cap_idx]

        # 加载对应 T5 embedding
        embs = np.load(item['latent_path']).astype(np.float32)  # [N_cap, 768]
        # 对应 caption 的 embedding（N_cap 可能 < len(captions)，取 min）
        emb_idx = min(cap_idx, embs.shape[0] - 1)
        text_emb = embs[emb_idx]  # [768]

        return (
            torch.from_numpy(motion),   # [T, 272]
            torch.from_numpy(text_emb), # [768]
            caption,                    # str
            item['name'],               # str
            T,                          # int（用于排序）
        )


def collate_fn(batch):
    """变长 motion 序列 padding + mask。"""
    batch.sort(key=lambda x: x[4], reverse=True)
    motions, text_embs, captions, names, lengths = zip(*batch)

    max_T = max(lengths)
    B     = len(motions)
    D     = motions[0].shape[-1]

    motion_padded = torch.zeros(B, max_T, D)
    motion_mask   = torch.ones(B, max_T, dtype=torch.bool)  # True=padding
    for i, (m, l) in enumerate(zip(motions, lengths)):
        motion_padded[i, :l] = m
        motion_mask  [i, :l] = False

    return (
        motion_padded,                    # [B, T_max, 272]
        torch.stack(text_embs),           # [B, 768]
        list(captions),                   # List[str]
        list(names),                      # List[str]
        list(lengths),                    # List[int]
        motion_mask,                      # [B, T_max] True=padding
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAE 中间特征提取（hook）
# ═══════════════════════════════════════════════════════════════════════════════

class TAEFeatureExtractor(nn.Module):
    """
    冻结 Causal_HumanTAE 编码器，利用 forward hook 提取
    最后一个 MLP (self.proj) 之前的 1024 维中间特征。

    该特征形状：[B, T', 1024]，T' = T // 4（down_t=2 时）。
    """

    def __init__(self, tae: Causal_HumanTAE):
        super().__init__()
        self.tae = tae
        # 冻结所有参数
        for p in self.tae.parameters():
            p.requires_grad = False
        self.tae.eval()

        self._feat = None
        # hook 挂在 encoder 的 model（Sequential）最后一层输出，即 proj 之前
        self.tae.tae.encoder.model.register_forward_hook(self._hook)

    def _hook(self, module, inp, output):
        # output shape: [B, 1024, T']  (channels first, Conv1d 输出)
        # 转换为 [B, T', 1024]
        self._feat = output.transpose(1, 2)

    @torch.no_grad()
    def extract(self, x_motion: torch.Tensor,
                motion_mask: torch.Tensor = None):
        """
        x_motion    : [B, T_max, 272]  已标准化的 motion，含 padding
        motion_mask : [B, T_max]       True = padding 位置

        返回
        ----
        feat        : [B, T', 1024]    T' = T_max // 4
        feat_mask   : [B, T']          True = padding 位置
        """
        self.tae.eval()
        # TAE preprocess: [B, 272, T]
        x_in = x_motion.permute(0, 2, 1).float()
        # 触发 forward（含 hook）
        _ = self.tae.tae.encoder.model(x_in)
        feat = self._feat                   # [B, T', 1024]

        # 对齐 motion_mask 到 T'
        feat_mask = None
        if motion_mask is not None:
            # 简单地用平均池化将 mask 下采样到 T'
            T_prime = feat.shape[1]
            # [B, T_max] -> [B, 1, T_max] -> avg_pool -> [B, T']
            m_float = motion_mask.float().unsqueeze(1)
            m_down  = F.adaptive_avg_pool1d(m_float, T_prime).squeeze(1)
            feat_mask = (m_down > 0.5)   # True = 大多是 padding

        return feat, feat_mask


# ═══════════════════════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════════════════════

def loss_mtc(motion_feat, text_feat, temp, accelerator=None, return_sim=False,
             motion_queue=None, text_queue=None):
    """
    Max Pool InfoNCE 对比损失（双向，带 label smoothing + optional negative queue）。
    motion_feat  : [B, Nq, D]  per-token L2-normed
    text_feat    : [B, D]      L2-normed
    motion_queue : [Q, Nq, D]  队列中的 motion 负样本（detached）
    text_queue   : [Q, D]      队列中的 text 负样本（detached）
    return_sim   : 若 True，额外返回 sim_i2t_local [B, B_all]（仅 in-batch 部分）供 hard neg mining
    """
    t = temp.clamp(min=1e-4)
    if accelerator is not None and accelerator.num_processes > 1:
        all_motion = accelerator.gather(motion_feat)   # [B_all, Nq, D]
        all_text   = accelerator.gather(text_feat)     # [B_all, D]
    else:
        all_motion = motion_feat
        all_text   = text_feat

    B_local = motion_feat.size(0)
    B_all   = all_motion.size(0)

    # 拼接 queue 负样本（queue 作为额外负样本，不参与正样本匹配）
    if text_queue is not None and motion_queue is not None:
        all_text_ext   = torch.cat([all_text,   text_queue.clone().detach()],   dim=0)  # [B_all+Q, D]
        all_motion_ext = torch.cat([all_motion, motion_queue.clone().detach()], dim=0)  # [B_all+Q, Nq, D]
    else:
        all_text_ext   = all_text
        all_motion_ext = all_motion

    # sim_q2t[b, a, q] = dot(motion[b,q], text_ext[a])  → [B, B_all+Q, Nq]
    sim_q2t = torch.einsum('bqd,ad->baq', motion_feat, all_text_ext)
    sim_i2t, _ = sim_q2t.max(dim=-1)    # [B, B_all+Q]
    sim_i2t = sim_i2t / t

    # sim_t2q[b, a, q] = dot(text[b], motion_ext[a,q])  → [B, B_all+Q, Nq]
    sim_t2q = torch.einsum('bd,aqd->baq', text_feat, all_motion_ext)
    sim_t2i, _ = sim_t2q.max(dim=-1)    # [B, B_all+Q]
    sim_t2i = sim_t2i / t

    if accelerator is not None and accelerator.num_processes > 1:
        labels = torch.arange(B_local, device=motion_feat.device) + accelerator.process_index * B_local
    else:
        labels = torch.arange(B_local, device=motion_feat.device)

    loss_m2t = F.cross_entropy(sim_i2t, labels, label_smoothing=0.1)
    loss_t2m = F.cross_entropy(sim_t2i, labels, label_smoothing=0.1)
    loss = (loss_m2t + loss_t2m) / 2

    if return_sim:
        # 只返回 in-batch 相似度矩阵（用于 hard neg mining，避免 queue 索引越界）
        return loss, sim_i2t[:, :B_all]
    return loss


def loss_mtm(itm_logits, labels):
    """
    二元交叉熵：motion-text 匹配。
    itm_logits : [B, 2]
    labels     : [B]  1=匹配 0=不匹配
    """
    return F.cross_entropy(itm_logits, labels)


def loss_mtg(lm_logits, input_ids, tokenizer, pad_token_id=0):
    """
    自回归文本生成损失（预测下一个 token）。
    lm_logits : [B, L, vocab_size]
    input_ids : [B, L]
    """
    # shift: 预测 [1:] 位置
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
    )
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# T5 hidden 缓存工具
# ═══════════════════════════════════════════════════════════════════════════════

def build_caption_cache(train_dataset, text_enc, max_text_len, device, batch_size=128):
    """
    启动前一次性运行冻结 T5 encoder，将所有 unique caption 的
    last_hidden_state 缓存到 CPU RAM（FP16），显著加速训练迭代。

    存储量估算: N_unique × 64 × 1024 × 2B ≈ 70K × 128KB ≈ 9 GB
    （H100 服务器 512GB RAM，完全可行）
    """
    all_caps = sorted({cap for item in train_dataset.data
                       for cap in item['captions']})
    cache = {}
    for i in tqdm(range(0, len(all_caps), batch_size),
                  desc='T5 pre-encoding captions'):
        batch = all_caps[i: i + batch_size]
        with torch.no_grad():
            enc    = text_enc.tokenize(batch, max_text_len, device)
            hidden = text_enc.encode_to_hidden(
                enc['input_ids'], enc.get('attention_mask'))  # [B, L, 1024] FP16
        for j, cap in enumerate(batch):
            cache[cap] = {
                'hidden': hidden[j].cpu(),                     # FP16
                'ids':    enc['input_ids'][j].cpu(),           # int64
                'mask':   enc['attention_mask'][j].cpu(),      # int64
            }
    return cache


def lookup_hidden(captions, cache, device):
    """从 CPU 缓存批量拉取并移至 GPU。"""
    return (
        torch.stack([cache[c]['hidden'] for c in captions]).to(device),  # [B,L,1024]
        torch.stack([cache[c]['ids']    for c in captions]).to(device),  # [B,L]
        torch.stack([cache[c]['mask']   for c in captions]).to(device),  # [B,L]
    )


def build_tae_cache(dataset, tae_extractor, device, batch_size=64):
    """
    启动前一次性运行冻结 TAE encoder，将所有 motion 的中间特征缓存到 CPU RAM。

    - 取每条 motion 的前 max_motion_len 帧（固定，无随机裁剪）
    - 对比学习任务中固定帧范围有助于特征一致性
    - 存储量估算: 14k × avg_T'=40 × 1024 × 4B ≈ 2.3 GB（H100 服务器完全可行）
    """
    cache = {}
    items = dataset.data
    max_motion_len = dataset.max_motion_len
    mean, std = dataset.mean, dataset.std

    for i in tqdm(range(0, len(items), batch_size), desc='TAE pre-computing'):
        batch = items[i: i + batch_size]
        motions_norm, names = [], []
        for item in batch:
            m = item['motion']                      # from RAM cache
            m = (m - mean) / std                    # normalize
            if m.shape[0] > max_motion_len:
                m = m[:max_motion_len]              # fixed first-N-frames crop
            motions_norm.append(m)
            names.append(item['name'])

        T_max = max(m.shape[0] for m in motions_norm)
        B, D  = len(motions_norm), motions_norm[0].shape[-1]
        m_pad  = torch.zeros(B, T_max, D, device=device)
        m_mask = torch.ones (B, T_max, dtype=torch.bool, device=device)
        for j, m in enumerate(motions_norm):
            T = m.shape[0]
            m_pad [j, :T] = torch.from_numpy(m).to(device)
            m_mask[j, :T] = False

        with torch.no_grad():
            feat, feat_mask = tae_extractor.extract(m_pad, m_mask)

        for j, name in enumerate(names):
            # 截去 padding 部分，只保存有效 T' 长度
            if feat_mask is not None:
                valid_len = int((~feat_mask[j]).sum().item())
            else:
                valid_len = feat.shape[1]
            cache[name] = feat[j, :valid_len, :].cpu()   # [T', 1024] float32

    return cache


def lookup_tae(names, tae_cache, device):
    """从 CPU 缓存批量拉取 TAE 特征并填充到同尺寸张量。"""
    feats   = [tae_cache[n] for n in names]
    max_T   = max(f.shape[0] for f in feats)
    B, D    = len(feats), feats[0].shape[-1]
    feat      = torch.zeros(B, max_T, D,    device=device)
    feat_mask = torch.ones (B, max_T, dtype=torch.bool, device=device)
    for j, f in enumerate(feats):
        T = f.shape[0]
        feat[j, :T]      = f.to(device)
        feat_mask[j, :T] = False
    return feat, feat_mask


@torch.no_grad()
def evaluate(qformer, tae_feat_extractor, val_loader, val_cache, pad_tok_id, device, args, accelerator,
             tae_cache=None):
    """
    评估指标（使用 val_cache 避免 T5 GPU 前向）：
      - MTC Recall@1 (motion->text)
      - MTM Accuracy
      - MTG Perplexity
    """
    qformer.eval()
    all_motion_feats, all_text_feats = [], []
    mtm_correct, mtm_total = 0, 0
    mtg_loss_sum, mtg_tok_count = 0.0, 0

    for batch in val_loader:
        if batch is None:          # accelerate 包装空 DataLoader 时可能 yield None
            continue
        motion_padded, text_emb, captions, names, lengths, motion_mask = batch
        motion_padded = motion_padded.to(device)
        text_emb      = text_emb.to(device)
        motion_mask   = motion_mask.to(device)

        if tae_cache is not None:
            feat, feat_mask = lookup_tae(names, tae_cache, device)
        else:
            feat, feat_mask = tae_feat_extractor.extract(motion_padded, motion_mask)

        # MTC feats（text_emb 来自预计算 sentence embedding，无需 T5）
        mf, tf, temp = qformer.forward_mtc(feat, text_emb, feat_mask)
        all_motion_feats.append(mf.cpu())
        all_text_feats.append(tf.cpu())

        # 从 CPU cache 取 hidden（T5 已在 CPU，无 GPU forward 开销）
        # 若 val_cache 没有该 caption（罕见），则跳过 MTM/MTG
        missing = [c for c in captions if c not in val_cache]
        if missing:
            continue

        pos_hid, pos_ids, pos_mask = lookup_hidden(captions, val_cache, device)

        # MTM: only positive pairs
        pos_logits = qformer.forward_mtm_from_hidden(feat, pos_hid, pos_mask, feat_mask)
        labels = torch.ones(pos_logits.size(0), dtype=torch.long, device=device)
        mtm_correct += (pos_logits.argmax(dim=-1) == labels).sum().item()
        mtm_total   += pos_logits.size(0)

        # MTG
        lm_logits, input_ids = qformer.forward_mtg_from_hidden(
            feat, pos_ids, pos_hid, pos_mask, feat_mask)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_g = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_tok_id, reduction='sum')
        valid_toks = (shift_labels != pad_tok_id).sum().item()
        mtg_loss_sum  += loss_g.item()
        mtg_tok_count += valid_toks

    # MTC R@1
    # val_loader 为空（latent 文件尚未生成）时直接返回占位值
    if not all_motion_feats:
        qformer.train()
        return 0.0, 0.0, float("inf")

    all_mf = torch.cat(all_motion_feats, dim=0)   # [N, Nq, D]
    all_tf = torch.cat(all_text_feats,   dim=0)   # [N, D]
    N_eval, Nq_eval = all_mf.shape[0], all_mf.shape[1]
    # Max Pool sim：每个 motion 的 Nq token 与所有 text 的最大相似度
    sim_q  = torch.matmul(all_mf.reshape(N_eval * Nq_eval, -1), all_tf.T)   # [N*Nq, N]
    sims, _ = sim_q.reshape(N_eval, Nq_eval, N_eval).max(dim=1)              # [N, N]
    ranks  = (sims.argsort(dim=1, descending=True) ==
              torch.arange(len(all_mf)).unsqueeze(1)).float()
    r1     = ranks[:, 0].mean().item()

    mtm_acc  = mtm_correct / max(mtm_total, 1)
    ppl      = math.exp(mtg_loss_sum / max(mtg_tok_count, 1))

    qformer.train()
    return r1, mtm_acc, ppl


# ═══════════════════════════════════════════════════════════════════════════════
# 学习率调度（warmup）
# ═══════════════════════════════════════════════════════════════════════════════

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * min(1.0, nb_iter / warm_up_iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    accelerator = Accelerator()
    device      = accelerator.device

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    out_dir = pjoin(args.out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = utils_model.get_logger(out_dir)
    writer = SummaryWriter(out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ── 数据集 ────────────────────────────────────────────────────────────────
    train_dataset = MotionCaptionDataset(
        args.data_root, args.text_latent_dir,
        split='train', max_motion_len=args.max_motion_len)
    val_dataset = MotionCaptionDataset(
        args.data_root, args.text_latent_dir,
        split='val', max_motion_len=args.max_motion_len,
        deterministic=True)  # 评估时固定 caption，消除 R@1 随机噪声

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=True, pin_memory=True,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=False, pin_memory=True,
        persistent_workers=(args.num_workers > 0))

    # ── TAE 编码器（冻结）────────────────────────────────────────────────────
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
    logger.info(f'TAE loaded from {args.tae_ckpt}')

    tae_extractor = TAEFeatureExtractor(tae)
    tae_extractor.to(device)

    # ── Q-Former ──────────────────────────────────────────────────────────────
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
        queue_size    = args.queue_size,
    )
    qformer.to(device)
    logger.info(f'Q-Former trainable params: '
                f'{sum(p.numel() for p in qformer.parameters() if p.requires_grad)/1e6:.2f}M')

    # ── 优化器 & 调度器 ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, qformer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    # Resume（先加载 qformer/optimizer 状态；scheduler 在 accelerator.prepare 后创建）
    start_iter = 0
    _saved_scheduler_state = None
    if args.resume_pth:
        ckpt_r = torch.load(args.resume_pth, map_location='cpu')
        qformer.load_state_dict(ckpt_r['qformer'])
        optimizer.load_state_dict(ckpt_r['optimizer'])
        start_iter = ckpt_r.get('iter', 0)
        if 'scheduler' in ckpt_r:
            _saved_scheduler_state = ckpt_r['scheduler']
        logger.info(f'Resumed from {args.resume_pth}, iter={start_iter}')

    # ── 预计算 T5 hidden states（启动时一次性，之后训练不再调用 T5）────────────
    if accelerator.is_main_process:
        logger.info('预计算所有 caption 的 T5 hidden states（约 1-2 分钟）...')
    _text_enc = qformer.text_encoder
    cap_cache = build_caption_cache(
        train_dataset, _text_enc, args.max_text_len, device, batch_size=args.t5_cache_batch)
    val_cache = build_caption_cache(
        val_dataset, _text_enc, args.max_text_len, device, batch_size=args.t5_cache_batch)
    if accelerator.is_main_process:
        logger.info(f'T5 hidden 预计算完成，共 {len(cap_cache)} 条 train / {len(val_cache)} 条 val unique caption')

    # ── 将 T5 移回 CPU，释放 ~22GB VRAM 用于训练 ─────────────────────────────
    _text_enc.t5.to('cpu')
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        logger.info('T5 encoder 已移至 CPU，显存已释放')

    # ── 预计算 TAE 中间特征（启动时一次性，之后训练不再调用 TAE encoder）────────
    if accelerator.is_main_process:
        logger.info('预计算所有 motion 的 TAE 中间特征（约 1 分钟）...')
    train_tae_cache = build_tae_cache(
        train_dataset, tae_extractor, device, batch_size=64)
    val_tae_cache = build_tae_cache(
        val_dataset, tae_extractor, device, batch_size=64)
    if accelerator.is_main_process:
        logger.info(f'TAE 预计算完成：{len(train_tae_cache)} train / '
                    f'{len(val_tae_cache)} val samples cached')

    # ── accelerate prepare ───────────────────────────────────────────────────
    qformer, optimizer, train_loader, val_loader = accelerator.prepare(
        qformer, optimizer, train_loader, val_loader)

    # ── LR 调度器（在 prepare 后创建，绑定到 accelerate 包装的 optimizer）──────
    # Bug fix: 旧版在 prepare 前创建 scheduler，导致 scheduler.optimizer 指向原始
    # 未包装对象，而训练循环中日志读取的是包装后的 optimizer.param_groups，
    # 两者实际上共享同一 param_groups dict，但 MultiStepLR.__init__ 调用 step()
    # 的时机早于 prepare，可能在某些 accelerate 版本下出现 last_epoch 不同步。
    # 将 scheduler 移至 prepare 后，确保绑定关系正确。
    _sched_steps_done = max(start_iter - args.warm_up_iter, 0)
    _sched_T = max(args.total_iter - args.warm_up_iter, 1)
    if args.lr_sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=_sched_T,
            eta_min=args.lr * args.lr_eta_min_ratio,
            last_epoch=_sched_steps_done - 1)
        logger.info(f'LR scheduler: CosineAnnealingLR  T_max={_sched_T}  '
                    f'eta_min={args.lr * args.lr_eta_min_ratio:.2e}')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_scheduler, gamma=args.gamma,
            last_epoch=_sched_steps_done - 1)
        logger.info(f'LR scheduler: MultiStepLR  milestones={args.lr_scheduler}')
    if _saved_scheduler_state is not None:
        scheduler.load_state_dict(_saved_scheduler_state)
        logger.info('LR scheduler state restored from checkpoint')

    pad_tok_id = qformer.module.text_encoder.tokenizer.pad_token_id \
        if hasattr(qformer, 'module') else \
        qformer.text_encoder.tokenizer.pad_token_id
    if pad_tok_id is None:
        pad_tok_id = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    nb_iter          = start_iter
    epoch            = 0
    avg_loss         = {'total': 0., 'mtc': 0., 'mtm': 0., 'mtg': 0.}
    best_r1          = 0.0
    no_improve_count = 0   # 早停计数器

    logger.info('==== 开始训练 ====')

    while nb_iter < args.total_iter:
        epoch += 1
        for batch in train_loader:
            if nb_iter >= args.total_iter:
                break

            motion_padded, text_emb, captions, names, lengths, motion_mask = batch
            motion_padded = motion_padded.to(device)
            text_emb      = text_emb.to(device)
            motion_mask   = motion_mask.to(device)

            # 从预计算缓存获取 TAE 特征（无需每步重跑 Conv1D encoder）
            feat, feat_mask = lookup_tae(names, train_tae_cache, device)

            # ── 学习率 warmup ────────────────────────────────────────────────
            if nb_iter < args.warm_up_iter:
                update_lr_warm_up(optimizer, nb_iter + 1, args.warm_up_iter, args.lr)

            # ── MTC 损失 ─────────────────────────────────────────────────────
            mf, tf, temp = (qformer.module if hasattr(qformer, 'module') else qformer)\
                            .forward_mtc(feat, text_emb, feat_mask)
            qf_module = qformer.module if hasattr(qformer, 'module') else qformer
            mq = qf_module.motion_queue if qf_module.queue_size > 0 else None
            tq = qf_module.text_queue   if qf_module.queue_size > 0 else None
            l_mtc, sim_i2t = loss_mtc(mf, tf, temp, accelerator, return_sim=True,
                                      motion_queue=mq, text_queue=tq)
            # 每步更新 queue（用当前 batch 特征，detach 避免梯度流入 queue）
            qf_module.dequeue_and_enqueue(mf, tf)

            # ── MTM / MTG 损失（使用预计算 T5 hidden，跳过 T5 encoder 调用）────
            B = feat.size(0)
            # Hard Negative Mining：基于 sim_i2t 采样困难负样本（单 GPU）
            if accelerator.num_processes == 1:
                with torch.no_grad():
                    sim_neg = sim_i2t.clone().detach()
                    sim_neg.fill_diagonal_(-10000)          # 屏蔽正样本对角线
                    weights = F.softmax(sim_neg, dim=1)     # 归一化为概率
                    hard_neg_idx = torch.multinomial(weights, 1).squeeze(1)  # [B]
                neg_captions = [captions[i] for i in hard_neg_idx.cpu().tolist()]
            else:
                # 多 GPU 时 sim_i2t 跨 rank，索引对应全局 batch，fallback 随机
                neg_idx_rand = torch.randperm(B, device=device)
                neg_captions = [captions[i] for i in neg_idx_rand.cpu().tolist()]

            # 从 CPU 缓存拉取（FP16 hidden，无 T5 forward 开销）
            pos_hid, pos_ids, pos_mask = lookup_hidden(captions,     cap_cache, device)
            neg_hid, neg_ids, neg_mask = lookup_hidden(neg_captions, cap_cache, device)

            qf = qformer.module if hasattr(qformer, 'module') else qformer
            # 将 pos 和 neg 在 batch 维拼接，一次前向传播完成（原来 2 次）
            feat_2x      = torch.cat([feat, feat], dim=0)        # [2B, T', D]
            fm_2x        = torch.cat([feat_mask, feat_mask], dim=0)                            if feat_mask is not None else None
            all_mtm_hid  = torch.cat([pos_hid, neg_hid], dim=0) # [2B, L, D]
            all_mtm_mask = torch.cat([pos_mask, neg_mask], dim=0)
            all_mtm_logits = qf.forward_mtm_from_hidden(
                feat_2x, all_mtm_hid, all_mtm_mask, fm_2x)      # [2B, 2]
            pos_logits = all_mtm_logits[:B]
            neg_logits = all_mtm_logits[B:]

            mtm_logits = torch.cat([pos_logits, neg_logits], dim=0)
            mtm_labels = torch.cat([
                torch.ones(B,  dtype=torch.long, device=device),
                torch.zeros(B, dtype=torch.long, device=device),
            ], dim=0)
            l_mtm = loss_mtm(mtm_logits, mtm_labels)

            # ── MTG 损失（复用 pos hidden，proj 仍参与梯度）────────────────────
            # 当 w_mtg==0 时跳过前向传播，避免浪费 1/4 的 Q-Former compute
            if args.w_mtg > 0.0:
                lm_logits, input_ids = qf.forward_mtg_from_hidden(
                    feat, pos_ids, pos_hid, pos_mask, feat_mask)
                l_mtg = loss_mtg(lm_logits, input_ids, None, pad_tok_id)
            else:
                l_mtg = torch.zeros(1, device=device)

            loss = (args.w_mtc * l_mtc +
                    args.w_mtm * l_mtm +
                    args.w_mtg * l_mtg)

            # ── 梯度累积 ─────────────────────────────────────────────────
            loss_scaled = loss / args.grad_accum
            accelerator.backward(loss_scaled)

            if nb_iter % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    (qformer.module if hasattr(qformer, 'module') else qformer).parameters(),
                    1.0)
                optimizer.step()
                optimizer.zero_grad()

            # scheduler 每 iter 步进（与 grad_accum 无关），使 milestone 与 nb_iter 对应
            if nb_iter >= args.warm_up_iter:
                scheduler.step()

            nb_iter += 1

            # 累计日志
            avg_loss['total'] += loss.item()
            avg_loss['mtc']   += l_mtc.item()
            avg_loss['mtm']   += l_mtm.item()
            avg_loss['mtg']   += l_mtg.item()

            if nb_iter % args.print_iter == 0 and accelerator.is_main_process:
                n = args.print_iter
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(
                    f'[iter {nb_iter:>7d}] '
                    f'loss={avg_loss["total"]/n:.4f}  '
                    f'MTC={avg_loss["mtc"]/n:.4f}  '
                    f'MTM={avg_loss["mtm"]/n:.4f}  '
                    f'MTG={avg_loss["mtg"]/n:.4f}  '
                    f'lr={lr_now:.2e}'
                )
                writer.add_scalar('loss/total', avg_loss['total']/n, nb_iter)
                writer.add_scalar('loss/mtc',   avg_loss['mtc']/n,   nb_iter)
                writer.add_scalar('loss/mtm',   avg_loss['mtm']/n,   nb_iter)
                writer.add_scalar('loss/mtg',   avg_loss['mtg']/n,   nb_iter)
                for k in avg_loss:
                    avg_loss[k] = 0.

            # 评估
            if nb_iter % args.eval_iter == 0 and accelerator.is_main_process:
                qf = qformer.module if hasattr(qformer, 'module') else qformer
                r1, mtm_acc, ppl = evaluate(
                    qf, tae_extractor, val_loader, val_cache, pad_tok_id, device, args, accelerator,
                    tae_cache=val_tae_cache)
                logger.info(
                    f'[EVAL iter {nb_iter}] '
                    f'R@1={r1:.4f}  MTM-Acc={mtm_acc:.4f}  PPL={ppl:.2f}')
                writer.add_scalar('eval/R@1',   r1,      nb_iter)
                writer.add_scalar('eval/MTM_acc', mtm_acc, nb_iter)
                writer.add_scalar('eval/PPL',   ppl,     nb_iter)

                if r1 > best_r1:
                    best_r1 = r1
                    no_improve_count = 0
                    torch.save({
                        'iter':      nb_iter,
                        'qformer':   qf.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, pjoin(out_dir, 'net_best_r1.pth'))
                    logger.info(f'  => Best R@1 checkpoint saved')
                else:
                    no_improve_count += 1
                    pat_str = f'/{args.early_stop_patience}' if args.early_stop_patience > 0 else ''
                    logger.info(f'  no improve {no_improve_count}{pat_str}')
                    if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                        logger.info(f'Early stopping triggered at iter {nb_iter} '
                                    f'(patience={args.early_stop_patience})')
                        nb_iter = args.total_iter  # 退出外层 while
                        break

            # 定期保存
            if nb_iter % args.save_iter == 0 and accelerator.is_main_process:
                qf = qformer.module if hasattr(qformer, 'module') else qformer
                torch.save({
                    'iter':      nb_iter,
                    'qformer':   qf.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, pjoin(out_dir, f'net_iter{nb_iter}.pth'))

    # 保存最终模型
    if accelerator.is_main_process:
        qf = qformer.module if hasattr(qformer, 'module') else qformer
        torch.save({
            'iter':    args.total_iter,
            'qformer': qf.state_dict(),
        }, pjoin(out_dir, 'net_last.pth'))
        logger.info('训练完成，最终模型已保存。')


if __name__ == '__main__':
    main()
