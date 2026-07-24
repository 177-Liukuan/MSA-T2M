"""
Demo: 验证 MSA-VAE 交集数据中 T5 文本嵌入转换是否正确

在保留原始 CLIP 数据说明与可解释解码流程的基础上，新增 T5 校验：
1) 检查 T5 文件是否存在、形状是否为 (n_frame, 768)
2) 基于 CLIP->label 查表重建每帧标签
3) 用 SentenceT5 对标签编码，得到期望的 T5 embedding
4) 与已保存的 T5 embedding 逐帧对比（cosine / L2 / max_abs）

数据来源:
  humanml3d_272/motion_data/     -> 272维动作数据 (30fps)
  humanml3d_272/texts/           -> 全局文本 (HumanML3D captions)
  humanml3d_272/clip_enc_single/ -> 帧级CLIP embedding (20fps, 512d, 来自BABEL)
  humanml3d_272/t5_enc_single/   -> 帧级T5 embedding (20fps, 768d, 由转换脚本生成)
  humanml3d_272/pca/             -> label_to_id.json + clip_embeddings.tsv (用于CLIP解码标签)
  humanml3d_272/split/train_ft.txt -> HumanML3D ∩ BABEL 交集ID列表
"""

import argparse
import json
import os
from os.path import join as pjoin

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DATA_ROOT = './humanml3d_272'
MOTION_DIR = pjoin(DATA_ROOT, 'motion_data')
TEXT_DIR = pjoin(DATA_ROOT, 'texts')
CLIP_ENC_DIR = pjoin(DATA_ROOT, 'clip_enc_single')
T5_ENC_DIR = pjoin(DATA_ROOT, 't5_enc_single')
PCA_DIR = pjoin(DATA_ROOT, 'pca')
SPLIT_FILE = pjoin(DATA_ROOT, 'split', 'train_ft.txt')


def merge_consecutive_labels(frame_labels):
    segments = []
    cur = frame_labels[0]
    start = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != cur:
            segments.append((cur, start, i - 1))
            cur = frame_labels[i]
            start = i
    segments.append((cur, start, len(frame_labels) - 1))
    return segments


def decode_clip_ids(clip_enc, label_embeddings):
    """Vectorized cosine nearest lookup: (T,512) -> ids(T,)"""
    label_norm = label_embeddings / (np.linalg.norm(label_embeddings, axis=1, keepdims=True) + 1e-8)
    clip_norm = clip_enc / (np.linalg.norm(clip_enc, axis=1, keepdims=True) + 1e-8)
    sims = clip_norm @ label_norm.T
    top_ids = np.argmax(sims, axis=1)
    top_sims = np.max(sims, axis=1)
    return top_ids, top_sims


def encode_labels_t5(unique_labels, t5_model_path, device='cuda', batch_size=16):
    model = SentenceTransformer(t5_model_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        embs = model.encode(
            unique_labels,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    embs = np.asarray(embs, dtype=np.float32)
    return {lab: embs[i] for i, lab in enumerate(unique_labels)}


def cosine_per_frame(a, b):
    """a,b: (T,D) -> cosine(T,)"""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(an * bn, axis=1)


def main(args):
    print('=' * 70)
    print('1. 加载 BABEL 标签词表（保留原CLIP说明）')
    print('=' * 70)
    with open(pjoin(PCA_DIR, 'label_to_id.json'), 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    label_embeddings = np.loadtxt(pjoin(PCA_DIR, 'clip_embeddings.tsv')).astype(np.float32)

    print(f'  标签总数:     {len(label2id)}')
    print(f'  CLIP词表形状: {label_embeddings.shape}  (N, 512)')
    print(f'  示例标签:     {list(label2id.keys())[:10]}')

    print('\n' + '=' * 70)
    print('2. 从 train_ft.txt 选取样本（需同时存在 CLIP 和 T5 文件）')
    print('=' * 70)
    with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
        ft_ids = [line.strip() for line in f if line.strip()]
    print(f'  train_ft 总数: {len(ft_ids)}')

    sample_id = None
    pick_idx = max(args.pick_index, 1)
    count = pick_idx
    for sid in ft_ids:
        mp = pjoin(MOTION_DIR, sid + '.npy')
        tp = pjoin(TEXT_DIR, sid + '.txt')
        cp = pjoin(args.clip_dir, sid + '.npy')
        t5p = pjoin(args.t5_dir, sid + '.npy')
        if os.path.exists(mp) and os.path.exists(tp) and os.path.exists(cp) and os.path.exists(t5p):
            count -= 1
            if count <= 0:
                sample_id = sid
                break

    assert sample_id is not None, 'No valid sample found with both CLIP and T5 files!'
    print(f'  选取样本 ID:  {sample_id}')

    print('\n' + '=' * 70)
    print('3. 加载运动与全局文本（保留原逻辑）')
    print('=' * 70)
    motion = np.load(pjoin(MOTION_DIR, sample_id + '.npy'))
    print(f'  运动形状:       {motion.shape}  (帧数, 特征维)')
    print(f'  估算时长:       {motion.shape[0] / 30.0:.2f} 秒 (30fps)')

    text_path = pjoin(TEXT_DIR, sample_id + '.txt')
    with open(text_path, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()

    global_captions = []
    for line in text_lines:
        parts = line.strip().split('#')
        if len(parts) >= 4:
            caption = parts[0]
            f_tag = float(parts[2]) if parts[2] != 'nan' else 0.0
            to_tag = float(parts[3]) if parts[3] != 'nan' else 0.0
            if f_tag == 0.0 and to_tag == 0.0:
                global_captions.append(caption)
    if not global_captions:
        global_captions = [text_lines[0].strip().split('#')[0]]

    print(f'  全局描述数量:   {len(global_captions)}')
    for i, cap in enumerate(global_captions[:3]):
        print(f'    [{i}] {cap}')

    print('\n' + '=' * 70)
    print('4. CLIP 局部文本解码（保留原说明）')
    print('=' * 70)
    clip_enc = np.load(pjoin(args.clip_dir, sample_id + '.npy')).astype(np.float32)
    print(f'  CLIP嵌入形状:   {clip_enc.shape}  (帧数@20fps, 512)')

    clip_ids, clip_sims = decode_clip_ids(clip_enc, label_embeddings)
    frame_labels_20 = [id2label[int(i)] for i in clip_ids]
    segs_20 = merge_consecutive_labels(frame_labels_20)

    print(f'  CLIP->标签 平均余弦: {clip_sims.mean():.6f}')
    print(f'  CLIP->标签 最小余弦: {clip_sims.min():.6f}')
    print(f'\n  解码后的局部文本段 @20fps (前20段):')
    print(f"  {'标签':<25s} {'起':<6s} {'止':<6s} {'时长':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
    for label, sf, ef in segs_20[:20]:
        print(f'  {label:<25s} {sf:<6d} {ef:<6d} {(ef - sf + 1) / 20.0:>6.2f}s')

    print('\n' + '=' * 70)
    print('5. 新增: T5 嵌入加载与形状校验')
    print('=' * 70)
    t5_enc = np.load(pjoin(args.t5_dir, sample_id + '.npy')).astype(np.float32)
    print(f'  T5嵌入形状:     {t5_enc.shape}  (帧数@20fps, 768)')

    assert t5_enc.ndim == 2 and t5_enc.shape[1] == 768, (
        f'T5 embedding shape must be (T,768), got {t5_enc.shape}'
    )
    assert t5_enc.shape[0] == clip_enc.shape[0], (
        f'Frame count mismatch: CLIP={clip_enc.shape[0]} vs T5={t5_enc.shape[0]}'
    )

    print('\n' + '=' * 70)
    print('6. 新增: 基于 CLIP->label 重建期望 T5，并逐帧对比')
    print('=' * 70)
    unique_labels = sorted(set(frame_labels_20))
    print(f'  当前样本唯一局部标签数: {len(unique_labels)}')
    print(f'  T5模型路径: {args.t5_model_path}')

    t5_by_label = encode_labels_t5(
        unique_labels=unique_labels,
        t5_model_path=args.t5_model_path,
        device=args.device,
        batch_size=args.t5_batch_size,
    )

    expected_t5 = np.stack([t5_by_label[lab] for lab in frame_labels_20], axis=0).astype(np.float32)

    cos_vals = cosine_per_frame(t5_enc, expected_t5)
    l2_vals = np.linalg.norm(t5_enc - expected_t5, axis=1)
    max_abs = np.max(np.abs(t5_enc - expected_t5), axis=1)

    print(f'  cosine mean:    {cos_vals.mean():.8f}')
    print(f'  cosine min:     {cos_vals.min():.8f}')
    print(f'  L2 mean:        {l2_vals.mean():.8f}')
    print(f'  L2 max:         {l2_vals.max():.8f}')
    print(f'  max_abs mean:   {max_abs.mean():.8f}')
    print(f'  max_abs max:    {max_abs.max():.8f}')

    fail_mask = cos_vals < args.cos_threshold
    fail_count = int(fail_mask.sum())
    print(f'  低于阈值帧数(cos < {args.cos_threshold}): {fail_count}/{len(cos_vals)}')

    if fail_count > 0:
        bad_ids = np.where(fail_mask)[0][:10]
        print('  示例异常帧(最多10个):')
        for i in bad_ids:
            print(f'    frame={i:4d} label="{frame_labels_20[i]}" cos={cos_vals[i]:.6f} l2={l2_vals[i]:.6f}')

    if args.strict and fail_count > 0:
        raise AssertionError('T5 conversion verification failed under strict mode.')

    print('\n' + '=' * 70)
    print('7. 总结')
    print('=' * 70)
    print(f'  ID:                {sample_id}')
    print(f'  运动数据:          {motion.shape}')
    print(f'  全局文本 (Global): "{global_captions[0]}"')
    print(f'  局部文本段数量:    {len(segs_20)} (由CLIP解码)')
    print('  T5转换验证:        完成')

    if fail_count == 0:
        print('  结论:              该样本的T5嵌入转换正确（逐帧一致）')
    else:
        print('  结论:              存在不一致帧，请检查转换脚本参数或模型版本')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify CLIP->T5 frame-level embedding conversion quality')
    parser.add_argument('--clip-dir', type=str, default=CLIP_ENC_DIR, help='Directory of frame-level CLIP embeddings')
    parser.add_argument('--t5-dir', type=str, default=T5_ENC_DIR, help='Directory of converted frame-level T5 embeddings')
    parser.add_argument('--t5-model-path', type=str, default='sentencet5-xxl/', help='Local SentenceT5 model path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--t5-batch-size', type=int, default=16, help='Batch size for T5 label encoding')
    parser.add_argument('--pick-index', type=int, default=1, help='Pick the N-th valid sample in train_ft')
    parser.add_argument('--cos-threshold', type=float, default=0.9999, help='Per-frame cosine threshold for pass/fail')
    parser.add_argument('--strict', action='store_true', help='Raise error if any frame falls below threshold')
    args = parser.parse_args()

    main(args)
