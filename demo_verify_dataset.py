"""
Demo: 验证 MSA-VAE 数据集构建是否正确
加载一条运动及其对应的全局文本(HumanML3D caption)和局部文本(BABEL 帧级标签)

数据来源:
  humanml3d_272/motion_data/     -> 272维动作数据 (30fps)
  humanml3d_272/texts/           -> 全局文本 (HumanML3D captions)
  humanml3d_272/clip_enc_single/ -> 帧级CLIP embedding (20fps, 512d, 来自BABEL)
  humanml3d_272/pca/             -> label_to_id.json + clip_embeddings.tsv (用于解码标签)
  humanml3d_272/split/train_ft.txt -> HumanML3D ∩ BABEL 交集ID列表
"""

import os
import json
import numpy as np
from os.path import join as pjoin

DATA_ROOT = './humanml3d_272'
MOTION_DIR = pjoin(DATA_ROOT, 'motion_data')
TEXT_DIR = pjoin(DATA_ROOT, 'texts')
CLIP_ENC_DIR = pjoin(DATA_ROOT, 'clip_enc_single')
PCA_DIR = pjoin(DATA_ROOT, 'pca')
SPLIT_FILE = pjoin(DATA_ROOT, 'split', 'train_ft.txt')

# 1. 加载 BABEL 标签词表和 CLIP 嵌入
print("=" * 70)
print("1. 加载 BABEL 标签词表")
print("=" * 70)
with open(pjoin(PCA_DIR, 'label_to_id.json'), 'r') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
label_embeddings = np.loadtxt(pjoin(PCA_DIR, 'clip_embeddings.tsv'), delimiter='\t')
print(f"  标签总数:     {len(label2id)}")
print(f"  CLIP嵌入维度: {label_embeddings.shape}")
print(f"  示例标签:     {list(label2id.keys())[:10]}")


def decode_clip_to_label(clip_vec, label_emb, id2lab, top_k=1):
    norms_db = np.linalg.norm(label_emb, axis=1) + 1e-8
    norm_q = np.linalg.norm(clip_vec) + 1e-8
    sims = (label_emb @ clip_vec) / (norms_db * norm_q)
    top_ids = np.argsort(sims)[-top_k:][::-1]
    return [(id2lab[i], float(sims[i])) for i in top_ids]


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


# 2. 选取样本
print("\n" + "=" * 70)
print("2. 从 train_ft.txt (HumanML3D ∩ BABEL 交集) 中选取样本")
print("=" * 70)
with open(SPLIT_FILE, 'r') as f:
    ft_ids = [line.strip() for line in f if line.strip()]
print(f"  train_ft 总数: {len(ft_ids)}")

sample_id = None
count = 5
for sid in ft_ids:
    mp = pjoin(MOTION_DIR, sid + '.npy')
    tp = pjoin(TEXT_DIR, sid + '.txt')
    cp = pjoin(CLIP_ENC_DIR, sid + '.npy')
    if os.path.exists(mp) and os.path.exists(tp) and os.path.exists(cp):
        count = count - 1
        if count <= 0:
            sample_id = sid
            break
assert sample_id, "No valid sample found!"
print(f"  选取样本 ID:  {sample_id}")


# 3. 加载运动数据
print("\n" + "=" * 70)
print("3. 加载运动数据 (272维, 30fps)")
print("=" * 70)
motion = np.load(pjoin(MOTION_DIR, sample_id + '.npy'))
print(f"  运动形状:       {motion.shape}  (帧数, 特征维)")
print(f"  估算时长:       {motion.shape[0] / 30.0:.2f} 秒 (30fps)")


# 4. 加载全局文本
print("\n" + "=" * 70)
print("4. 加载全局文本 (HumanML3D caption)")
print("=" * 70)
text_path = pjoin(TEXT_DIR, sample_id + '.txt')
with open(text_path, 'r') as f:
    text_lines = f.readlines()

global_captions = []
sub_segments = []
for line in text_lines:
    parts = line.strip().split('#')
    if len(parts) >= 4:
        caption = parts[0]
        f_tag = float(parts[2]) if parts[2] != 'nan' else 0.0
        to_tag = float(parts[3]) if parts[3] != 'nan' else 0.0
        if f_tag == 0.0 and to_tag == 0.0:
            global_captions.append(caption)
        else:
            sub_segments.append({'caption': caption, 'from': f_tag, 'to': to_tag})

print(f"  全局描述数量:   {len(global_captions)}")
for i, cap in enumerate(global_captions):
    print(f"    [{i}] {cap}")
if sub_segments:
    print(f"  子片段描述数量: {len(sub_segments)}")
    for seg in sub_segments[:3]:
        print(f"    \"{seg['caption']}\"  [{seg['from']:.1f}s - {seg['to']:.1f}s]")


# 5. 加载局部文本 (CLIP -> decode)
print("\n" + "=" * 70)
print("5. 加载局部文本 (BABEL frame-level CLIP embeddings)")
print("=" * 70)
clip_enc = np.load(pjoin(CLIP_ENC_DIR, sample_id + '.npy'))
print(f"  CLIP嵌入形状:   {clip_enc.shape}  (帧数@20fps, 512)")
print(f"  运动帧数@30fps: {motion.shape[0]}")
print(f"  帧率比:         {motion.shape[0] / clip_enc.shape[0]:.4f}  (约1.5)")

frame_labels_20 = []
for i in range(clip_enc.shape[0]):
    top = decode_clip_to_label(clip_enc[i], label_embeddings, id2label)
    frame_labels_20.append(top[0][0])

segs_20 = merge_consecutive_labels(frame_labels_20)
print(f"\n  解码后的局部文本段 @20fps:")
print(f"  {'标签':<25s} {'起':<6s} {'止':<6s} {'时长':>6s}")
print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
for label, sf, ef in segs_20:
    print(f"  {label:<25s} {sf:<6d} {ef:<6d} {(ef-sf+1)/20.0:>6.2f}s")


# 6. 帧率对齐
print("\n" + "=" * 70)
print("6. 帧率对齐: 20fps CLIP嵌入 -> 30fps")
print("=" * 70)
T_30 = motion.shape[0]
T_20 = clip_enc.shape[0]
indices = np.round(np.linspace(0, T_20 - 1, T_30)).astype(int)
clip_enc_30 = clip_enc[indices]
print(f"  上采样后形状:   {clip_enc_30.shape}  (与运动帧数一致)")

frame_labels_30 = []
for i in range(T_30):
    top = decode_clip_to_label(clip_enc_30[i], label_embeddings, id2label)
    frame_labels_30.append(top[0][0])

segs_30 = merge_consecutive_labels(frame_labels_30)
print(f"\n  上采样后的局部文本段 @30fps:")
print(f"  {'标签':<25s} {'起':<6s} {'止':<6s} {'时长':>6s}")
print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
for label, sf, ef in segs_30:
    print(f"  {label:<25s} {sf:<6d} {ef:<6d} {(ef-sf+1)/30.0:>6.2f}s")


# 7. 总结
print("\n" + "=" * 70)
print("7. MSA-VAE 训练样本总结")
print("=" * 70)
print(f"  ID:                {sample_id}")
print(f"  运动数据:          {motion.shape}")
print(f"  全局文本 (Global): \"{global_captions[0]}\"")
print(f"  局部文本 (Local):  {len(segs_30)} 个动作段:")
for label, sf, ef in segs_30:
    print(f"    [{sf:3d}-{ef:3d}] \"{label}\"")
print(f"\n  MSA-VAE 对齐用途:")
gc = global_captions[0][:50]
print(f"    L_global_align: h_cls  <-> CLIP(\"{gc}...\")")
print(f"    L_local_align:  z_i    <-> clip_enc_single[i] (逐token对齐)")
print(f"\n  数据集构建验证完成!")
