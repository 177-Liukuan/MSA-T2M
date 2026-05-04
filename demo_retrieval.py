"""
demo_retrieval.py
=================
检索演示脚本：输入任意文本，从 RAG DB 中检索最相似的 RAG Token，
并可视化对应的运动序列（骨架图）。

使用示例
--------
  # 纯文本检索
  python demo_retrieval.py \\
      --db-dir global_rag/QFormer_t2m_272_v1 \\
      --text "a person walks forward and then turns around" \\
      --topk 3

  # 若有 Q-Former + TAE，可直接用文本检索最相似 RAG Token
  python demo_retrieval.py \\
      --db-dir global_rag/QFormer_t2m_272_v1 \\
      --qformer-ckpt Experiments/QFormer_t2m_272_v1/net_best_r1.pth \\
      --tae-ckpt Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth \\
      --text "a person kicks with left leg" \\
      --topk 5 \\
      --visualize
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='全局 RAG DB 检索演示')
    p.add_argument('--db-dir',  required=True, type=str,
                   help='RAG DB 目录（含 keys.npy / values.npy / meta.json）')
    p.add_argument('--text',    required=True, type=str,
                   help='查询文本')
    p.add_argument('--topk',    default=5, type=int)
    p.add_argument('--pool',    default='mean', choices=['mean', 'flatten'],
                   help='使用 mean-pooled 或 flatten 的 RAG Token')
    p.add_argument('--t5-model-path', default='sentencet5-xxl/', type=str,
                   help='Sentence-T5 模型路径（用于编码查询文本）')
    p.add_argument('--visualize', action='store_true',
                   help='可视化检索到的运动（需要 matplotlib）')
    p.add_argument('--data-root', default='./humanml3d_272', type=str,
                   help='HumanML3D 数据根目录（可视化时加载 motion）')
    p.add_argument('--device',  default='cuda', type=str)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 文本编码（与训练时一致：使用 Sentence-T5）
# ═══════════════════════════════════════════════════════════════════════════════

def encode_query_text(text: str, t5_model_path: str, device) -> np.ndarray:
    """用 Sentence-T5 对查询文本编码，返回 L2-normalized embedding [768]."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(t5_model_path)
        model.to(device)
        model.eval()
        with torch.no_grad():
            emb = model.encode([text], convert_to_tensor=True, device=device)
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy()[0]   # [768]
    except ImportError:
        raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")


# ═══════════════════════════════════════════════════════════════════════════════
# 检索
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(query_emb: np.ndarray, keys: np.ndarray,
             values: np.ndarray, meta: list, topk: int):
    """
    余弦相似度检索。

    query_emb : [D]
    keys      : [N, D]  L2-normed
    values    : [N, Dv]

    返回 List[dict]  topk 结果
    """
    # keys 已在构建时 L2-norm，query 也 normalize
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    sims = keys @ q              # [N]
    idx  = np.argsort(sims)[::-1][:topk]

    results = []
    for rank, i in enumerate(idx):
        results.append({
            'rank':      rank + 1,
            'score':     float(sims[i]),
            'rag_token': values[i],     # [Dv]
            'meta':      meta[i],
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化（骨架图）
# ═══════════════════════════════════════════════════════════════════════════════

# HumanML3D-272 到 3D 关节点的解析可以参考 utils/eval_trans.py
# 这里使用简化版：直接从 npy 文件提取 3D 关节位置
HML_JOINT_PARENTS = [
    -1, 0, 0, 0,    # 0 pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1
     3, 3, 3,        # 4 L_Knee, 5 R_Knee, 6 Spine2
     6, 6, 6,        # 7 L_Ankle, 8 R_Ankle, 9 Spine3
     7, 8, 9,        # 10 L_Foot, 11 R_Foot, 12 Neck
    12,              # 13 Head
    12, 12,          # 14 L_Collar, 15 R_Collar  [只有22关节时没有Collar]
    14, 15,          # 16 L_Shoulder, 17 R_Shoulder
    16, 17,          # 18 L_Elbow, 19 R_Elbow
    18, 19,          # 20 L_Wrist, 21 R_Wrist
]  # 22 关节

def visualize_motion(motion_npy: np.ndarray, title: str,
                     save_path: str = None, n_frames: int = 5):
    """
    简单骨架可视化：在 n_frames 个等间距帧绘制 3D 骨架。
    motion_npy : [T, 272]  原始（未标准化）motion 数据
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print('[可视化] 未安装 matplotlib，跳过。')
        return

    # HumanML3D-272 的前 66 维是 3D 关节位置（22 关节 × 3）
    # 参考 recover_from_local_position in eval_trans.py
    # 此处简单直接取前 66 维的全局位置
    T = motion_npy.shape[0]
    # 取关节坐标：前66维 (22关节 × 3)
    joints = motion_npy[:, :66].reshape(T, 22, 3)
    # HumanML3D 坐标系：x=forward, y=up, z=right
    # 简单可视化：y=高度, x/z=地面

    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)
    fig = plt.figure(figsize=(3 * n_frames, 4))
    fig.suptitle(title, fontsize=9)

    for col, fi in enumerate(frame_indices):
        ax = fig.add_subplot(1, n_frames, col + 1, projection='3d')
        pos = joints[fi]   # [22, 3]

        # 画骨骼连线
        for j, parent in enumerate(HML_JOINT_PARENTS):
            if parent < 0:
                continue
            ax.plot(
                [pos[j, 0], pos[parent, 0]],
                [pos[j, 2], pos[parent, 2]],
                [pos[j, 1], pos[parent, 1]],
                'b-', lw=2
            )
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
                   c='red', s=20, depthshade=True)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 2])
        ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
        ax.set_title(f'Frame {fi}', fontsize=8)
        ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  [可视化] 已保存至: {save_path}')
    else:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── 加载 RAG DB ───────────────────────────────────────────────────────────
    from os.path import join as pjoin
    keys_path   = pjoin(args.db_dir, 'keys.npy')
    values_path = pjoin(args.db_dir,
                        'values_mean.npy' if args.pool == 'mean' else 'values.npy')
    meta_path   = pjoin(args.db_dir, 'meta.json')

    assert os.path.exists(keys_path),   f'DB 不存在: {keys_path}'
    assert os.path.exists(values_path), f'DB 不存在: {values_path}'
    assert os.path.exists(meta_path),   f'DB 不存在: {meta_path}'

    keys   = np.load(keys_path)    # [N, D]
    values = np.load(values_path)  # [N, Dv]
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print(f'已加载 RAG DB: {len(keys)} 条目  keys={keys.shape}  values={values.shape}')

    # ── 编码查询文本 ───────────────────────────────────────────────────────────
    print(f'\n查询文本: "{args.text}"')
    print('正在编码文本...')
    query_emb = encode_query_text(args.text, args.t5_model_path, device)

    # ── 检索 ──────────────────────────────────────────────────────────────────
    results = retrieve(query_emb, keys, values, meta, args.topk)

    print(f'\n===== Top-{args.topk} 检索结果 =====')
    for res in results:
        m = res['meta']
        print(f"\n  Rank {res['rank']}  score={res['score']:.4f}")
        print(f"    motion_id : {m['name']}")
        print(f"    caption   : {m['caption']}")
        print(f"    motion_len: {m['length']} frames")
        print(f"    RAG Token shape: {res['rag_token'].shape}  "
              f"norm={np.linalg.norm(res['rag_token']):.4f}")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    if args.visualize:
        print('\n===== 可视化 =====')
        motion_dir = pjoin(args.data_root, 'motion_data')
        mean = np.load(pjoin(args.data_root, 'mean_std', 'Mean.npy'))
        std  = np.load(pjoin(args.data_root, 'mean_std', 'Std.npy'))
        std[std < 1e-6] = 1.0

        for res in results:
            m = res['meta']
            motion_path = pjoin(motion_dir, m['name'] + '.npy')
            if not os.path.exists(motion_path):
                print(f"  [警告] motion 文件不存在: {motion_path}")
                continue
            # 加载原始（未标准化）motion
            motion = np.load(motion_path).astype(np.float32)
            # 反标准化
            motion_raw = motion * std + mean

            save_path = f"retrieval_rank{res['rank']}_{m['name']}.png"
            title = (f"Rank {res['rank']}  score={res['score']:.3f}\n"
                     f"{m['caption'][:60]}")
            visualize_motion(motion_raw, title, save_path=save_path)

    # ── 返回 RAG Token（可供下游使用）────────────────────────────────────────
    print('\n===== RAG Token 摘要 =====')
    for res in results:
        print(f"  Rank {res['rank']}  shape={res['rag_token'].shape}  "
              f"mean={res['rag_token'].mean():.4f}  std={res['rag_token'].std():.4f}")

    return results


if __name__ == '__main__':
    main()
