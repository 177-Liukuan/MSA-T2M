import os
import sys
import glob
import warnings
import argparse
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
import models.msa_vae as msa_vae
import options.option_transformer as option_trans

# 导入可视化所需模块
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# =========================================================================
# RAG 检索引擎
# =========================================================================
class RAGRetriever:
    """实时加载 h_cls 动作语义库，对输入的文本特征进行 Top-K 余弦相似度检索"""
    def __init__(self, hcls_dir, topk=3, text_embed_dim=768, device=torch.device('cuda')):
        self.topk = topk
        self.device = device
        self.text_embed_dim = text_embed_dim

        hcls_files = sorted(glob.glob(os.path.join(hcls_dir, '*.npy')))
        if len(hcls_files) == 0:
            raise RuntimeError(f'找不到检索库文件: {hcls_dir}')

        vectors = []
        for path in hcls_files:
            h = np.load(path).astype(np.float32)
            h = h.mean(axis=0) if h.ndim == 2 else h.reshape(-1)
            if h.shape[0] == self.text_embed_dim:
                vectors.append(h)

        if len(vectors) == 0:
            raise RuntimeError(f'在 {hcls_dir} 中未找到维度为 {self.text_embed_dim} 的有效特征。')

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.library = lib
        self.library_norm = lib / (lib.norm(dim=-1, keepdim=True) + 1e-6)
        print(f"[RAG] 成功加载检索库，共包含 {len(vectors)} 条动作语义先验。")

    @torch.no_grad()
    def retrieve(self, text_emb):
        q = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.matmul(q, self.library_norm.t())
        k = min(self.topk, sim.shape[1])
        top_scores, top_indices = torch.topk(sim, k=k, dim=1)
        top_hcls = self.library[top_indices]

        # 补齐逻辑
        if k < self.topk:
            pad_h = torch.zeros(text_emb.shape[0], self.topk - k, self.text_embed_dim, device=self.device)
            pad_s = torch.full((text_emb.shape[0], self.topk - k), -1e6, device=self.device)
            top_hcls = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores

# =========================================================================
# 辅助函数
# =========================================================================
def load_state_strip_module(state_dict):
    """去除 DDP 保存权重时带有的 'module.' 前缀"""
    new_state = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state[new_key] = value
    return new_state

# =========================================================================
# 参数解析 (包含修复 Namespace 丢失属性的关键逻辑)
# =========================================================================
def parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    
    # 核心 Demo 控制参数
    extra_parser.add_argument("--text", type=str, required=True, help="你想生成的动作描述")
    extra_parser.add_argument("--mode", type=str, default="pos", choices=["pos", "rot"], help="pos: 导出GIF骨架; rot: 导出272维npy")
    extra_parser.add_argument("--out_dir", type=str, default="./demo_output", help="输出结果保存路径")
    
    # RAG 与 T5 参数
    extra_parser.add_argument("--hcls_dir", type=str, default="./humanml3d_272/h_cls_latents_msa_vae/exp")
    extra_parser.add_argument("--retrieval_topk", type=int, default=3)
    extra_parser.add_argument("--cfg_scale", type=float, default=7.0)
    extra_parser.add_argument("--text_embed_dim", type=int, default=768)
    extra_parser.add_argument("--t5_model_path", type=str, default="sentencet5-xxl/")
    extra_parser.add_argument("--reject_threshold", type=float, default=0.60, help="动态拒止机制的相似度阈值")
    extra_parser.add_argument("--stop_threshold", type=float, default=0.1, help="早停L2距离阈值")
    extra_parser.add_argument("--reference_end_latent_path", type=str, default="./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5/reference_end_latent_msa_vae_t2m_272.npy")
    
    # 修复：补齐 VAE 训练时的架构参数，防止实例化报错
    extra_parser.add_argument("--trans_d_model", type=int, default=768)
    extra_parser.add_argument("--trans_nhead", type=int, default=8)
    extra_parser.add_argument("--trans_enc_layers", type=int, default=6)
    extra_parser.add_argument("--trans_dec_layers", type=int, default=6)
    extra_parser.add_argument("--trans_ff_size", type=int, default=2048)
    extra_parser.add_argument("--trans_dropout", type=float, default=0.1)
    extra_parser.add_argument("--clip_dim", type=int, default=768)

    # 1. 优先解析我们的自定义参数
    custom_args, remaining = extra_parser.parse_known_args()

    # 2. 调用原版的 args parser 解析剩余参数
    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    # 3. 【修复核心】将自定义参数全部强行注入到主 args 对象中！
    # 这解决了 AttributeError: 'Namespace' object has no attribute 'trans_d_model' 的报错
    for k, v in vars(custom_args).items():
        setattr(args, k, v)

    return args


# =========================================================================
# 主程序
# =========================================================================
def main():
    args = parse_args()
    comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[1/5] 正在加载在线文本编码器 (T5-XXL)...")
    t5_model = SentenceTransformer(args.t5_model_path)
    t5_model.eval()
    for p in t5_model.parameters():
        p.requires_grad = False
    
    # 提取输入文本的特征与空特征
    text_emb = torch.from_numpy(np.asarray(t5_model.encode([args.text]), dtype=np.float32)).to(comp_device)
    empty_text_emb = torch.from_numpy(np.asarray(t5_model.encode([""]), dtype=np.float32)).reshape(-1).to(comp_device)

    print(f"\n[2/5] 正在 RAG 库中检索与 '{args.text}' 最匹配的动作先验...")
    retriever = RAGRetriever(args.hcls_dir, topk=args.retrieval_topk, text_embed_dim=args.text_embed_dim, device=comp_device)
    top_hcls, top_scores = retriever.retrieve(text_emb)

    # ---------------------------------------------------------
    # 动态检索拒止机制 (Dynamic Retrieval Fallback)
    # ---------------------------------------------------------
    max_sim = top_scores[0, 0].item()
    if max_sim < args.reject_threshold:
        print(f"  [⚠️ 触发拒止] 最高相似度极低 ({max_sim:.3f} < {args.reject_threshold})。")
        print(f"  [🛡️ 兜底策略] 放弃无关动作先验，使用 T5-XXL 文本特征进行代理以保证生成下限！")
        # 用精准的文本特征强行冒充检索特征，消除垃圾检索的误导
        top_hcls = text_emb.unsqueeze(1).expand(-1, args.retrieval_topk, -1)
        top_scores = torch.ones_like(top_scores) * 10.0 
    else:
        print(f"  -> 检索成功！最高相似度为: {max_sim:.3f}")

    print("\n[3/5] 正在加载生成模型 (MSA-VAE & LLaMA-RAG)...")
    # 初始化 VAE
    clip_range = [-30, 20]
    net = msa_vae.MSA_HumanVAE(
        hidden_size=args.hidden_size, down_t=args.down_t, stride_t=args.stride_t, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate, activation='relu', latent_dim=args.latent_dim,
        clip_range=clip_range, trans_d_model=args.trans_d_model, trans_nhead=args.trans_nhead,
        trans_enc_layers=args.trans_enc_layers, trans_dec_layers=args.trans_dec_layers,
        trans_ff_size=args.trans_ff_size, trans_dropout=args.trans_dropout, clip_dim=args.clip_dim
    )
    
    ckpt_vae = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt_vae['net'] if 'net' in ckpt_vae else ckpt_vae, strict=True)
    net.eval().to(comp_device)

    # 初始化 LLaMA-RAG
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    rag_model = LLaMARAGWrapper(base_model=base_model, model_dim=config.n_embd)
    
    ckpt_trans = torch.load(args.resume_trans, map_location='cpu')
    rag_model.base_model.load_state_dict(load_state_strip_module(ckpt_trans['trans']), strict=False)
    rag_model.load_state_dict(load_state_strip_module(ckpt_trans['rag']), strict=False)
    rag_model.eval().to(comp_device)

    # 加载早停参考 Token
    if os.path.exists(args.reference_end_latent_path):
        reference_end_latent = torch.from_numpy(np.load(args.reference_end_latent_path).astype(np.float32)).reshape(-1).to(comp_device)
    else:
        reference_end_latent = None
        print(f"[警告] 未找到早停潜变量文件 ({args.reference_end_latent_path})，将生成固定最大帧数。")

    print(f"\n[4/5] 🚀 开始自回归生成动作 (CFG Scale: {args.cfg_scale})...")
    max_tokens = 196 // 4
    xs = None

    with torch.no_grad():
        for k in range(max_tokens):
            prefix = torch.zeros((1, 0, args.latent_dim), device=comp_device) if xs is None else xs
            
            # 单步自回归预测
            next_token = rag_model.sample_next_with_cfg(
                motion_prefix=prefix,
                text_emb=text_emb,
                top3_h_cls=top_hcls,
                top3_sim_scores=top_scores,
                empty_text_emb=empty_text_emb,
                cfg_scale=args.cfg_scale,
                temperature=1.0
            )
            
            next_token = next_token.unsqueeze(1)
            xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

            # 早停判定 (跳过前5步，防止模型开头不稳定直接停掉)
            if reference_end_latent is not None and k > 5:
                dist = torch.linalg.norm(next_token.squeeze() - reference_end_latent)
                if dist < args.stop_threshold:
                    print(f"  -> 动作已在第 {k * 4} 帧自然结束 (L2 Dist: {dist:.3f})。")
                    break

        print("\n[5/5] 正在解码生成的三维运动特征...")
        motion_seqs = net.forward_decoder(xs)
        motion_272 = motion_seqs.squeeze(0).cpu().numpy()

    # 安全的文件名截断
    safe_text = args.text.replace(" ", "_").replace("/", "").replace("'", "")[:50]
    
    if args.mode == 'pos':
        # 提取位置并在当前脚本内直接画出 GIF
        print("  -> 正在渲染 3D 骨架 GIF 动图...")
        mean = np.load('./humanml3d_272/mean_std/Mean.npy')
        std = np.load('./humanml3d_272/mean_std/Std.npy')
        
        # 反归一化并转为 XYZ 坐标
        pred_xyz = recover_from_local_position(motion_272 * std + mean, 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        
        out_gif = os.path.join(args.out_dir, f'{safe_text}.gif')
        # plot_3d.draw_to_batch 底层使用 matplotlib 保存动画，传入 .gif 后缀会自动存为 GIF
        plot_3d.draw_to_batch(xyz, [args.text], [out_gif], fps=30)
        print(f"🎉 成功！GIF 动图已保存至: {out_gif}")

    elif args.mode == 'rot':
        # 保存 272维旋转特征，供后续使用 SMPL/aitviewer 渲染真实三维蒙皮
        out_npy = os.path.join(args.out_dir, f'{safe_text}_272.npy')
        np.save(out_npy, motion_272)
        print(f"🎉 成功！特征序列已保存至: {out_npy}")
        print("\n👉 若需渲染高保真 SMPL 实体网格，请执行我们专门修复好的 aitviewer 脚本:")
        print(f"python render_smpl_aitviewer_rot.py --motion_file {out_npy} --format mp4")
        
    else:
        raise ValueError(f"无效的 mode 参数: {args.mode}")

if __name__ == '__main__':
    main()